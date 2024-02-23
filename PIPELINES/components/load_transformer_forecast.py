from kfp.components import InputPath, OutputPath

def LoadAndForecastTransformer(input_data_path: InputPath(str),input_weather_path: InputPath(str),
    diff_time,
    num_days,
    url_minio,
    access_key,
    secret_key,
    pilot_name,
    measurement_name,
    asset_name,
    forecast_data_path: OutputPath(str),
    results_path: OutputPath(str)
):
    # LIBRARIES REQUIRED
    import pandas as pd
    import json
    import maya
    import numpy as np
    from minio import Minio

    # FUNCTIONS
    from tqdm import tqdm
    from darts.models import TransformerModel
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.utils.missing_values import fill_missing_values

    def PredictFromTransformer(model, data,
                        weather_data,
                        diff_time,
                        days_ahead=1):

        """
        Takes a Transformer model and makes the prediction.

        Parameters
        ----------
        model : TransformerModel
            The trained Transformer model.
        data : pandas.DataFrame
            The input data in the format ["ds", "y"], where "ds" is a column of datetime or string in the format "%Y-%m-%d %H:%M:%S".
            This dataset is used as input for the model and must have a length greater than or equal to the input_chunk_length specified in the TransformerModel.
            Forecasts start following the end of this dataset.
        weather_data : pandas.DataFrame
            The weather data used as covariates for the model. It should have a column named "ds" representing the time and other columns representing weather features.
        diff_time : int
            The time in minutes between two consecutive measurements. Missing values will be interpolated.
        days_ahead : int, optional
            The number of days to forecast. Default is 1.

        Returns
        -------
        forecast : pandas.DataFrame
            Dataframe of the forecasted values with columns "ds" and "yhat_transformer"
            ds: string in format "%Y-%m-%d %H:%M:%S", yhat_transformer: float
        """

        measures_per_hour = int(60 / int(diff_time))

        input_series = TimeSeries.from_dataframe(
            data, 'ds', 'y',fill_missing_dates=True, freq="{minutes}T".format(minutes=diff_time)
        )
        input_series = fill_missing_values(input_series)

        transformer = Scaler()
        input_series_transformed = transformer.fit_transform(input_series)

        try:
            weather_series = TimeSeries.from_dataframe(
                weather_data, time_col='ds', fill_missing_dates=True, freq="{minutes}T".format(minutes=diff_time)
            )
            weather_series = fill_missing_values(weather_series)

            weather_transformer = Scaler()
            weather_transformed = weather_transformer.fit_transform(weather_series)
        except:
            weather_transformed = None

        forecast = model.predict(
            n=24 * days_ahead * measures_per_hour, seriess=input_series_transformed, past_covariates=weather_transformed
        )
        forecast = transformer.inverse_transform(forecast)
        forecast = forecast.pd_dataframe().reset_index()
        forecast = forecast[['ds', 'y']]
        forecast.columns = ['ds', 'yhat_transformer']
        forecast['ds'] = forecast['ds'].apply(str)

        return forecast

    def DownloadModel(url_minio,
                      access_key,
                      secret_key,
                      pilot_name,
                      measurement_name,
                      asset_name):

        client = Minio(
            url_minio,
            access_key=access_key,
            secret_key=secret_key,
        )

        bucket_name = "{pilot_name}-{measurement}-{asset}".format(
            pilot_name = pilot_name.lower().replace("_", "-"),
            measurement = measurement_name.lower().replace("_", "-"),
            asset = asset_name.lower().replace("_","-")
        )

        client.fget_object(bucket_name,
                        "asset_transformers_config.json",
                        file_path = "transformers_config.json")

        with open("transformers_config.json") as file:
            config_ = json.load(file)
        
        model_name = config_["model_name"]

        client.fget_object(bucket_name,
                        model_name,
                        file_path = "transformers_model.pt")
        client.fget_object(bucket_name,
                        model_name + ".ckpt",
                        file_path = "transformers_model.pt.ckpt")

        return model_name

    weather_data = pd.read_feather(input_weather_path)
    model_name = DownloadModel(
        url_minio,
        access_key,
        secret_key,
        pilot_name,
        measurement_name,
        asset_name
    )
    transformer_model = TransformerModel.load("transformers_model.pt")  # Load model

    with open(input_data_path) as file:
        data_str = json.load(file)

    data = pd.DataFrame(data_str)
    data = data[data.asset_name == asset_name]
    data["ds"] = pd.to_datetime(data["ds"])
    data = data.set_index("ds").asfreq("{minutes}T".format(minutes=diff_time)).reset_index()

    print("The data we are using to predict is:\n")
    print(data.head())

    forecast_ = PredictFromTransformer(transformer_model, data, weather_data, diff_time, days_ahead=num_days)

    forecast_.to_feather(forecast_data_path)

    print("Model {model_name}".format(model_name = model_name))

    results_dict = {
        "model_name": model_name
    }

    with open(results_path, "w") as file:
        json.dump(results_dict, file)