from kfp.components import InputPath, OutputPath

def LoadAndForecastTransformer(model_saved_path: InputPath(str), input_data_path: InputPath(str),
    diff_time,
    num_days,
    asset_name,
    forecast_data_path: OutputPath(str),
):
    # LIBRARIES REQUIRED
    import pandas as pd
    import json
    import maya
    import numpy as np

    # FUNCTIONS
    from tqdm import tqdm
    from darts.models import TransformerModel
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.utils.missing_values import fill_missing_values

    def PredictFromTransformer(model, data,
                        diff_time,
                        days_ahead=1):

        """
        Takes a Transformer model and makes the prediction.

        Parameters
        ----------
        data: Data in the format ["ds", "y"], where ds is a columns in datetime or string = "%Y-%m-%d %H:%M:%S.
                this dataset is used as input for the model; must have length >= input_chunk_length specified in TransformerModel; forecasts start following the end of this dataset
        diff_time: time in minutes between two consecutive measurements; missing values will be interpolated
        days_ahead: number of days to forecast

        Returns
        -------
        forecast: dataframe with columns "ds" and "yhat_transformer" - ds: string in format "%Y-%m-%d %H:%M:%S", yhat_transformer: float
        """

        measures_per_hour = int(60/int(diff_time))

        input_series = TimeSeries.from_dataframe(data, 'ds', 'y',fill_missing_dates=True, freq="{minutes}T".format(minutes = diff_time))
        input_series = fill_missing_values(input_series)

        transformer = Scaler()
        input_series_transformed = transformer.fit_transform(input_series)

        forecast = model.predict(n=24*days_ahead*measures_per_hour, series = input_series_transformed)
        forecast = transformer.inverse_transform(forecast)
        forecast = forecast.pd_dataframe().reset_index()
        forecast = forecast[['ds', 'y']]
        forecast.columns = ['ds', 'yhat_transformer']
        forecast['ds'] = forecast['ds'].apply(str)

        return forecast

    transformer_model = TransformerModel.load(model_saved_path)  # Load model

    with open(input_data_path) as file:
        data_str = json.load(file)

    data = pd.DataFrame(data_str)
    data = data[data.asset_name == asset_name]
    data["ds"] = pd.to_datetime(data["ds"])
    data = data.set_index("ds").asfreq("{minutes}T".format(minutes=diff_time)).reset_index()

    print("The data we are using to predict is:\n")
    print(data.head())

    forecast_ = PredictFromTransformer(transformer_model, data, diff_time, days_ahead=num_days)

    forecast_.to_feather(forecast_data_path)
