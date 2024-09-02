from kfp.components import InputPath, OutputPath

def Forecast(
        input_data_path: InputPath(str), input_weather_path: InputPath(str),
        config_model_path: InputPath(str),
        diff_time, num_days,
        pilot_name, measurement_name, asset_name,
        url_minio, access_key, secret_key,
        forecast_data_path: OutputPath(str)
):
    
    # LIBRARIES REQUIRED    

    import pandas as pd
    import json
    import numpy as np
    from minio import Minio
    import maya

    from prophet.serialize import model_from_json
    from darts.models import RNNModel
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.utils.missing_values import fill_missing_values
    from darts.models import TransformerModel
    
    def PredictFromProphet(model, 
                            weather_data,
                            freq_hourly, 
                            days_ahead:int = 1):

        """
        Takes a Prophet model and makes the prediction. It

        """

        print(type(days_ahead))
        print(days_ahead)

        if freq_hourly <= 1:
            freq_of_hours = np.round(1/freq_hourly,0)
            freq = "{num_hours}H".format(num_hours = freq_of_hours)
            periods = np.round(days_ahead*24 / freq_of_hours,0)
        else:
            freq_in_minutes = np.round(60/freq_hourly,0)
            freq = "{num_minutes}T".format(num_minutes = freq_in_minutes)
            periods = np.round(days_ahead*24*60 / freq_in_minutes,0)
            
        future = model.make_future_dataframe(periods = int(periods), freq = freq, include_history = False)
        future["ds"] = future["ds"].apply(str)
        future = pd.merge(future, weather_data, on = "ds")
        forecast = model.predict(future)
        forecast = forecast[["ds", "yhat"]]
        forecast.columns = ["ds", "yhat_prophet"]
        forecast["ds"] = forecast["ds"].apply(str)
        return forecast
    
    def PredictFromLSTM(model, data, weather_data,
                        diff_time,
                        days_ahead = 1):

        """
        Takes a LSTM model and makes the prediction.

        Parameters
        ----------
        data: Data in the format ["ds", "y"], where ds is a columns in datetime or string = "%Y-%m-%d %H:%M:%S. 
                this dataset is used as input for the model; must have length >= input_chunk_length specified in RNNModel; forecasts start following the end of this dataset
        diff_time: time in minutes between two consecutive measurements; missing values will be interpolated
            
        Returns
        -------
        forecast: The model trained

        """

        measures_per_hour = int(60/int(diff_time))

        input_series = TimeSeries.from_dataframe(data, 'ds', 'y',fill_missing_dates=True, freq="{minutes}T".format(minutes = diff_time))
        input_series = fill_missing_values(input_series)
        transformer = Scaler()
        input_series_transformed = transformer.fit_transform(input_series)

        # TODO check if weather data covers enough time to be used as future covariates in RNNModel
        try:
            weather_series = TimeSeries.from_dataframe(
                weather_data, time_col='ds', fill_missing_dates=True, freq="{minutes}T".format(minutes=diff_time)
            )
            weather_series = fill_missing_values(weather_series)

            weather_transformer = Scaler()
            weather_transformed = weather_transformer.fit_transform(weather_series)
        except:
            weather_transformed = None

        forecast = model.predict(n=days_ahead*24*measures_per_hour, series = input_series_transformed, future_covariates=weather_transformed)
        forecast = transformer.inverse_transform(forecast)
        forecast = forecast.pd_dataframe().reset_index()
        forecast = forecast[['ds', 'y']]
        forecast.columns = ['ds', 'yhat_lstm']
        forecast['ds'] = forecast['ds'].apply(str)

        return forecast

    def PredictFromTransformer(model, data, weather_data, diff_time, days_ahead=1):
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
            data, 'ds', 'y', fill_missing_dates=True, freq="{minutes}T".format(minutes=diff_time)
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
            n=24 * days_ahead * measures_per_hour, series=input_series_transformed, past_covariates=weather_transformed
        )
        forecast = transformer.inverse_transform(forecast)
        forecast = forecast.pd_dataframe().reset_index()
        forecast = forecast[['ds', 'y']]
        forecast.columns = ['ds', 'yhat_transformer']
        forecast['ds'] = forecast['ds'].apply(str)

        return forecast

    def DownloadModel(client, model_name, type_model):

        if type_model == "prophet":
            client.fget_object(bucket_name,
                        model_name,
                        file_path = "model.json")
        else:
            client.fget_object(bucket_name,
                        model_name,
                        file_path = "model.pt")
            client.fget_object(bucket_name,
                            model_name + ".ckpt",
                            file_path = "model.pt.ckpt")

    # BUCKET CONFIG

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

    print(bucket_name)

    # DATA READ

    with open(config_model_path) as file:
        dict_model = json.load(file)
    
    
    with open(input_data_path) as file:
        data_str = json.load(file)
    
    data = pd.DataFrame(data_str)

    freq_hourly = 60/int(diff_time)

    data = data[data.asset_name == asset_name]

    weather_data = pd.read_feather(input_weather_path)

    try:
        type_model = dict_model["type_model"]
        name_model = dict_model["model_name"]

        DownloadModel(client, name_model, type_model)

        if type_model == "prophet":
            with open("prophet_model.json", 'r') as fin:
                prophet_model = model_from_json(fin.read())
            latest_date = prophet_model.make_future_dataframe(periods = 1, freq = "1H", include_history= False)
            latest_date = latest_date["ds"].tolist()[0]

            num_days_extra = (maya.now() - maya.MayaDT(latest_date)).days

            num_days = num_days_extra + num_days

            forecast_ = PredictFromProphet(prophet_model, weather_data, freq_hourly = 60/ diff_time, days_ahead = num_days)

            forecast_.to_feather(forecast_data_path)

        elif type_model == "lstm":

            lstm_model = RNNModel.load("model.pt")  # Load model
            data["ds"] = pd.to_datetime(data["ds"])
            data = data.set_index("ds").asfreq("{minutes}T".format(minutes=diff_time)).reset_index()

            forecast_ = PredictFromLSTM(lstm_model, data, weather_data, diff_time, days_ahead=num_days)

            forecast_.reset_index().to_feather(forecast_data_path)
        
        elif type_model == "transformers":
            
            transformer_model = TransformerModel.load("transformers_model.pt")  # Load model

            data["ds"] = pd.to_datetime(data["ds"])
            data = data.set_index("ds").asfreq("{minutes}T".format(minutes=diff_time)).reset_index()

            print("The data we are using to predict is:\n")
            print(data.head())

            forecast_ = PredictFromTransformer(transformer_model, data, weather_data, diff_time, days_ahead=num_days)

            forecast_.reset_index().to_feather(forecast_data_path)



    except KeyError:
        print("No model is set for this asset in this measurement")
