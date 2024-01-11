from kfp.components import InputPath, OutputPath

# TODO don't forget the input_data_path and asset_name arguments in forecast_data_pipeline
def LoadAndForecastLSTM(model_saved_path: InputPath(str), input_data_path: InputPath(str),
    input_weather_path: InputPath(str),
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
    from darts.models import RNNModel
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.utils.missing_values import fill_missing_values
    from darts.models import RNNModel

    def PredictFromLSTM(model, data,
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
        forecast['ds'] = forecast['ds'].apply(str) # TODO check if 'ds' column is datetime type (and converting to string is necessary and possible?)

        return forecast


    weather_data = pd.read_feather(input_weather_path)
    lstm_model = RNNModel.load(model_saved_path)  # Load model
    
    with open(input_data_path) as file:
        data_str = json.load(file)
    data = pd.DataFrame(data_str)
    data = data[data.asset_name == asset_name]
    data["ds"] = pd.to_datetime(data["ds"])
    data = data.set_index("ds").asfreq("{minutes}T".format(minutes=diff_time)).reset_index()

    forecast_ = PredictFromLSTM(lstm_model, data, weather_data, diff_time, days_ahead=num_days)

    forecast_.to_feather(forecast_data_path)
