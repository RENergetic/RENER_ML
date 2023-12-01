from kfp.components import InputPath, OutputPath

def LoadAndForecastProphet(model_saved_path: InputPath(str), input_weather_path: InputPath(str),
    diff_time,
    num_days,
#    mlpipeline_metrics_path: OutputPath('Metrics'),
    forecast_data_path: OutputPath(str),
    ):

    # LIBRARIES REQUIRED    

    import pandas as pd
    import json
    import maya
    import numpy as np

    # FUNCTIONS

    from prophet import Prophet
    import itertools
    from tqdm import tqdm
    from prophet.diagnostics import cross_validation
    from prophet.diagnostics import performance_metrics
    from prophet.serialize import model_to_json, model_from_json


    def PredictFromProphet(model, 
                            weather_data,
                            freq_hourly, 
                            days_ahead = 1):

        """
        Takes a Prophet model and makes the prediction. It

        """

        if freq_hourly <= 1:
            freq_of_hours = np.round(1/freq,0)
            freq = "{num_hours}H".format(num_hours = freq_of_hours)
            periods = np.round(days_ahead*24 / freq_of_hours,0)
        else:
            freq_in_minutes = np.round(60/freq,0)
            freq = "{num_minutes}T".format(num_minutes = freq_in_minutes)
            periods = np.round(days_ahead*24*60 / freq_in_minutes,0)
            
        future = model.make_future_dataframe(periods = periods, freq = freq, include_history = False)
        future["ds"] = future["ds"].apply(str)
        future = pd.merge(future, weather_data, on = "ds")
        forecast = model.predict(future)

        return forecast[["ds", "yhat"]]


    weather_data = pd.read_feather(input_weather_path)
    with open(model_saved_path, 'r') as fin:
        prophet_model = model_from_json(fin.read())  # Load model
    
    latest_date = prophet_model.make_future_dataframe(periods = 1, freq = "1H", include_history= False)
    latest_date = latest_date["ds"].tolist()[0]

    num_days_extra = (maya.now() - maya.MayaDT(latest_date)).days

    num_days = num_days_extra + num_days

    forecast_ = PredictFromProphet(prophet_model, weather_data, freq_hourly = 60/ diff_time, days_ahead = num_days)

    forecast_.to_feather(forecast_data_path)






