from kfp.components import InputPath, OutputPath

def LoadAndForecastProphet(input_weather_path: InputPath(str),
    diff_time,
    num_days,
    path_minio,
    access_key,
    secret_key,
    pilot_name,
    measurement_name,
    asset_name,
#    mlpipeline_metrics_path: OutputPath('Metrics'),
    forecast_data_path: OutputPath(str),
    results_path: OutputPath(str)
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
    from minio import Minio
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
            pilot_name = pilot_name,
            measurement = measurement_name,
            asset = asset_name
        )


        client.fget_object(bucket_name,
                        "asset_prophet_config.json",
                        file_path = "prophet_config.json")

        with open("prophet_config.json") as file:
            config_ = json.load(file)
        
        model_name = config_["model_name"]

        client.fget_object(bucket_name,
                        model_name,
                        file_path = "prophet_model.json")

        return model_name


    weather_data = pd.read_feather(input_weather_path)

    model_name = DownloadModel(
        path_minio,
        access_key,
        secret_key,
        pilot_name,
        measurement_name,
        asset_name
    )
    
    with open("prophet_model.json", 'r') as fin:
        prophet_model = model_from_json(fin.read())  # Load model
    
    latest_date = prophet_model.make_future_dataframe(periods = 1, freq = "1H", include_history= False)
    latest_date = latest_date["ds"].tolist()[0]

    num_days_extra = (maya.now() - maya.MayaDT(latest_date)).days

    num_days = num_days_extra + num_days

    forecast_ = PredictFromProphet(prophet_model, weather_data, freq_hourly = 60/ diff_time, days_ahead = num_days)

    forecast_.to_feather(forecast_data_path)

    print("Model {model_name} used for test".format(model_name = model_name))

    results_dict = {
        "model_name": model_name
    }

    with open(results_path, "w"):
        json.dump(results_dict, results_path)






