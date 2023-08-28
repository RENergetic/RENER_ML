from kfp.components import InputPath, OutputPath

def PredictFromPreviousModel(input_data_path:InputPath(str), input_weather_path: InputPath(str),
        name_pilot, measurement_name, asset_name, name_model, max_date, num_days, diff_time,
        forecast_data_path: OutputPath(str)):

    import maya
    import json
    from icecream import ic
    import requests
    import pandas as pd
    from prophet.serialize import model_to_json, model_from_json
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from minio import Minio
    import boto3
    from tqdm import tqdm
    from sklearn.metrics import mean_absolute_error, r2_score
    from datetime import datetime

    def ManageDateHour(ds_obj):
            return datetime.strftime(maya.parse(ds_obj).datetime(), "%Y-%m-%d %H:00:00")

    def ModifyData(data, asset_name):
        data_ds = data[data.asset_name == asset_name][["time_registered", "value"]]
        try:
            last_cummulative_value = data[data.asset_name == asset_name]["value_cummulative"].tolist()[-1]
        except:
            last_cummulative_value = 0
        data_ds.columns = ["ds", "y"]
        if data_ds.shape[0] == 0:
            max_date = datetime.strftime(maya.when("now").datetime(),"%Y-%m-%d %H:%M:%S")
        else:
            max_date = max(data_ds.ds)
        ic(max_date)
        ic(last_cummulative_value)
        ic(data_ds.shape)
        ic(len(pd.unique(data_ds.y)))

        return data_ds, max_date, last_cummulative_value

    s3 = boto3.resource(
        service_name='s3',
        aws_access_key_id='QyvycO9kc2cm58K8',
        aws_secret_access_key='tKtUrdQzQgWfhfBwhbQF3yGbyZ43oPn92iGAT7g0',
        endpoint_url='https://s3.tebi.io'
    )

    s3_client = boto3.client('s3',
        aws_access_key_id='QyvycO9kc2cm58K8',
        aws_secret_access_key='tKtUrdQzQgWfhfBwhbQF3yGbyZ43oPn92iGAT7g0',
        endpoint_url='https://s3.tebi.io'
    )

    # model_Ghent_pv_de-nieuwe-dokken-pv-017A-xxxxx9A1.json
    with open(input_data_path) as file:
        data_str = json.load(file)
    
    data = pd.DataFrame(data_str)
    data, max_date, last_cummulative_value = ModifyData(data, asset_name)
    metrics_list = []
    ic(asset_name)

    weather_data = pd.read_feather(input_weather_path)
    
    my_bucket = s3.Bucket('test-pf')
    list_objects = []
    for my_bucket_object in my_bucket.objects.all():
        list_objects.append(my_bucket_object.key)
    
    try:
        file_stats = "stats_{name_pilot}_{measurement_name}_{asset_name}.json".format(
            name_pilot = name_pilot,measurement_name = measurement_name, asset_name = asset_name
        )
        
        with open(file_stats, 'wb') as f:
            s3_client.download_fileobj('test-pf', file_stats, f)
        
        with open(file_stats) as f:
            stats_asset_models = json.load(f)
        
        if "catboost" in stats_asset_models.keys():
            last_date_catboost = stats_asset_models["catboost"]["last_update_date"]
        else:
            last_date_catboost = datetime.strftime(maya.when("1 Jan 1970").datetime(), format = "%Y-%m-%d %H:%M:%S")
        
        if "prophet" in stats_asset_models.keys():
            last_date_prophet = stats_asset_models["prophet"]["last_update_date"]
        else:
            last_date_prophet = datetime.strftime(maya.when("1 Jan 1970").datetime(), format = "%Y-%m-%d %H:%M:%S")
        
        if last_date_prophet == last_date_catboost:
            raise ValueError("No models trained")
        elif last_date_catboost > last_date_prophet:
            type_ = "catboost"
        else:
            type_ = "prophet"

    except:
        return False
    
    name_model = "model_{name_pilot}_{measurement_name}_{asset_name}_{type_}".format(
        name_pilot = name_pilot,measurement_name = measurement_name, asset_name = asset_name, type_ = type_
    )

    if type_ == "prophet":
        with open('model_prophet.json', 'r') as fin:
            m = model_from_json(fin.read())

        from_date_obj = maya.parse(last_date_prophet).add(days = -2)
        to_date_obj = maya.when(max_date).add(days = num_days)
        days_forecast = (to_date_obj - from_date_obj).days
        measures_per_hour = 60/diff_time
        future = m.make_future_dataframe(periods= 24*(3 + days_forecast)*measures_per_hour , freq="{minutes}T".format(minutes = diff_time))
        future["ds"] = future["ds"].apply(str)
        future["ds_hour"] = future["ds"].apply(ManageDateHour)
        future = pd.merge(future, weather_data, on = "ds_hour")

        forecast = m.predict(future)

        forecast[forecast.ds > max_date].to_csv(forecast_data_path, index = False)

    else:
        return True

