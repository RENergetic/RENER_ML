from kfp.components import InputPath

def CalculateForecastMetrics(input_data_metric_path: InputPath(str), asset_name, mae_threshold: float) -> bool:
    
    import json
    import maya
    from datetime import datetime
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    from icecream import ic

    with open(input_data_metric_path) as file:
        data_metrics = json.load(file)
    
    data_metrics = pd.DataFrame(data_metrics)
    data_metrics = data_metrics[data_metrics.asset_name == asset_name]

    max_ds = max(data_metrics.time_registered)
    min_ds = min(data_metrics.time_registered)
    
    if (maya.when(max_ds)- maya.when(min_ds)).days < 30:
        return True
    else:
        date_metric = datetime.strftime(maya.when(max_ds).add(days = -7).datetime(), "%Y-%m-%d")
        data_metrics_real = data_metrics[(data_metrics.time_registered >= date_metric) & (data_metrics.type == "real")][["time_registered", "value"]]
        data_metrics_forecast = data_metrics[(data_metrics.time_registered >= date_metric) & (data_metrics.type == "forecast")][["time_registered", "value"]]
        data_metrics_real.columns = ["ds", "value_real"]
        data_metrics_forecast.columns = ["ds", "value_forecast"]

        data_metrics = pd.merge(data_metrics_real, data_metrics_forecast, on = "ds")
        try:
            mae_metric = mean_absolute_error(data_metrics["value_real"], data_metrics["value_forecast"])
        except ValueError:
            mae_metric = mae_threshold + 1

        ic(mae_metric)
        ic(mae_threshold)

        if mae_metric > mae_threshold:
            return True
        else:
            return False

def CheckModelInList(
    forecast_model,
    forecast_list
) -> bool:
    
    try:

        if forecast_list == "all":
            return True
        elif type(forecast_list) == list:
            return forecast_model in forecast_list
        else:
            return False

    except:
        return False