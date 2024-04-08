from kfp.components import InputPath, OutputPath

def CompareForecast(input_data_real_path: InputPath(str), input_data_forecast_path: InputPath(str),
                    asset_name, output_metrics_path: OutputPath(str)):

    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from icecream import ic
    import json
    import pandas as pd
    import numpy as np

    def CompareMinMax(list_, min_perc, max_perc):
        real = list_[0]
        yhat = list_[1]

        if (yhat >= real*min_perc) and (yhat <= real*max_perc):
            return True
        else:
            return False


    def Coverage(data_for_metrics, min_perc = 0.8, max_perc = 1.2):
        
        data_for_metrics["Cov"] = data_for_metrics[["y_real", "y_forecast"]].apply(CompareMinMax, min_perc = min_perc,
                                                         max_perc = max_perc, axis = 1)
        return np.round(
            data_for_metrics[data_for_metrics.Cov == True].shape[0]/ data_for_metrics.shape[0]*100,2)
    
    with open(input_data_real_path) as file:
        data_str = json.load(file)
        
    data_real = pd.DataFrame(data_str)

    data_real.columns = [["y_real", "ds", "asset_name"]]

    with open(input_data_forecast_path) as file:
        data_str = json.load(file)
        
    data_forecast = pd.DataFrame(data_str)

    data_forecast.columns = [["y_forecast", "ds", "asset_name"]]

    data_real = data_real[data_real.asset_name == asset_name].reset_index()
    data_forecast = data_forecast[data_forecast.asset_name == asset_name].reset_index()

    data_for_metrics = pd.merge(data_real, data_forecast, on = "ds")

    metric_dict = {
        "R2": r2_score(data_for_metrics.y_real, data_for_metrics.y_forecast),
        "MAE": mean_absolute_error(),
        "RMSE": np.sqrt(mean_squared_error()),
        "Coverage": Coverage(data_for_metrics)
    }

    with open(output_metrics_path, "w") as file:
        json.dump(metric_dict, file)


    