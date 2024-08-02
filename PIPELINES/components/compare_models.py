from kfp.components import InputPath, OutputPath

def CheckSetModels(input_data_ts_path: InputPath(str),
                    input_data_prophet_path: InputPath(str), 
                   input_data_transformers_path: InputPath(str), 
                   input_data_lstm_path: InputPath(str),
                   input_data_forge_one_path: InputPath(str),
                   input_data_forge_two_path: InputPath(str),
                   list_forges: list,
                   asset_name,
                   metric_name: str = "mae"
                   ) -> str:
    
    import pandas as pd
    import json
    from icecream import ic
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    class VoidData(Exception):
        print("No data was computed")

    def MergeForecast(forecast_test, new_forecast):
        try:
            forecast_test = pd.merge(forecast_test, new_forecast, on = "ds")
            if forecast_test.shape[0] == 0:
                ic("Failed merged")
                raise VoidData
        except (VoidData, KeyError) as e:
            print(e)

        return forecast_test

    name_forge_one = list_forges[0]
    name_forget_two = list_forges[1]

    with open(input_data_ts_path) as file:
        data_str = json.load(file)
    
    data = pd.DataFrame(data_str)
    data = data[data.asset_name == asset_name]

    try:
        input_data_prophet = pd.read_feather(input_data_prophet_path)
    except:
        input_data_prophet = pd.DataFrame()

    ic(input_data_prophet.tail())

    forecast_test = MergeForecast(data, input_data_prophet)

    ic(forecast_test.tail())


    try:
        input_data_lstm = pd.read_feather(input_data_lstm_path)
    except:
        input_data_lstm = pd.DataFrame()
    
    forecast_test = MergeForecast(forecast_test, input_data_lstm)

    ic(forecast_test.tail())

    try:
        input_data_transformers = pd.read_feather(input_data_transformers_path)
    except:
        input_data_transformers = pd.DataFrame()
    
    forecast_test = MergeForecast(forecast_test, input_data_transformers)

    try:
        input_data_forge_one = pd.read_feather(input_data_forge_one_path)
    except:
        input_data_forge_one = pd.DataFrame()
    
    forecast_test = MergeForecast(forecast_test, input_data_forge_one)

    try:
        input_data_forge_two = pd.read_feather(input_data_forge_two_path)
    except:
        input_data_forge_two = pd.DataFrame()
    
    forecast_test = MergeForecast(forecast_test, input_data_forge_two)

    ic(forecast_test.shape)
    ic(forecast_test.columns)
    ic(forecast_test.tail())

    metric_rmse = {}
    metric_mae = {}

    if "yhat_prophet" in forecast_test.columns.values:
        metric_rmse["prophet"] = mean_squared_error(forecast_test.y, forecast_test.yhat_prophet)
        metric_mae["prophet"] = mean_absolute_error(forecast_test.y, forecast_test.yhat_prophet)
    if "yhat_lstm" in forecast_test.columns.values:
        metric_rmse["lstm"] = mean_squared_error(forecast_test.y, forecast_test.yhat_lstm)
        metric_mae["lstm"] = mean_absolute_error(forecast_test.y, forecast_test.yhat_lstm)
    if "yhat_transformers" in forecast_test.columns.values:
        metric_rmse["transformers"] = mean_squared_error(forecast_test.y, forecast_test.yhat_transformer)
        metric_mae["transformers"] = mean_absolute_error(forecast_test.y, forecast_test.yhat_transformer)
    if f"yhat_{name_forge_one}" in forecast_test.columns.values:
        metric_rmse[name_forge_one] = mean_squared_error(forecast_test.y, forecast_test[f"yhat_{name_forge_one}"])
        metric_mae[name_forge_one] = mean_absolute_error(forecast_test.y, forecast_test[f"yhat_{name_forge_one}"])
    
    if f"yhat_{name_forge_two}" in forecast_test.columns.values:
        metric_rmse[name_forge_two] = mean_squared_error(forecast_test.y, forecast_test[f"yhat_{name_forge_two}"])
        metric_mae[name_forge_two] = mean_absolute_error(forecast_test.y, forecast_test[f"yhat_{name_forge_two}"])
    
    metrics_forecast = {
        "mae": metric_mae,
        "rmse": metric_rmse
    }

    if metric_name in metrics_forecast.keys():
        metrics_values = metrics_forecast[metric_name]
    else:
        metrics_values = metrics_forecast["mae"]
    
    main_score_value = 1000000000000
    
    model_to_set = "No model set"

    for key_ in metrics_values.keys():
        value_metric = metrics_values[key_]
        if value_metric < main_score_value:
            model_to_set = key_
            main_score_value = value_metric
    
    return model_to_set

