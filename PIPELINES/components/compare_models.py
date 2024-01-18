from kfp.components import InputPath, OutputPath

def CheckSetModels(input_data_ts_path: InputPath(str),
                    input_data_prophet_path: InputPath(str), 
                   input_data_transformers_path: InputPath(str), 
                   input_data_lstm_path: InputPath(str),
                   metric_name: str = "mae"
                   ) -> str:
    
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    def MergeForecast(forecast_test, new_forecast):
        if forecast_test.shape[0] != 0 & new_forecast.shape[0] != 0:
            forecast_test = pd.merge(forecast_test, new_forecast, on = "ds")
        
        return forecast_test

    forecast_test = pd.read_feather(input_data_ts_path)

    try:
        input_data_prophet = pd.read_feather(input_data_prophet_path)
    except:
        input_data_prophet = pd.DataFrame()

    forecast_test = MergeForecast(forecast_test, input_data_prophet)

    try:
        input_data_lstm = pd.read_feather(input_data_lstm_path)
    except:
        input_data_lstm = pd.DataFrame()
    
    forecast_test = MergeForecast(forecast_test, input_data_lstm)

    try:
        input_data_transformers = pd.read_feather(input_data_transformers_path)
    except:
        input_data_transformers = pd.DataFrame()
    
    forecast_test = MergeForecast(forecast_test, input_data_transformers)

    metric_rmse = {}
    metric_mae = {}

    if "yhat_prophet" in forecast_test.columns.values:
        metric_rmse["prophet"] = mean_squared_error(forecast_test.value, forecast_test.yhat_prophet)
        metric_mae["prophet"] = mean_absolute_error(forecast_test.value, forecast_test.yhat_prophet)
    if "yhat_lstm" in forecast_test.columns.values:
        metric_rmse["lstm"] = mean_squared_error(forecast_test.value, forecast_test.yhat_lstm)
        metric_mae["lstm"] = mean_absolute_error(forecast_test.value, forecast_test.yhat_lstm)
    if "yhat_transformers" in forecast_test.columns.values:
        metric_rmse["transformers"] = mean_squared_error(forecast_test.value, forecast_test.yhat_transformer)
        metric_mae["transformers"] = mean_absolute_error(forecast_test.value, forecast_test.yhat_transformer)
    
    metrics_forecast = {
        "mae": metric_mae,
        "rmse": metric_rmse
    }

    if metric_name in metrics_forecast.keys():
        metrics_values = metrics_forecast[metric_name]
    else:
        metrics_values = metrics_forecast["mae"]
    
    main_score_value = 1000000000000
    
    for key_ in metrics_values.keys():
        value_metric = metrics_values[key_]
        if value_metric < main_score_value:
            model_to_set = key_
            main_score_value = value_metric
    
    return model_to_set

