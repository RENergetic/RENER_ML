from kfp.components import InputPath, OutputPath

def ProcessForecast(
        output_forecast_path: OutputPath(str),
        output_results_path: OutputPath(str),
        input_forecast_path: InputPath(str) = None,
        input_results_path: InputPath(str)= None,
        failed = False
):
    import pandas as pd
    import json

    if failed == False:
        data = pd.read_feather(input_forecast_path)
        data.to_feather(output_forecast_path)
        with open(input_results_path) as file:
            data_model = json.load(file)
        
        with open(output_results_path, "w") as file:
            json.dump(data_model, file)
    else:
        data = pd.DataFrame()
        data.to_feather(output_forecast_path)
        dict_ = {}
        with open(output_results_path, "w") as file:
            json.dump(dict_, file)

def MergeForecast(input_forecast_path: InputPath(str),
                  input_forecast_load_path: InputPath(str),
                  input_results_path: InputPath(str),
                  input_results_load_path: InputPath(str),
                  output_results_path:OutputPath(),
                  output_forecast_path: OutputPath(str)):
    
    import pandas as pd
    import json

    data_forecast = pd.read_feather(input_forecast_path)
    data_forecast_load = pd.read_feather(input_forecast_load_path)

    if data_forecast.shape[0] == 0:
        data_forecast_load.to_feather(output_forecast_path)
        with open(input_results_load_path) as file:
            data_model = json.load(file)
    else:
        data_forecast.to_feather(output_forecast_path)
        with open(input_results_path) as file:
            data_model = json.load(file)
        

    with open(output_results_path, "w") as file:
            json.dump(data_model, file)
    