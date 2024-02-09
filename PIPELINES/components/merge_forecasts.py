from kfp.components import InputPath, OutputPath

def MergeForecast(
        input_forecast_path: InputPath(str),
        input_results_path: InputPath(str), 
        output_forecast_path: OutputPath(str),
        output_results_path: OutputPath(str)
):
    import pandas as pd
    import json
    data = pd.read_feather(input_forecast_path)
    data.to_feather(output_forecast_path)

    with open(input_results_path) as file:
        data_model = json.load(file)
    
    with open(output_results_path, "w") as file:
        json.dump(data_model, file)