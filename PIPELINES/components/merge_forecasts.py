from kfp.components import InputPath, OutputPath

def MergeForecast(
        input_forecast_path: InputPath(str), 
        output_forecast_path: OutputPath(str)
):
    import pandas as pd
    data = pd.read_feather(input_forecast_path)
    data.to_feather(output_forecast_path)