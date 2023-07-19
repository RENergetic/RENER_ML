from kfp.components import OutputPath

def DownloadDataFromInfluxDB(timestamp:float, output_weather_info_path: OutputPath(str)):
    import pandas as pd

    data = pd.DataFrame()
    try:
        data.to_feather(output_weather_info_path)
    except:
        data.to_csv(output_weather_info_path, index = False)