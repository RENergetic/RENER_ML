from kfp.components import InputPath

def CheckDataAvailability(input_processed_data_path: InputPath(str), asset_name,diff_time, availability_perc, num_days_check) -> bool:

    import pandas as pd
    import json
    import maya
    from datetime import datetime

    with open(input_processed_data_path) as file:
        data_str = json.load(file)
    
    data = pd.DataFrame(data_str)
    data.columns = [["value", "ds", "asset_name"]]
    data = data[data.asset_name == asset_name]
    data = data[data.ds >= datetime.strftime(maya.when("now").add(days = -2).datetime(), "%Y-%m-%d")]

    expected_values = (60/diff_time) * 24 * num_days_check
    
    return data.shape[0]/expected_values*100 < availability_perc