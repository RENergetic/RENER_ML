from kfp.components import InputPath, OutputPath

def CheckDataAvailability(input_processed_data_path: InputPath(str), asset_name,
                          diff_time, availability_perc, num_days_check) -> bool:

    import pandas as pd
    import json
    import maya
    from datetime import datetime

    with open(input_processed_data_path) as file:
        data_str = json.load(file)
    
    data = pd.DataFrame(data_str)
    data.columns = [["value", "ds", "asset_name"]]
    data = data[data.asset_name == asset_name]
    data = data[data.ds >= datetime.strftime(maya.when("now").add(days = -1 * num_days_check).datetime(), "%Y-%m-%d")]

    expected_values = (60/diff_time) * 24 * num_days_check
    
    return data.shape[0]/expected_values*100 < availability_perc

def CheckModelSet(url_minio, access_key, secret_key,
                  pilot_name, measurement_name, asset_name,
                  output_model_path: OutputPath(str)):
    
    from minio import Minio
    import json

    client = Minio(
        url_minio,
        access_key= access_key, 
        secret_key= secret_key
    )

    bucket_name = "{pilot_name}-{measurement}-{asset}".format(
            pilot_name = pilot_name.lower().replace("_", "-"),
            measurement = measurement_name.lower().replace("_", "-"),
            asset = asset_name.lower().replace("_","-")
        )
    print(bucket_name)

    dict_model = {}

    if client.bucket_exists(bucket_name) != True:
        print("ERROR: There is no bucket created for the asset: {asset} with measurement {meas} in the pilot {pilot}".format(
            pilot = pilot_name.lower().replace("_", "-"),
            meas = measurement_name.lower().replace("_", "-"),
            asset = asset_name.lower().replace("_","-")
        ))
    else:
        list_objects = client.list_objects(bucket_name)
        object_names = []
        for obj_ in list_objects:
            object_names.append(obj_._object_name)
        
        if "asset_model_config.json" not in object_names:
            print("No model set for this asset")
        else:
            client.fget_object(bucket_name, "asset_model_config.json", file_path = "asset_model_config.json")
            
            with open("asset_model_config.json") as file:
                dict_model = json.load(file)

    with open(output_model_path, "w") as file:
        json.dump(dict_model, file)
            
