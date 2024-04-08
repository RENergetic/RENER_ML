from kfp.components import InputPath

def SetModel(model_path: InputPath(str),
             type_model: str,
             url_minio,
             access_key,
             secret_key,
             pilot_name,
             measurement_name,
             asset_name):

    import maya
    from datetime import datetime
    from minio import Minio
    import json

    with open(model_path) as file:
        data_model = json.load(file)
    
    model_name = data_model["model_name"]

    dict_model_measurement_asset = {
        "model_name": model_name,
        "set_date":  datetime.strftime(maya.now().datetime(), "%Y-%m-%d"),
        "train_date": model_name[-15:-5].replace("_","-"),
        "type_model": type_model
    }

    client = Minio(
            url_minio,
            access_key=access_key,
            secret_key=secret_key,
        )
    
    bucket_name = "{pilot_name}-{measurement}-{asset}".format(
            pilot_name = pilot_name.lower().replace("_", "-"),
            measurement = measurement_name.lower().replace("_", "-"),
            asset = asset_name.lower().replace("_","-")
        )
    

    
    with open("asset_model_config.json", "w") as file:
        json.dump(dict_model_measurement_asset, file)
    
    client.fput_object(bucket_name,
                        "asset_model_config.json",
                        file_path = "asset_model_config.json")

