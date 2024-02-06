from kfp.components import InputPath

def ProduceModelProphet(
        model_saved_path: InputPath(str),
        url_minio,
        access_key,
        secret_key,
        pilot_name,
        measurement_name,
        asset_name):

    from minio import Minio
    import maya
    from datetime import datetime

    client = Minio(
        url_minio,
        access_key=access_key,
        secret_key=secret_key,
    )

    bucket_name = "{pilot_name}-{measurement}-{asset}".format(
        pilot_name = pilot_name,
        measurement = measurement_name,
        asset = asset_name
    )

    client.fput_object(bucket_name,"set_model.json",file_path = model_saved_path)
    client.fput_object(bucket_name,
                       "set_model_{date}.json".format(date = datetime.strftime(maya.now().datetime(), "%Y_%m_%d")),
                       file_path = model_saved_path)
    print("Model saved")

def ProduceModel():

    print("model not available to be produced")