from kfp.components import InputPath, OutputPath

def ProcessData(input_data_path: InputPath(str), hourly_aggregate, minute_aggregate ,min_date, max_date, list_forges, output_data_forecast: OutputPath(str)):

    import maya
    from datetime import datetime
    import json
    import pandas as pd
    from icecream import ic
    from tqdm import tqdm
    from minio import Minio
    import dill
    
    min_date = datetime.strftime(maya.when(min_date).datetime(), "%Y-%m-%d")
    max_date = datetime.strftime(maya.when(max_date).datetime(), "%Y-%m-%d")
    
    ic(hourly_aggregate)
    ic(minute_aggregate)

    with open(input_data_path) as file:
        data_str = json.load(file)
    
    data = pd.DataFrame(data_str)

    # DEFINE HOURLY AGGREGATE PROCESS

    def ProcessHourly(data, hourly_aggregate, min_date, max_date):
        list_dicts = []
        list_data = []

        for asset_name in pd.unique(data.asset_name):
            data_iter = data[data.asset_name == asset_name]
            
            def GetHourDate(str_):
                import maya
                from datetime import datetime

                return datetime.strftime(maya.parse(str_).datetime(), "%Y-%m-%d %H:00:00")

            data_iter["hour"] = data_iter["time_registered"].apply(GetHourDate)
            data_group = (data_iter.groupby('hour')
                .agg({'time_registered':'count', 'value': hourly_aggregate})
                .reset_index()
            )

            df = data_group[["hour", "value"]]
            df = df.rename(columns={'hour':'ds', 'value': 'y'})
            last_value = df.y.tolist()[0]
            for ds_obj in tqdm(pd.date_range(min_date, max_date)):
                for i in range(24):
                    if i < 10:
                        str_i = "0"+str(i)
                    else:
                        str_i = str(i)
                    ds_str = "{date} {H}:00:00".format(date = ds_obj.strftime("%Y-%m-%d"), H = str_i)

                    if df[df.ds == ds_str].shape[0] == 0:
                        if hourly_aggregate == "max":
                                value_ = last_value
                        else:
                            value_ = 0

                        dict_ = {
                            "time_registered": ds_str,
                            "value": value_,
                            "asset_name": asset_name
                        }
                    else:
                        dict_ = {
                            "time_registered": ds_str,
                            "value": df[df.ds == ds_str].y.tolist()[0],
                            "asset_name": asset_name
                        }
                        if minute_aggregate == "max":
                            last_value = dict_["value"]
                    list_dicts.append(dict_)
            if hourly_aggregate == "max":
                data_1 = pd.DataFrame(list_dicts)
                data_1["value_1"] = data_1["value"].shift(1)
                data_1["value"] = data_1["value"] - data_1["value_1"]
                data_1 = data_1.drop(["value_1"], axis = 1)
                data_1["value"] = data_1["value"].fillna(0)
                list_data.append(data_1)
                list_dicts = []

        if hourly_aggregate == "max":
            output_data = pd.concat(list_data, ignore_index = True)
        else:
            output_data = pd.DataFrame(list_dicts)
        
        return output_data
    
    def ProcessMinutely(data, minute_aggregate, min_date, max_date):
        list_dicts = []
        list_data = []
        for asset_name in tqdm(pd.unique(data.asset_name)):
            data_iter = data[data.asset_name == asset_name]
            def GetHourMinuteDate(str_):
                import maya
                from datetime import datetime

                return datetime.strftime(maya.parse(str_).datetime(), "%Y-%m-%d %H:%M:00")
            
            data_iter["minute"] = data_iter["time_registered"].apply(GetHourMinuteDate)
            data_group = (data_iter.groupby('minute')
                .agg({'time_registered':'count', 'value': minute_aggregate})
                .reset_index()
            )

            df = data_group[["minute", "value"]]
            df = df.rename(columns={'minute':'ds', 'value': 'y'})

            last_value = df.y.tolist()[0]

            for ds_obj in tqdm(pd.date_range(min_date, max_date)):
                for i in range(24):
                    for j in range(60):
                        if i < 10:
                            str_i = "0"+str(i)
                        else:
                            str_i = str(i)
                        
                        if j < 10:
                            str_j = "0" + str(j)
                        else:
                            str_j = str(j)
                        ds_str = "{date} {H}:{M}:00".format(date = ds_obj.strftime("%Y-%m-%d"), H = str_i, M = str_j)

                        if df[df.ds == ds_str].shape[0] == 0:

                            if minute_aggregate == "max":
                                value_ = last_value
                            else:
                                value_ = 0

                            dict_ = {
                                "time_registered": ds_str,
                                "value": value_,
                                "asset_name": asset_name
                            }
                        else:
                            dict_ = {
                                "time_registered": ds_str,
                                "value": df[df.ds == ds_str].y.tolist()[0],
                                "asset_name": asset_name
                            }
                            if minute_aggregate == "max":
                                last_value = dict_["value"]
                        list_dicts.append(dict_)
            if minute_aggregate == "max":
                data_1 = pd.DataFrame(list_dicts)
                data_1["value_1"] = data_1["value"].shift(1)
                data_1["value_cummulative"] = data_1["value"].copy()
                data_1["value"] = data_1["value"] - data_1["value_1"]
                data_1 = data_1.drop(["value_1"], axis = 1)
                data_1["value"] = data_1["value"].fillna(0)
                list_data.append(data_1)
                list_dicts = []

        if minute_aggregate == "max":
            output_data = pd.concat(list_data, ignore_index = True)
        else:
            output_data = pd.DataFrame(list_dicts)
        
        return output_data
    
    def DownloadForge(forge_name, url_minio, access_key, secret_key):
        
        """
        
        Download Forge Class to process (if necessary)

        Input:
        forge_name -- Name of the class. It is recommended to write it as "{process_name}_v_0_0_1.dill"
        url_minio
        access_key
        secret_key

        Output:
        Message eithe "Accepted" or "ERROR: {message of error}"

        """

        try:
            client = Minio(
                url_minio,
                access_key=access_key,
                secret_key=secret_key,
            )

            bucket_name = "forge-classes"

            client.fget_object(bucket_name,
                            f"{forge_name}.pkg",
                            file_path = "forge.pkg")

            return "Accepted"
        except Exception as e:
            return f"ERROR: {e}"

    def LoadForge():

        with open('forge.pkg', 'rb') as inp:
            forge = dill.load(inp)

        return forge


    if hourly_aggregate in ["mean","sum", "max"]:
        print("hourly process")
        output_data = ProcessHourly(data, hourly_aggregate, min_date, max_date)
    elif minute_aggregate in ["max", "sum", "mean"]:
        print("minutely process")
        output_data = ProcessMinutely(data, minute_aggregate, min_date, max_date)
    else:
        output_data = data

    for forge_name in list_forges:
        if forge_name != None:
            message_ = DownloadForge(forge_name)
            if message_ == "Accepted":
                try:
                    forge = LoadForge()
                    output_data = forge.process(output_data)
                    ic(forge_name)
                    ic(output_data.shape[0])
                except:
                    print("{forge_name} Failed")
        


    

    data_output = {
        "y": output_data["value"].tolist(),
        "ds": output_data["time_registered"].tolist(),
        "asset_name": output_data["asset_name"].tolist()
    }

    if minute_aggregate == "max" or hourly_aggregate == "max":
        data_output["value_cummulative"] = output_data["value_cummulative"].tolist()

    with open(output_data_forecast, "w") as file:
        json.dump(data_output, file)

