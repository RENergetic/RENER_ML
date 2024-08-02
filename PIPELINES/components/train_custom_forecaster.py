from kfp.components import InputPath, OutputPath

def TrainCustomForecaster(
    input_data_path: InputPath(str),
    input_weather_path: InputPath(str),
    diff_time,
    num_days,
    pilot_name,
    measurement_name,
    asset_name,
    url_minio,
    access_key,
    secret_key,
    list_forge,
    forecast_data_path: OutputPath(str),
    results_path: OutputPath(str)
):
    
    
    import dill
    from minio import Minio
    import json
    import pandas as pd

    
    def CustomProcess(
            # MINIO ARGUMENTS
            url_minio,
            access_key,
            secret_key,
            # MEASUREMENT-ASSET ARGUMENTS
            pilot_name,
            measurement_name,
            asset_name,
            # FORECAST ARGUMENTS
            diff_time,
            num_days,
            # DATA ARGUMENTS
            data,
            weather_data,
            # OPTION ARGUMENTS
            load_custom):
        
        client = Minio(
                    url_minio,
                    access_key=access_key,
                    secret_key=secret_key,
                )
        if load_custom:
            bucket_name = "{pilot_name}-augur".format(
                    pilot_name = pilot_name.lower().replace("_", "-")
                )
        else:
            bucket_name = "{pilot_name}-{measurement}-{asset}".format(
                pilot_name = pilot_name.lower().replace("_", "-"),
                measurement = measurement_name.lower().replace("_", "-"),
                asset = asset_name.lower().replace("_","-")
            )
        
        client.fget_object(bucket_name,
                                "augur_{model_name}.pkl".format(model_name = model_name),
                                file_path = "augur.pkl")

        with open('augur.pkl', 'rb') as inp:
            augur_custom = dill.load(inp)

        data, weather_data = augur_custom.process_data_to_forecast(data, weather_data)
       
        if not load_custom:
            data, weather_data = augur_custom.train(data, weather_data)

        forecast_ = augur_custom.predict(data, weather_data, diff_time, days_ahead=num_days)

        return forecast_, augur_custom

    model_name = list_forge[0]
    load_custom = list_forge[1]

    if model_name == "None":
        print("No custom model selected")
        forecast_ = pd.DataFrame()
        class Augur:
            ''' Dynamic class for timeseries forecast '''

            def __init__(self, model_name = None, type_model = None, training_date = None, metric = {}, model = None):
                self.model_name = model_name
                self.type_model = type_model
                self.training_date = training_date
                self.metric = metric
                self.model = model
        augur_custom = Augur(model = "None")
    else:
        with open(input_data_path) as file:
            data_str = json.load(file)
        

        data = pd.DataFrame(data_str)
        data = data[data.asset_name == asset_name]

        weather_data = pd.read_feather(input_weather_path)

        try:
            forecast_, augur_custom = CustomProcess(
                url_minio,
                access_key,
                secret_key,
                pilot_name,
                measurement_name,
                asset_name,
                diff_time,
                num_days,
                data,
                weather_data,
                load_custom
            )
            
        except Exception as e:
            forecast_ = pd.DataFrame()
            class Augur:
                ''' Dynamic class for timeseries forecast '''

                def __init__(self, model_name = None, type_model = None, training_date = None, metric = {}, model = None):
                    self.model_name = model_name
                    self.type_model = type_model
                    self.training_date = training_date
                    self.metric = metric
                    self.model = model
            augur_custom = Augur(model = "FAIL")

            print(f"ERROR: {e}")
    try:
        forecast_.columns = ["ds", f"yhat_{model_name}"]
        forecast_.reset_index().to_feather(forecast_data_path)
    except:
        forecast_.reset_index().to_feather(forecast_data_path)
    
    results_dict = {
        "model_name": augur_custom.model_name,

    }

    with open(results_path, "w") as file:
        json.dump(results_dict, file)

    


