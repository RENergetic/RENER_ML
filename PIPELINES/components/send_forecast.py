from kfp.components import InputPath

def SendForecast(input_forecast_data_path: InputPath(str),url_pilot:str, pilot_name:str, asset_name : str, measurement_name: str, key_measurement: str, num_days):
    import json
    import pandas as pd
    from icecream import ic
    import requests
    import pandas as pd
    import maya
    from tqdm import tqdm
    from datetime import datetime

    from urllib3.exceptions import InsecureRequestWarning
    import warnings
    import contextlib

    ## Function Definition ## 

    def GetRequest(url, headers ={}, payload = {}):

        old_merge_environment_settings = requests.Session.merge_environment_settings

        @contextlib.contextmanager
        def no_ssl_verification():
            opened_adapters = set()

            def merge_environment_settings(self, url, proxies, stream, verify, cert):
                # Verification happens only once per connection so we need to close
                # all the opened adapters once we're done. Otherwise, the effects of
                # verify=False persist beyond the end of this context manager.
                opened_adapters.add(self.get_adapter(url))

                settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
                settings['verify'] = False

                return settings

            requests.Session.merge_environment_settings = merge_environment_settings

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', InsecureRequestWarning)
                    yield
            finally:
                requests.Session.merge_environment_settings = old_merge_environment_settings

                for adapter in opened_adapters:
                    try:
                        adapter.close()
                    except:
                        pass
        
        with no_ssl_verification():
            response = requests.request("GET", url, headers = headers, data = payload)
            
        try:
            return response.json()
        except:
            dict_ = {
                "status_code": response.status_code,
                "text": response.text
            }
            return dict_

    def GetFeaturesMeasurement(dicts_measurements, dicts_assets, asset_name, measurement_name):
        domain_ = "None"
        direction = "None"
        type_ = "None"

        for dict_ in dicts_measurements:
            if dict_["name"] == measurement_name:
                try:
                    if dict_["asset"]["name"] == asset_name:
                        print("Asset Found")
                        domain_ = dict_["domain"]
                        direction = dict_["direction"]
                        type_ = dict_["type"]["name"]
                except:
                    continue
        
        if domain_ == "None" and direction == "None" and type_ == "None":
            for dict_ in dicts_assets:
                if dict_["name"] == asset_name:
                    for meas in dict_["measurements"]:
                        if meas["sensor_name"] == measurement_name:
                            if meas["type"]["name"] == key_measurement:
                                print("Measurement Found")
                                domain_ = meas["domain"]
                                direction = meas["direction"]
                                type_ = meas["type"]["name"]

                                break

        
        return domain_, direction, type_

    def GetMeasurementInfo(pilot_name, url_pilot, measurement_name, asset_name):
        if pilot_name != "Virtual":
            url_measurements = "{url_pilot}/api-postgre/1.0/api/measurements".format(
                        url_pilot = url_pilot
                    )
            url_assets = "{url_pilot}/api-postgre/1.0/api/assets".format(
                        url_pilot = url_pilot
                    )
            
            dict_measurement = GetRequest(url_measurements)
            dict_asset = GetRequest(url_assets)
            
            
        else:

            dict_asset= [
                        {"name": "building1", "measurements":[{"sensor_name": "heat_meter", "domain": "heat","direction":"in", "type":{"name": "power_wh"}}, 
                                                                {"sensor_name": "electricity_meter", "domain": "electricity","direction":"in", "type":{"name": "power_wh"}}]},
                        {"name": "building2", "measurements":[{"sensor_name": "heat_meter", "domain": "heat","direction":"in", "type":{"name": "power_wh"}}, 
                                                                {"sensor_name": "electricity_meter", "domain": "electricity","direction":"in", "type":{"name": "power_wh"}}]},
                        {"name": "gas_boiler1", "measurements":[{"sensor_name": "heat_meter", "domain": "heat","direction":"out", "type":{"name": "power_wh"}}]},
                        {"name": "gas_boiler2", "measurements":[{"sensor_name": "heat_meter", "domain": "heat","direction":"out", "type":{"name": "power_wh"}}]},
                        {"name": "cogenerator1", "measurements":[{"sensor_name": "electricity_meter", "domain": "electricity","direction":"out", "type":{"name": "power_wh"}}]},
                        {"name": "cogenerator2", "measurements":[{"sensor_name": "electricity_meter", "domain": "electricity","direction":"out", "type":{"name": "power_wh"}}]},
                        {"name": "wind_farm_1", "measurements":[{"sensor_name": "electricity_meter", "domain": "electricity","direction":"out", "type":{"name": "power_wh"}}]},
                        {"name": "pv_panels_1", "measurements":[{"sensor_name": "electricity_meter", "domain": "electricity","direction":"out", "type":{"name": "power_wh"}}]},
                        {"name": "solar_collector1", "measurements":[{"sensor_name": "heat_meter", "domain": "heat","direction":"out", "type":{"name": "power_wh"}}]}

                    ]
            dict_measurement = []

        return GetFeaturesMeasurement(dict_measurement, dict_asset, asset_name, measurement_name)
 
    def GetPostData(time_, value, 
                    measurement_name, 
                    asset_name,  domain_,
                    direction_energy, type_, 
                    time_prediction):

        data_post = {
                "bucket": "renergetic",
                "measurement": measurement_name,
                "fields":{
                    key_measurement: value,
                    "time": time_,
                },
                "tags":{
                    "domain": domain_,
                    "type_data": "forecasting",
                    "direction": direction_energy,
                    "prediction_window": "{hours}h".format(hours = int(num_days) * 24),
                    "asset_name": asset_name,
                    "measurement_type": type_,
                    "time_prediction": time_prediction
                }
            }
        return data_post
    
    def PostData(data_post, url_pilot, pilot_name):
        from urllib3.exceptions import InsecureRequestWarning
        import warnings
        import contextlib

        old_merge_environment_settings = requests.Session.merge_environment_settings

        @contextlib.contextmanager
        def no_ssl_verification():
            opened_adapters = set()

            def merge_environment_settings(self, url, proxies, stream, verify, cert):
                # Verification happens only once per connection so we need to close
                # all the opened adapters once we're done. Otherwise, the effects of
                # verify=False persist beyond the end of this context manager.
                opened_adapters.add(self.get_adapter(url))

                settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
                settings['verify'] = False

                return settings

            requests.Session.merge_environment_settings = merge_environment_settings

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', InsecureRequestWarning)
                    yield
            finally:
                requests.Session.merge_environment_settings = old_merge_environment_settings

                for adapter in opened_adapters:
                    try:
                        adapter.close()
                    except:
                        pass
        if pilot_name == "Virtual":
            url = "http://influx-api-ren-prototype.apps.paas-dev.psnc.pl/api/measurement"
        else:
            url = url_pilot + "/api-measurement/1.0/api/measurement"

        with no_ssl_verification():
            response = requests.request("POST", url, headers=headers, data=json.dumps(data_post))
        
        return response.status_code


    class TestError(Exception):
        "This error shows to avoid the completion of the task for test purposes"
        pass
    
    ## Procedure ##

    

    forecasted_data = pd.read_csv(input_forecast_data_path)
    domain_, direction_, type_ = GetMeasurementInfo(pilot_name, url_pilot,measurement_name, asset_name)
    ic(type_)
    ic(direction_)
    ic(domain_)
    time_prediction = datetime.strftime(maya.now().datetime(), "%Y-%m-%d %H:%M:%S")

    if measurement_name == "electricity_meter":
        domain_ = "electricity"
    elif measurement_name == "heat_meter":
        domain_ = "heat"
    
    values_ok = []
    values_not_ok = []

    for index, row in tqdm(forecasted_data.iterrows(), total = forecasted_data.shape[0]):
        time_ = str(row["ds"])
        value = row["yhat"]

        
        data_post = GetPostData(time_, value, 
                                measurement_name= measurement_name,
                                domain_ = domain_, asset_name= asset_name, direction_energy= direction_,
                                type_ = type_, time_prediction= time_prediction)

        headers = {
            "Content-Type": "application/json"
        }

        try:
            status_code = PostData(data_post, url_pilot, pilot_name)

        except:
            message = "Error in updating value for measurement name: {measurement_name} in asset: {asset_name} in time {time_pred}"\
                .format(measurement_name = "electricity_meter", asset_name = asset_name, time_pred = data_post["fields"]["time"])
            ic(message)
            status_code = 400
        
        if status_code > 299:
            ic(time_)
            ic(value)
            message = "Error in sending the value for measurement name: {measurement_name} in asset: {asset_name} in time {time_pred}"\
                .format(measurement_name = measurement_name, asset_name = asset_name, time_pred = data_post["fields"]["time"])
            ic(message)
            if len(values_not_ok) == 0:
                print(data_post)
                print(url_pilot)
            
            values_not_ok.append(time_)

        else:
            values_ok.append(time_)
        
    ic(values_ok)
    ic(values_not_ok)
    ic(data_post)
    ic(url_pilot)
        