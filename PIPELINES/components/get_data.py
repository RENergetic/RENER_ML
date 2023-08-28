from kfp.components import OutputPath

def GetData(measurement_name: str, min_date: str, max_date: str,url_pilot : str, pilot_name:str, type_measurement :str,key_measurement : str,
            filter_vars:list , filter_cases: list, output_data_forecast: OutputPath(str), output_data_metric : OutputPath(str)):

    import requests # To REQUIREMENTS
    import json
    import pandas as pd # To REQUIREMENTS
    import maya # To REQUIREMENTS
    from tqdm import tqdm
    from icecream import ic
    from retry import retry # TO REQUIREMENTS

    #Functions definitions

    def GetRequest(url, headers ={}, payload = {}):

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
    def DownloadAssetsData(measurement_name, url_pilot,bucket = "renergetic", min_date = "yesterday", max_date = "tomorrow"):
        
        from datetime import datetime
        import pandas as pd
        import maya
        from tqdm import tqdm
        from icecream import ic

        test = True

        try:
            min_date_from = maya.when(min_date).datetime()
        except:
            ValueError("Please introduce correct time format for MIN_DATE")
        
        try: 
            max_date_from = maya.when(max_date).datetime()
        except:
            ValueError("Please introduce correct time format for MAX_DATE")
        
        datelist = pd.date_range(min_date_from, max_date_from)

        data_ = []
        for i in tqdm(range(len(datelist)-1)):
            from_obj = datelist[i]
            to_obj = datelist[i+1]
            from_ = datetime.strftime(from_obj, "%Y-%m-%d 00:00:00")
            to_ = datetime.strftime(to_obj, "%Y-%m-%d 00:00:00")

            if pilot_name == "Virtual":
                url = "http://influx-api-ren-prototype.apps.paas-dev.psnc.pl/api/measurement/data?measurements={measurement_name}&from={from_}&to={to_}"\
                    .format(measurement_name = measurement_name, from_ = from_, to_= to_)
            else:
                url = url_pilot + "/api-measurement/1.0/api/measurement/data?measurements={measurement_name}&from={from_}&to={to_}"\
                    .format(measurement_name = measurement_name, from_ = from_, to_= to_)
            info_ = GetRequest(url)
            if type(info_) == list:
                data_ = data_ + info_
            elif type(info_) == dict:
                print("Error")
                print(from_)
                print(to_)
        return data_
    def DataFrameAssests(list_data, name_field):
        dicts = []
        for data in list_data:
            try:
                if "energy" in data["fields"].keys():
                    name_value = "energy"
                else:
                    name_value = name_field
                dict_ = {
                    "asset_name": data["tags"]["asset_name"],
                    "value": float(data["fields"][name_value]),
                    "ds": data["fields"]["time"]
                }

                if "type_data" in data["tags"].keys():
                    dict_["type"] = data["tags"]["type_data"]
                elif "typeData" in data["tags"].keys():
                    dict_["type"] = data["tags"]["typeData"]
                else:
                    dict_["type"] = "None"

                if "measurement_type" in data["tags"].keys():
                    dict_["measurement_type"] = data["tags"]["measurement_type"]
                else:
                    dict_["measurement_type"] = "None"
                
                if "direction" in data["tags"].keys():
                    dict_["direction"] = data["tags"]["direction"]
                else:
                    dict_["direction"] = "None"
                if "domain" in data["tags"].keys():
                    dict_["domain"] = data["tags"]["domain"]
                else:
                    dict_["domain"] = "None"
                
                if "sensor_id" in data["tags"].keys():
                    dict_["id_sensor"] = data["tags"]["sensor_id"]
                else:
                    dict_["id_sensor"] = "None"
                
                if "interpolation_method" in data["tags"].keys():
                    dict_["interpolation"] = data["tags"]["interpolation_method"]
                else:
                    dict_["interpolation"] = "None"

                dicts.append(dict_)
            except:
                continue
        return pd.DataFrame(dicts)
    
    @retry(tries= 3)
    def DownloadAndProcess(measurement_name, url_pilot, min_date, max_date, key_measurement):
        # max_date = maya.now().add(days = 3).iso8601()
        list_ = DownloadAssetsData(measurement_name, url_pilot,min_date = min_date, max_date = max_date)
        data = DataFrameAssests(list_, key_measurement)

        return data
    
    def FilterCases(var_value, filter_cases):
        if var_value in filter_cases:
            return True
        else:
            return False

    def FilterData(data, type_measurement, filter_vars, filter_cases):

        for i in range(len(filter_vars)):
            data["Filter"] = data[filter_vars[i]].apply(FilterCases, filter_cases = filter_cases[i])
            data = data[data["Filter"] == True]
        
        return data    
    
    def SendAlert(data):
        ic(data.shape[0])

        if data.shape[0] == 0:
            message = "Not enough data for {measurement_name}".format(measurement_name = measurement_name)
            ic(message)
            
            raise ValueError("Void data to forecast")

    # Code Execution
    
    data_all = DownloadAndProcess(measurement_name, url_pilot, min_date, max_date, key_measurement)
    data_filtered = FilterData(data_all, type_measurement, filter_vars, filter_cases)
    data_to_train = data_filtered[data_filtered.type == type_measurement]
    SendAlert(data_to_train)
    

    data_output = {
        "value": data_to_train["value"].tolist(),
        "time_registered": data_to_train["ds"].tolist(),
        "asset_name": data_to_train["asset_name"].tolist()
    }

    with open(output_data_forecast, "w") as file:
        json.dump(data_output, file)

    
    data_output_metrics = {
        "value": data_filtered["value"].tolist(),
        "asset_name": data_filtered["asset_name"].tolist(),
        "time_registered": data_filtered["ds"].tolist(),
        "type": data_filtered["type"].tolist()
    }

    with open(output_data_metric, "w") as file:
        json.dump(data_output_metrics, file)