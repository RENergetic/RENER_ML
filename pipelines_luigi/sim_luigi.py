import luigi

class GenerateData(luigi.Task):
    def output(self):
        return luigi.LocalTarget("data.json")
    
    def run(self):
        from datetime import datetime
        import maya
        import random
        import numpy as np
        import requests
        import json
        from discord_webhook import DiscordWebhook
        from icecream import ic
        import math
        
        def SimulateTSValue(td = maya.now(), trend_coef = 0.5, trend_sd = 0.025, seed_offset = 0):
            # Random Trend component around 0.5
            # Definition: 0.5 + uniform random number between -0.025 and 0.025
            trend_ts = trend_coef + random.uniform(-trend_sd,trend_sd)
        
            # Random Weekly component
            # Definition Component Based on the day of the week fixed numver + uniform error
            list_comp_days = [0.2,0.23,0.26,0.24,0.21,0.18,0.18, 0.18]
            sd_days = [0.01,0.02,0.01,0.02,0.01,0.01,0.005, 0.005]
        
            day_of_week = td.weekday
            component_weekly = list_comp_days[day_of_week] + random.uniform(-sd_days[day_of_week], sd_days[day_of_week])
        
            # Random Hourly Component
            # Shape based on the day of the simulation
            random.seed(td.day + seed_offset)
            index_shape = random.randint(1,3)
            shape_name = "Shape_" + str(index_shape)
            shapes_dict = {
                "Shape_1":[-0.1       , -0.16165509, -0.19580765, -0.20749836, -0.2017679 ,-0.18365698, -0.15820627, -0.13045646, -0.10544825, -0.08790597,
                -0.07969859, -0.08118035, -0.0926916 , -0.11431926, -0.14248568,-0.17085613, -0.19302855, -0.20271256, -0.19646934, -0.17387802,
                -0.13466275, -0.07860226, -0.0084324 ,  0.06870108,  0.14529271,0.21385705,  0.27034905,  0.31803569,  0.36111925,  0.40380147,
                    0.44921795,  0.49721282,  0.54699702,  0.59778148,  0.64873965,0.69887417,  0.74713902,  0.79248819,  0.83404121,  0.87204349,
                    0.90720933,  0.94025454,  0.9719917 ,  1.00424743,  1.03945362,1.08005027,  1.12821682,  1.18164127,  1.23422369,  1.27974554,
                    1.3121125 ,  1.32919786,  1.33357925,  1.3281073 ,  1.31564286,1.29981316,  1.28552937,  1.27782515,  1.28172227,  1.29863822,
                    1.32133515,  1.34130265,  1.35003053,  1.34175905,  1.32038759,1.29193471,  1.26241895,  1.23771333,  1.22293425,  1.22295303,
                    1.24264089,  1.28498662,  1.33826295,  1.38381853,  1.40296166,1.37869494,  1.31479044,  1.22894771,  1.13912333,  1.06269959,
                    1.00510714,  0.96048399,  0.92252581,  0.8849787 ,  0.84365402,0.79710807,  0.744087  ,  0.68334896,  0.61496013,  0.54145249,
                    0.46563119,  0.39030138,  0.31826821,  0.25233682,  0.19531237,0.15],
                "Shape_2":[ 0., -0.04103375, -0.08817198, -0.13946464, -0.1929617 ,-0.24671311, -0.29876883, -0.34717883, -0.38999305, -0.42547586,
                -0.45382685, -0.47627221, -0.49404756, -0.50822958, -0.51759701,-0.51919966, -0.51004514, -0.48723606, -0.45030097, -0.40133591,
                -0.3425603 , -0.27619642, -0.20462236, -0.1304485 , -0.05630419,0.01519295,  0.08343867,  0.15210782,  0.2254226 ,  0.30760414,
                    0.4007446 ,  0.50036352,  0.60071601,  0.69605718,  0.78173214,0.85805425,  0.92675155,  0.98955223,  1.04791236,  1.10143735,
                    1.14896196,  1.18931843,  1.22156961,  1.24719603,  1.26912132,1.29028841,  1.31348881,  1.3389036 ,  1.36451233,  1.38822559,
                    1.40799846,  1.42320642,  1.43490909,  1.44426382,  1.45241457,1.45950627,  1.46401005,  1.46423742,  1.45850549,  1.44684352,
                    1.43339228,  1.42289702,  1.42010287,  1.42783964,  1.44221093,1.45784466,  1.46936874,  1.4722723 ,  1.46652136,  1.45353202,
                    1.4347211 ,  1.41136867,  1.38368551,  1.35137928,  1.31415472,1.27179803,  1.22509424,  1.17549813,  1.12447684,  1.07347197,
                    1.02339291,  0.97464625,  0.92761887,  0.88266371,  0.83874269,0.79296899,  0.74232789,  0.68382295,  0.6164467 ,  0.54294106,
                    0.46646335,  0.3901709 ,  0.31722102,  0.25077103,  0.19397825,0.15],
                "Shape_3":[ 0.1,  0.04368145, -0.00611102, -0.05146214, -0.09445662,-0.13717918, -0.18171454, -0.23014741, -0.28456252, -0.34620278,
                -0.40871244, -0.46170489, -0.49475657, -0.49836437, -0.47633539,-0.44249094, -0.41089688, -0.39514781, -0.39680727, -0.40470557,
                -0.40706122, -0.39222805, -0.3558937 , -0.30468233, -0.2461103 ,-0.18767166, -0.13303198, -0.07772009, -0.01622396,  0.05696667,
                    0.14399016,  0.23657478,  0.32444615,  0.39732988,  0.44853585,0.48771076,  0.52915333,  0.5871626 ,  0.67308554,  0.77819026,
                    0.88538361,  0.97754486,  1.03916537,  1.07163453,  1.08642819,1.09515719,  1.10896923,  1.13102895,  1.15776826,  1.18540828,
                    1.21023488,  1.23060292,  1.24732041,  1.26133773,  1.27357663,1.2828239 ,  1.28428948,  1.2728421 ,  1.24336168,  1.19413061,
                    1.13160193,  1.06342997,  0.99726893,  0.93901253,  0.888372  ,0.84370217,  0.80335787,  0.7663065 ,  0.73469977,  0.71172084,
                    0.70055338,  0.703921  ,  0.72095085,  0.74907786,  0.78572713,0.82837812,  0.87517695,  0.92471675,  0.97559894,  1.02624828,
                    1.07141393,  1.10237207,  1.11026281,  1.08642757,  1.03044917,0.95286434,  0.86496734,  0.77801236,  0.69888604,  0.62624165,
                    0.55782034,  0.49136324,  0.42461152,  0.35530631,  0.28118875,0.15]
            }
            comp_hour = shapes_dict[shape_name][td.hour*4 + int(td.minute/15)] + random.uniform(-0.01,0.01)
        
            final_value = trend_ts + component_weekly + comp_hour

            return final_value
        
        # CALCULATE VALUES
        td = maya.now()
        gas_boiler1 = SimulateTSValue(td = td, trend_coef = 0, trend_sd = 0.05, seed_offset = 1994)
        gas_boiler2 = SimulateTSValue(td = td, trend_coef = 1, trend_sd = 0.05, seed_offset = 133)
        
        print(1994)

        b1_consumption = SimulateTSValue(td = td, trend_coef = 3, trend_sd = 0.05, seed_offset= 217)
        b2_consumption = SimulateTSValue(td = td, trend_coef = 3, trend_sd = 0.05, seed_offset= 1511)

        cog_1_supply = SimulateTSValue(td = td, trend_coef = 2, trend_sd = 0.05, seed_offset= 42)
        cog_2_supply = SimulateTSValue(td = td, trend_coef = 2, trend_sd = 0.05, seed_offset= 73)

        b1_elec_consumption = SimulateTSValue(td = td, trend_coef = 7, trend_sd = 0.25, seed_offset= 528)
        b2_elec_consumption = SimulateTSValue(td = td, trend_coef = 6, trend_sd = 0.15, seed_offset= 491)

        payload={}
        headers = {}

        key_weather = None
        key_first = "ff58bcc61d0445199d950a6fa6d15489"
        key_second = "f6d8ac05e90c4b48a7382a39918a8eca"
        
        url_usage = "https://api.weatherbit.io/v2.0/subscription/usage?key={api_key}"\
            .format(api_key = key_first)
        count_usage = requests.request("GET", url_usage, headers=headers, data=payload).json()

        if count_usage["calls_count"] == None:
            count_usage["calls_count"] = 0
            
        try:

            if int(count_usage["calls_count"]) == 50:
                message = "First API Limit excedeed. Back Online at {date_reset}"\
                    .format(date_reset = maya.DT(count_usage["calls_reset_ts"]).iso8601())
                url_disc = "https://discord.com/api/webhooks/1002537248622923816/_9XY9Hi_mjzh2LTVqnmSKXlIFJ5rgBO2b8xna5pynUrzALgtC4aXSFq89uMdlW_v-ZzT"
                webhook = DiscordWebhook(url = url_disc, content= message)
                key_weather = key_second
            elif float(count_usage["calls_count"]) < 50:
                key_weather = key_first
            else:
                key_weather = key_second
        except:
            ic(count_usage)
            key_weather = key_second

        try:
            if key_weather == key_second:
                url_usage = "https://api.weatherbit.io/v2.0/subscription/usage?key={api_key}"\
                    .format(api_key = key_second)
                count_usage = requests.request("GET", url_usage, headers=headers, data=payload).json()
                if float(count_usage["calls_count"]) == 50:
                    message = "First API Limit excedeed. Back Online at {date_reset}"\
                        .format(date_reset = maya.DT(count_usage["calls_reset_ts"]).iso8601())
                    url_disc = "https://discord.com/api/webhooks/1002537248622923816/_9XY9Hi_mjzh2LTVqnmSKXlIFJ5rgBO2b8xna5pynUrzALgtC4aXSFq89uMdlW_v-ZzT"
                    webhook = DiscordWebhook(url = url_disc, content= message)
                    key_weather = None
                elif float(count_usage["calls_count"]) < 50:
                    key_weather = key_second
                else:
                    key_weather = None
        except:
            ic(count_usage)
            key_weather = None

        if key_weather != None:
            url = "https://api.weatherbit.io/v2.0/current?city=Poznan&country=PL&key={api_key}"\
                .format(api_key = key_weather)
            response = requests.request("GET", url, headers=headers, data=payload)
            try:
                data_weather = response.json()
                temperature = data_weather["data"][0]["temp"]
                solar_rad = data_weather["data"][0]["solar_rad"]
                wind_speed = data_weather["data"][0]["wind_spd"]
                wind_direction = data_weather["data"][0]["wind_dir"]
                global_irradiation = data_weather["data"][0]["ghi"]
                weather_info = data_weather["data"][0]["weather"]["description"]
            except:
                message = "API FAILED!"
                url_disc = "https://discord.com/api/webhooks/1002537248622923816/_9XY9Hi_mjzh2LTVqnmSKXlIFJ5rgBO2b8xna5pynUrzALgtC4aXSFq89uMdlW_v-ZzT"
                webhook = DiscordWebhook(url = url_disc, content= message)
                temperature = 15
                solar_rad = 0
                weather_info = "No Info"
        else:
            temperature = 15
            solar_rad = 0
            weather_info = "No Info"

        # WEATHER CONDITIONS

        b1_temperature = temperature + np.round(random.uniform(-1.5, 1.5),1)
        b2_temperature = temperature + np.round(random.uniform(-1.5, 1.5),1)

        b1_solar = solar_rad * random.uniform(0.95, 1.05)
        b2_solar = solar_rad * random.uniform(0.95, 1.05)

        wind_speed_b1 = wind_speed * random.uniform(0.95,1.05)
        wind_speed_b2 = wind_speed * random.uniform(0.95,1.05)

        ghi_b1 = global_irradiation * random.uniform(0.95, 1.05) 
        ghi_b2 = global_irradiation * random.uniform(0.95, 1.05)
        
        # SOLAR COLLECTOR
        solar_collector = np.mean([b1_solar, b2_solar]) * random.uniform(0.965,0.975)/1000*2

        # WIND FARM
        wind_farm = np.mean([wind_speed_b1, wind_speed_b2]) * random.uniform(0.965,0.975)/25*4
        
        perfect_angle = 210
        angle_dir = wind_direction-perfect_angle
        angle_rad = (2 * math.pi * angle_dir)/360

        coef = abs(math.cos(angle_rad))

        wind_farm = coef * wind_farm

        # PV PLANTS

        pv_plant = np.mean([ghi_b1, ghi_b2]) * random.uniform(0.965,0.975)/1000*2

        dict_heat_consumption = {
            "building1": b1_consumption,
            "building2": b2_consumption
        }

        dict_heat_supply = {
            "gas_boiler1": gas_boiler1,
            "gas_boiler2": gas_boiler2,
            "solar_collector1": solar_collector
        }

        dict_weather = {
            "current_temperature": {
                "building1": b1_temperature,
                "building2": b2_temperature
            } ,
            "sun_radiation": {
                "building1": int(b1_solar),
                "building2": int(b2_solar)
            },
            "wind_speed":{
                "building1": wind_speed_b1,
                "buidling2": wind_speed_b2
            },
            "global_irradiance":{
                "building1": ghi_b1,
                "buidling2": ghi_b2
            },
            "wind_dir":{
                "buidling1": wind_direction,
                "buidling2": wind_direction
            },
            "weather_info": {"energy_island": weather_info}
        }

        dict_elec_supply = {
            "wind_farm_1": wind_farm,
            "pv_panel_1": pv_plant,
            "cogenerator_1": cog_1_supply,
            "cogenerator_2": cog_2_supply
        }

        dict_elec_consumption = {
            "building1": b1_elec_consumption,
            "building2": b2_elec_consumption
        }

        dict_ds = {
            "datetime": td.epoch
        }

        dict_all = {
            "ds": dict_ds,
            "heat_supply": dict_heat_supply,
            "heat_consumption": dict_heat_consumption,
            "electricity_supply": dict_elec_supply,
            "electricity_consumption": dict_elec_consumption,
            "weather_info": dict_weather
        }

        with self.output().open("w") as file:
            json.dump(dict_all, file)
class SendInfoInflux(luigi.Task):

    

    def requires(self):
        return GenerateData()
    
    def output(self):
        return luigi.LocalTarget("metrics.json")
    
    def run(self):
        import json
        import json
        import requests
        from discord_webhook import DiscordWebhook
        import maya

        with self.input().open("r") as file:
            data = json.load(file)
        
        def SendData2Influx(measurement_name,dict_, dict_time):
            import json
            import requests
            from discord_webhook import DiscordWebhook
            import maya

            data_str = dict_
            time_str = dict_time
            
            def GetDictPost(measurement_name, value, asset_name, time_pred):
                if measurement_name in ["current_temperature", "sun_radiation", "wind_speed","global_irradiance", "wind_dir"]:
                    if measurement_name == "current_temperature":
                        field_name = "temperature"
                    else:
                        field_name = measurement_name
                        
                    data_post = {
                        "bucket": "renergetic",
                        "measurement": "weather_API",
                        "fields":{
                            field_name: value,
                            "time": maya.MayaDT(time_pred).iso8601()[0:19].replace("T", " ")
                        },
                        "tags":{
                            "domain": "weather",
                            "typeData": "simulated",
                            "direction": "None",
                            "prediction_window": "Oh",
                            "asset_name": asset_name,
                            "time_prediction": maya.when("now").epoch
                        }
                    }
                elif measurement_name in ["heat_supply", "heat_consumption"]:
                    
                    if measurement_name == "heat_supply":
                        direction_energy = "out"
                    elif measurement_name == "heat_consumption":
                        direction_energy = "in"
                    else:
                        direction_energy = "None"
                    
                    data_post = {
                        "bucket": "renergetic",
                        "measurement": "heat_meter",
                        "fields":{
                            "energy": value,
                            "time": maya.MayaDT(time_pred).iso8601()[0:19].replace("T", " ")
                        },
                        "tags":{
                            "domain": "heat",
                            "typeData": "simulated",
                            "direction": direction_energy,
                            "prediction_window": "Oh",
                            "asset_name": asset_name,
                            "time_prediction": maya.when("now").epoch
                        }
                    }
                elif measurement_name in ["electricity_supply", "electricity_consumption"]:
                    if measurement_name == "electricty_supply":
                        direction_energy = "out"
                    elif measurement_name == "electricity_consumption":
                        direction_energy = "in"
                    else:
                        direction_energy = "None"
                    
                    data_post = {
                        "bucket": "renergetic",
                        "measurement": "electricity_meter",
                        "fields":{
                            "energy": value,
                            "time": maya.MayaDT(time_pred).iso8601()[0:19].replace("T", " ")
                        },
                        "tags":{
                            "domain": "electricity",
                            "typeData": "simulated",
                            "direction": direction_energy,
                            "prediction_window": "Oh",
                            "asset_name": asset_name,
                            "time_prediction": maya.when("now").epoch
                        }
                    }
                else:
                    data_post = {}
                    
                return data_post


            url =  "http://influx-api-swagger-ren-prototype.apps.paas-dev.psnc.pl/api/measurement"

            headers = {
            "Content-Type": "application/json"
            }
            
            msg = "Upload Status:\n"
            
            for key_dict in data_str.keys():
                if measurement_name != "weather_info":
                    value = data_str[key_dict]
                    asset_name = key_dict
                    time_pred = time_str["datetime"]
                    
                    data_post = GetDictPost(measurement_name, value, asset_name, time_pred)
                    try:
                        response = requests.request("POST", url, headers=headers, data=json.dumps(data_post))
                        status_code = response.status_code
                    except:
                        url_disc = "https://discord.com/api/webhooks/1002537248622923816/_9XY9Hi_mjzh2LTVqnmSKXlIFJ5rgBO2b8xna5pynUrzALgtC4aXSFq89uMdlW_v-ZzT"
                        message = "Error in updating value for measurement name: {measurement_name} in asset: {asset_name}"\
                            .format(measurement_name = measurement_name, asset_name = asset_name)
                        webhook = DiscordWebhook(url = url_disc, content = message)
                        status_code = 500
                    
                    if status_code == 200:
                        msg = msg + measurement_name + "_" + key_dict + " -> SUCCESS" + "\n"
                    else:
                        msg = msg + measurement_name + "_" + key_dict + " -> FAILED" + "\n"
                else:
                    for key_specific in data_str[key_dict].keys():
                        measurement_name_spec = key_dict
                        value = data_str[key_dict][key_specific]
                        asset_name = key_specific
                        time_pred = time_str["datetime"]
                        
                        data_post = GetDictPost(measurement_name_spec, value, asset_name, time_pred)
                        
                        response = requests.request("POST", url, headers=headers, data=json.dumps(data_post))
                        
                        if response.status_code == 200:
                            msg = msg + key_dict + "_" + key_specific + " -> SUCCESS" + "\n"
                        else:
                            print(measurement_name)
                            print(measurement_name_spec)
                            print(data_post)
                            msg = msg + key_dict + "_" + key_specific + " -> FAILED" + "\n"
                            
            print(msg)
        
        list_correct = []
        list_incorrect = []

        for key_ in data.keys():
            if key_ != "ds":
                try:
                    SendData2Influx(key_, data[key_], data["ds"])
                    list_correct.append(key_)
                except:
                    list_incorrect.append(key_)
        
        dict_metrics = {
            "measurements_correct": list_correct,
            "measurements_incorrect" : list_incorrect
        }

        with self.output().open("w") as file:
            json.dump(dict_metrics, file)


                

    
