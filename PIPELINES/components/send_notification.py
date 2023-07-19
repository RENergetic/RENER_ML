from kfp.components import InputPath

def SendNotification(forecast_data_path: InputPath(str), threshold_data_path: InputPath(str), asset_name, pilot_name, url_pilot):
    
    import json 
    import pandas as pd
    from discord_webhook import DiscordWebhook
    import maya
    from datetime import datetime
    import requests
    import fuckit
    from icecream import ic
    import numpy as np

    
    # get notification code for anomaly high and low
    @fuckit
    def GetNotificationCodes(pilot_name, url_pilot):
        if pilot_name == "Virtual":
            url_notification_definition = "http://api-ren-prototype.apps.paas-dev.psnc.pl/api/notification/definition"
        else:
            url_notification_definition = "{url_pilot}/api-postgre/1.0/api/notification/definition".format(
            url_pilot = url_pilot
        )
        payload={}
        headers = {}

        response = requests.request("GET", url_notification_definition, headers=headers, data=payload)

        

        try:
            dict_notifications = response.json()
            if response.status_code > 299:
                ic(url_notification_definition)
                raise ValueError("The request was not successful")
        except:
            return {}
        
        code_high = 0
        code_low = 0

        for notif in dict_notifications:
            if notif["message"] == "message.anomaly.high":
                code_high = notif["code"]
            elif notif["message"] == "message.anomaly.low":
                code_low = notif["code"]
        codes = {
            "code_high": code_high,
            "code_low": code_low
        }
        return codes
    
    def ObtainCodes(codes):
        if "code_low" in codes.keys():
            code_low = codes["code_low"]
        else:
            code_low = 0

        if "code_high" in codes.keys():
            code_high = codes["code_high"]
        else:
            code_high = 0
        return code_high, code_low
    
    def GetIds(asset_name, pilot_name, url_pilot):
        # get asset_id for asset_name
        payload = {}
        headers = {}
        
        if pilot_name == "Virtual":
            url_asset_name = "http://api-ren-prototype.apps.paas-dev.psnc.pl/api/assets?name={asset_name}".format(asset_name = asset_name)
        else:
            url_asset_name = "{url_pilot}/api-postgre/1.0/api/assets?name={asset_name}".format(url_pilot = url_pilot, asset_name = asset_name)
        
        try:
            response = requests.request("GET", url_asset_name, headers=headers, data=payload)
            dict_asset = response.json()[0]
            id_asset = dict_asset["id"]
        except:
            dict_asset = {}
            id_asset = -1
        
        if "measurements" in dict_asset and len(dict_asset["measurements"]) > 0:
            id_measurement = dict_asset["measurements"][0]["id"]
        else:
            id_measurement = -1

        id_dashboard = 1

        return id_asset, id_dashboard, id_measurement
    
    def PostNotification(code_low, date_from, date_to, id_asset, id_dashboard, value, measurement_id, time_, name_pilot, url_pilot):
        date_to = maya.parse(time_).add(minutes = 15).epoch
        dict_post = {
            "notification_code": code_low,
            "date_from": date_from,
            "date_to": date_to,
            "asset": id_asset,
            "dashboard": id_dashboard,
            "value": value,
            "measurement": measurement_id,
        }
        if name_pilot == "Virtual":
            url = "http://api-ren-prototype.apps.paas-dev.psnc.pl/api/notification"
        else:
            url = "{url_pilot}/api-postgre/1.0/api/notification".format(
                url_pilot = url_pilot
            )
        headers = {
            "Content-Type": "application/json"
        }
        try:
            response = requests.request("POST", url, headers=headers, data=json.dumps(dict_post))
            status_code = response.status_code
        except:
            print(url)
            print(dict_post)
            raise ValueError
        
        if response.status_code > 299:
            print(response.text)
            print(response.status_code)
            print(url)
            print(dict_post)
            raise ValueError

        url_disc = "https://discord.com/api/webhooks/1002537248622923816/_9XY9Hi_mjzh2LTVqnmSKXlIFJ5rgBO2b8xna5pynUrzALgtC4aXSFq89uMdlW_v-ZzT"
        message = "Anomaly detect between {date_from} and {date_to} to asset {asset_name}. Response of Notification {status_code}".\
            format(date_from = date_from, date_to = date_to, status_code = status_code, asset_name = asset_name)
        webhook = DiscordWebhook(url = url_disc, content = message)
        webhook.execute()

        return response, status_code

    def NotificationProcess(forecast_data, code_low, code_high, id_asset, id_dashboard, measurement_id, name_pilot):

        failed_notifications = []
        success_notification = []

        max_ds = max(forecast_data["ds"])
        date_notification = datetime.strftime(maya.now().datetime(), "%Y-%m-%d %H:%M:%S")
        mode = "none"
        values = []

        for index,row in forecast_data.iterrows():
            value = row["yhat"]
            time_ = str(row["ds"])
            print(mode)
            # SEND NOTIFICATION
            if mode == "none":
                if value < threshold_min:
                    date_from = maya.parse(time_).epoch
                    mode = "lower"

                    if time_ == max_ds:
                        response, status_code = PostNotification(code_low, date_from, date_to, id_asset, id_dashboard, value, measurement_id, time_, name_pilot, url_pilot)
                        if status_code > 299:
                            failed_notifications.append("Failed Notification send from {date_from} to {date_to}".format(date_from = date_from, date_to = date_to))
                            failed_notifications.append(response.text)
                        else:
                            success_notification.append("Success Notification send from {date_from} to {date_to}".format(date_from = date_from, date_to = date_to))
                    else:
                        values.append(value)
                if value > threshold_max:
                    date_from = maya.parse(time_).epoch
                    mode = "upper"

                    if time_ == max_ds:
                        response, status_code = PostNotification(code_low, date_from, date_to, id_asset, id_dashboard, value, measurement_id, time_, name_pilot, url_pilot)
                        if status_code > 299:
                            failed_notifications.append("Failed Notification send from {date_from} to {date_to}".format(date_from = date_from, date_to = date_to))
                            failed_notifications.append(response.text)
                        else:
                            success_notification.append("Success Notification send from {date_from} to {date_to}".format(date_from = date_from, date_to = date_to))
                    else:
                        values.append(value)
            elif mode == "lower":
                if value > threshold_min or time_ == max_ds:
                    date_to = maya.parse(time_).epoch
                    if len(values) == 0:
                        value_notification = 0
                    else:
                        value_notification = np.mean(values)

                    response, status_code = PostNotification(code_low, date_from, date_to, id_asset, id_dashboard, value_notification, measurement_id, time_, name_pilot, url_pilot)
                    if status_code > 299:
                        failed_notifications.append("Failed Notification send from {date_from} to {date_to}".format(date_from = date_from, date_to = date_to))
                        failed_notifications.append(response.text)
                    else:
                        success_notification.append("Success Notification send from {date_from} to {date_to}".format(date_from = date_from, date_to = date_to))

                    if value > threshold_max:
                        mode = "upper"
                        date_from = maya.parse(time_).epoch
                        values = [value]
                    else:
                        values = []
                        mode = "none"
                else:
                    values.append(value)
            elif mode == "upper":
                if value < threshold_max or time_ == max_ds:
                    date_to = maya.parse(time_).epoch
                    if len(values) == 0:
                        value_notification = 0
                    else:
                        value_notification = np.mean(values)

                    response, status_code = PostNotification(code_low, date_from, date_to, id_asset, id_dashboard, value_notification, measurement_id, time_, name_pilot, url_pilot)
                    if status_code > 299:
                        failed_notifications.append("Failed Notification send from {date_from} to {date_to}".format(date_from = date_from, date_to = date_to))
                        failed_notifications.append(response.text)
                    else:
                        success_notification.append("Success Notification send from {date_from} to {date_to}".format(date_from = date_from, date_to = date_to))


                    if value < threshold_min:
                        date_from = maya.parse(time_).epoch
                        mode = "lower"
                        values = [value]
                    else:
                        values = []
                        mode = "none"

                    
                else:
                    values.append(value)

        return success_notification, failed_notifications

    with open(threshold_data_path) as file:
        dict_threshold = json.load(file)
        try:
            threshold_min = dict_threshold[asset_name]
        except:
            threshold_min = 0
        
        try:
            threshold_max = dict_threshold[asset_name]
        except:
            threshold_max = 1000000000000000
        
        threshold_min = 10
        threshold_max = 0
    
    forecast_data = pd.read_csv(forecast_data_path)
    

    codes = GetNotificationCodes(pilot_name, url_pilot)
    if codes == None:
        codes = {}
    code_high, code_low = ObtainCodes(codes)
    id_asset, id_dashboard, id_measurement = GetIds(asset_name, pilot_name, url_pilot)
    success_notifications, failed_notifications = NotificationProcess(forecast_data, code_low, code_high, id_asset, id_dashboard, id_measurement, pilot_name)

    print("SUCCESS")
    for not_ in success_notifications:
        print(not_)
    
    print("-----------")

    print("FAILED")
    for not_ in failed_notifications:
        print(not_)

