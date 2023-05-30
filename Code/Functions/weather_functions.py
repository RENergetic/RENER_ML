import maya
from datetime import datetime
import requests
import pandas as pd

def ManageDateTime(ds_obj):
    return datetime.strftime(maya.parse(ds_obj).datetime(), "%Y-%m-%d %H:%M:%S")

class InvalidArgument(Exception):
    "Input Either City name or Lat AND Lon for city"
    pass



def GetWeatherData(city = "None", lon = "no", lat= "no", start_date= "yesterday", end_date= "today"):

    # Input city or lat/lon + start date and end date

    if city != "None":
        url = "https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=10&language=en&format=json".format(city_name = city)
        geo_ = requests.get(url)
        results_ = geo_.json()["results"][0]
        lat = results_["latitude"]
        lon = results_["longitude"]
    elif lat == "no" or lon == "no":
        raise InvalidArgument
    start_date_parsed = datetime.strftime(maya.when(start_date).datetime(), "%Y-%m-%d")
    end_date_parsed = datetime.strftime(maya.when(end_date).datetime(), "%Y-%m-%d")
    url = "https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,cloudcover,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance"\
        .format(lat = lat, lon = lon, start_date = start_date_parsed, end_date = end_date_parsed)
    data = requests.get(url)
    try:
        data = data.json()["hourly"]
    except:
        print(data)
    data = pd.DataFrame(data)
    data["ds"] = data["time"].apply(ManageDateTime)

    url = "https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&past_days=10&hourly=temperature_2m,cloudcover,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance"\
        .format(latitude = lat, longitude = lon)
    forecast_data = requests.get(url)
    forecast_data = pd.DataFrame(forecast_data.json()["hourly"])
    forecast_data["ds"] = forecast_data["time"].apply(ManageDateTime)

    data["type"] = "real"
    forecast_data["type"] = "forecast"


    return pd.concat([data, forecast_data], ignore_index= True)