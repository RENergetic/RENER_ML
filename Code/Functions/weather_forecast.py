from datetime import datetime
import maya
import requests
import pandas as pd


def ManageDateTime(ds_obj):
    return datetime.strftime(maya.parse(ds_obj).datetime(), "%Y-%m-%d %H:%M:%S")

def ManageDateMinute(ds_obj):
    return datetime.strftime(maya.parse(ds_obj).datetime(), "%Y-%m-%d %H:%M:00")
def ManageDateHour(ds_obj):
    return datetime.strftime(maya.parse(ds_obj).datetime(), "%Y-%m-%d %H:00:00")

class InvalidArgument(Exception):
    "Input Either City name or Lat AND Lon for city"
    pass
def GetWeatherData(city = "None", lon = "no", lat= "no", start_date= "yesterday", end_date= "today"):
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

def ProcessWeatherData(weather_data):
    weather_data_real = weather_data[weather_data.type == "real"]
    weather_data_real["check"] = weather_data_real["temperature_2m"].apply(lambda x: pd.isna(x) != True)
    weather_data_real = weather_data_real[weather_data_real.check == True]
    real_ds = weather_data_real.ds.tolist()

    weather_data_forecast = weather_data[weather_data.type == "forecast"]
    weather_data_forecast["check"] = weather_data_forecast["ds"].apply(lambda x: x not in real_ds)
    weather_data_forecast = weather_data_forecast[weather_data_forecast.check == True]

    weather_data = pd.concat([weather_data_real.drop(["check"], axis = 1), weather_data_forecast.drop(["check"], axis = 1)], ignore_index= True)

    weather_data = weather_data[["ds", 'temperature_2m', 'cloudcover', 'shortwave_radiation',
       'direct_radiation', 'diffuse_radiation', 'direct_normal_irradiance']]
    weather_data.columns = ["ds_hour", 'temperature_2m', 'cloudcover', 'shortwave_radiation',
        'direct_radiation', 'diffuse_radiation', 'direct_normal_irradiance']
    

    return weather_data

def TrainProphetWeather(train_data, regressor_names):
    from prophet import Prophet

    m = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale = 0.05)



    for name in regressor_names:
        m.add_regressor(name)

    m.fit(train_data)
    from prophet.utilities import regressor_coefficients
    add_coef = regressor_coefficients(m)

    return m, add_coef

