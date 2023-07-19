import kfp

from typing import NamedTuple
import kfp.components as comp
from kfp import compiler, dsl
from kfp import dsl
from kfp.components import InputPath, OutputPath
from kubernetes.client.models import V1EnvVar

import time

from components.get_data import GetData
from components.download_weather_data_open_meteo import DownloadWeatherData_OpenMeteo
from components.download_data_from_influx_db import DownloadDataFromInfluxDB
from components.calculate_forecast_metrics import CalculateForecastMetrics
from components.get_thresholds import GetThresholds
from components.get_list_assets import Get_List_Assets
from components.process_data import ProcessData
from components.forecast_process import ForecastProcess
from components.predict_from_previous_model import PredictFromPreviousModel
from components.check_send_forecast import CheckSendForecast
from components.send_forecast import SendForecast
from components.check_send_notification import CheckSendNotification
from components.send_notification import SendNotification

def REN_Forecast_Test_Pipeline(url_pilot,
    diff_time:int,
    filter_vars:list = [],
    filter_case:list = [],
    url = "minio-kubeflow-renergetic.apps.dcw1-test.paas.psnc.pl",
    access_key="minio",
    secret_key="DaTkKc45Hxr1YLR4LxR2xJP2",
    min_date = "5 May 2023",
    max_date = "today",
    mode = "no notifications",
    list_measurements:list = ["electricity_meter", "heat_meter"],
    dict_assets : dict = {
        "electricity_meter": ["building1", "building2"],
        "heat_meter": ["building1", "building2"]
    },
    key_measurement = "energy",
    type_measurement = "simulated",
    pilot_name = "Virtual",
    hourly_aggregate = "no",
    minute_aggregate = "no",
    num_days: int = 1,
    send_forecast = "no",
    mae_threshold:float = 1000000,
    mode_prophet: str = "additive",
    daily_seasonality:int = 10,
    weekly_seasonality: int = 10,
    timestamp: float = time.time()
    ):

    env_var = V1EnvVar(name='HOME', value='/tmp')
    download_data_op = comp.create_component_from_func(
        GetData, packages_to_install = ["requests", "numpy", "maya","pandas", "icecream", "tqdm", "discord-webhook", "retry"], output_component_file = "download_data_op_component.yaml")
    download_weather_open_meteo_op = comp.create_component_from_func(
        DownloadWeatherData_OpenMeteo, output_component_file= "open_meteo_component.yaml", packages_to_install=["requests", "numpy", "maya","pandas", "icecream", "tqdm", "discord-webhook", "retry", "pyarrow"]
    )
    download_weather_influx_db_op = comp.create_component_from_func(
        DownloadDataFromInfluxDB, output_component_file="weather_influx_db_component.yaml", packages_to_install=["pandas", "pyarrow"]
    )

    check_metrics_forecast_op = comp.create_component_from_func(
        CalculateForecastMetrics, packages_to_install=["maya", "icecream", "pandas","scikit-learn"], output_component_file = "metric_check_op.yaml"
    )

    get_thresholds_op = comp.create_component_from_func(
        GetThresholds, packages_to_install= ["requests"], output_component_file= "thresholds_component.yaml"
    )
    get_list_op = comp.create_component_from_func(
        Get_List_Assets, output_component_file= "get_list_component.yaml"
    )
    process_data_op = comp.create_component_from_func(
        ProcessData, packages_to_install= ["maya", "pandas", "icecream", "tqdm"], output_component_file= "process_data_op_component.yaml"
    )
    forecast_and_train_data_op = comp.create_component_from_func(
        ForecastProcess, packages_to_install = [],base_image= "adcarras/ren-docker-forecast:0.0.1", output_component_file = "forecast_data_op_component.yaml")
    forecast_data_op = comp.create_component_from_func(
        PredictFromPreviousModel, packages_to_install= [], base_image= "adcarras/ren-docker-forecast:0.0.1", output_component_file= "forecast_from_previous.yaml"
    )
    check_send_forecast_op = comp.create_component_from_func(
        CheckSendForecast, packages_to_install=["discord-webhook"], output_component_file= "check_send_forecast_component.yaml"
    )
    send_forecast_op = comp.create_component_from_func(SendForecast, packages_to_install=["requests", "numpy", "maya","pandas", "icecream", "discord-webhook", "tqdm", "minio", "boto3"], output_component_file= "send_forecast_comp.yaml")

    check_send_notification_op = comp.create_component_from_func(
        CheckSendNotification, output_component_file= "check_send_notification.yaml"
    )

    send_notification_op = comp.create_component_from_func(
        SendNotification, packages_to_install=["pandas", "discord-webhook", "maya", "fuckit", "icecream"],output_component_file="send_notification.yaml"
    )

    # BEGIN PIPELINE DEFINITION

    get_thresholds_task = get_thresholds_op(url_pilot, pilot_name)

    download_weather_influx_task = download_weather_influx_db_op(timestamp)

    download_weather_open_meteo_task = download_weather_open_meteo_op(download_weather_influx_task.output, pilot_name, min_date)

   
    

    with dsl.ParallelFor(list_measurements) as measurement:
        download_task = (download_data_op(measurement, min_date, max_date, url_pilot,pilot_name, type_measurement, key_measurement, filter_vars, filter_case).add_env_variable(env_var)
                            .set_memory_request('2Gi')
                            .set_memory_limit('4Gi')
                            .set_cpu_request('2')
                            .set_cpu_limit('4'))
        process_task = (process_data_op(download_task.outputs["output_data_forecast"], 
                        hourly_aggregate,
                        minute_aggregate,
                        min_date, 
                        max_date)
                        .set_memory_request('2Gi')
                            .set_memory_limit('4Gi')
                            .set_cpu_request('2')
                            .set_cpu_limit('4'))
        
        get_list_task = (get_list_op(measurement, dict_assets))

        check_send_forecast_task = check_send_forecast_op(send_forecast)
        check_send_notification_task = check_send_notification_op(mode)

        
        with dsl.ParallelFor(get_list_task.output) as asset:
            check_forecast_task = (check_metrics_forecast_op(download_task.outputs["output_data_metric"], asset, mae_threshold = mae_threshold)
                                   .set_memory_request('2Gi')
                                    .set_memory_limit('4Gi')
                                    .set_cpu_request('2')
                                    .set_cpu_limit('4'))
            with dsl.Condition(check_forecast_task.output == True):
                forecast_train_task = (forecast_and_train_data_op(process_task.output, download_weather_open_meteo_task.output,
                measurement, 
                url, 
                access_key, 
                secret_key, 
                mode,
                url_pilot,
                diff_time,
                pilot_name,
                send_forecast,
                asset,
                num_days,
                mode_prophet, daily_seasonality, weekly_seasonality).add_env_variable(env_var)
                .set_memory_request('2Gi')
                .set_memory_limit('4Gi')
                .set_cpu_request('2')
                .set_cpu_limit('4')
                )

                with dsl.Condition(check_send_forecast_task.output == True):
                    send_forecast_task = send_forecast_op(forecast_train_task.outputs["forecast_data"], url_pilot, pilot_name, asset, measurement, key_measurement, num_days)

                with dsl.Condition(check_send_notification_task.output == True):
                    send_notification_task = send_notification_op(forecast_train_task.outputs["forecast_data"], get_thresholds_task.output, asset,pilot_name, url_pilot)
            
            with dsl.Condition(check_forecast_task.output == False):
                forecast_task = forecast_data_op(process_task.output, download_weather_open_meteo_task.output, 
                                                 pilot_name, measurement, asset, "", 
                                                 max_date, num_days, diff_time)
                with dsl.Condition(check_send_forecast_task.output == True):
                    send_forecast_task = send_forecast_op(forecast_task.outputs["forecast_data"], url_pilot, pilot_name, asset, measurement, key_measurement, num_days)

                with dsl.Condition(check_send_notification_task.output == True):
                    send_notification_task = send_notification_op(forecast_task.outputs["forecast_data"], get_thresholds_task.output, asset,pilot_name, url_pilot)

compiler.Compiler().compile(pipeline_func = REN_Forecast_Test_Pipeline, package_path ="Forecast_Data_Pipeline.yaml")