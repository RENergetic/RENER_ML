import kfp.components as comp
from kfp import compiler, dsl
from kubernetes.client.models import V1EnvVar

import time

from components.get_data import GetData
from components.download_weather_data_open_meteo import DownloadWeatherData_OpenMeteo
from components.download_data_from_influx_db import DownloadDataFromInfluxDB
from components.get_thresholds import GetThresholds
from components.process_data import ProcessData
from components.load_prophet_forecast import LoadAndForecastProphet
from components.send_forecast import SendForecast
from components.send_notification import SendNotification
from components.check_availability_forecast import CheckDataAvailability
from components.utils import GetListofDict, Get_List_Assets, CheckSendForecast, CheckSendNotification, TypeModelParser
from components.report_utils import ReportError

env_var = V1EnvVar(name='HOME', value='/tmp')
download_data_op = comp.create_component_from_func(
    GetData, packages_to_install = ["requests", "numpy", "maya","pandas", "icecream", "tqdm", "retry"], output_component_file = "download_data_op_component.yaml")
download_weather_open_meteo_op = comp.create_component_from_func(
    DownloadWeatherData_OpenMeteo, output_component_file= "open_meteo_component.yaml", packages_to_install=["requests", "numpy", "maya","pandas", "icecream", "tqdm", "retry", "pyarrow"]
)
download_weather_influx_db_op = comp.create_component_from_func(
    DownloadDataFromInfluxDB, output_component_file="weather_influx_db_component.yaml", packages_to_install=["pandas", "pyarrow"]
)

send_forecast_op = comp.create_component_from_func(SendForecast, packages_to_install=["requests", "numpy", "maya","pandas", "icecream", "tqdm", "minio", "boto3"], output_component_file= "send_forecast_comp.yaml")
send_notification_op = comp.create_component_from_func(
        SendNotification, packages_to_install=["pandas", "maya", "fuckit", "icecream"],output_component_file="send_notification.yaml"
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

check_data_availability_op = comp.create_component_from_func(CheckDataAvailability, output_component_file="check_data_availability_forecast.yaml", packages_to_install=list)
check_send_forecast_op = comp.create_component_from_func(
        CheckSendForecast, packages_to_install=[], output_component_file= "check_send_forecast_component.yaml"
    )
check_send_notification_op = comp.create_component_from_func(
        CheckSendNotification, packages_to_install=[], output_component_file= "check_send_forecast_component.yaml"
    )
error_report_op = comp.create_component_from_func(
    ReportError, 
    output_component_file = "error_report_component.yaml", 
    packages_to_install=[])

list_dicts_measurements_op = comp.create_component_from_func(
    GetListofDict, output_component_file= "list_measurements.yaml"
)

parser_op = comp.create_component_from_func(
    TypeModelParser, output_component_file="parser_model_component.yaml"
)

def ForecastProcessPipeline(
        url_pilot: str,
        pilot_name: str,
        measurements_assets_dict: dict,
        key_measurement: dict,
        hourly_aggregate: str,
        minute_aggregate: str,
        type_measurement: str,
        availability_minimum: int,
        num_days_check: int = 2,
        min_date:str = "2 weeks ago",
        max_date:str = "today",
        diff_time:int = 60,
        filter_vars:list = [],
        filter_case:list = [],
        send_forecast = "yes",
        send_notification = "no",
        timestamp:float = time.time()
):
    

    download_weather_influx_task = download_weather_influx_db_op(timestamp)
    download_weather_open_meteo_task = download_weather_open_meteo_op(download_weather_influx_task.output, pilot_name, min_date)

    list_measurements = list_dicts_measurements_op(measurements_assets_dict)

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
        
        get_list_task = (get_list_op(measurement, measurements_assets_dict))

        check_send_forecast_task = check_send_forecast_op(send_forecast)
        check_send_notification_task = check_send_notification_op(send_notification)

        with dsl.ParallelFor(get_list_task.output) as asset:
            
            get_thresholds_task = get_thresholds_op(url_pilot, pilot_name)

            # CHECK IF FORECAST IS VIABLE
            check_availability_task = check_data_availability_op(process_task.output, asset, diff_time, availability_minimum, num_days_check)

            with dsl.Condition(check_availability_task.output == True):
                model_name = "measurement_asset_prophet"
                get_type_model = parser_op(model_name)
                with dsl.Condition(get_type_model == "Prophet"):
                    forecast_task = LoadAndForecastProphet(model_name, download_weather_open_meteo_task.output, diff_time,
                                                           num_days = num_days_check)
                with dsl.Condition(get_type_model == "No model"):
                    report_error_task = error_report_op()
            with dsl.Condition(check_availability_task.output == False):
                report_error_task = error_report_op()
            
            with dsl.Condition(check_send_forecast_task.output):
                send_forecast_op(forecast_task.outputs["forecast_data"], url_pilot, pilot_name, asset, measurement, key_measurement, num_days_check)
                
            with dsl.Conition(check_send_notification_task.output):
                send_notification_op(forecast_task.outputs["forecast_data"], get_thresholds_task.output, asset,pilot_name, url_pilot)
            
