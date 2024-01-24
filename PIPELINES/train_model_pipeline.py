
import kfp.components as comp
from kfp import compiler, dsl
from kfp import dsl
from kubernetes.client.models import V1EnvVar

import time

from components.get_data import GetData
from components.download_weather_data_open_meteo import DownloadWeatherData_OpenMeteo
from components.download_data_from_influx_db import DownloadDataFromInfluxDB
from components.calculate_forecast_metrics import CalculateForecastMetrics
from components.get_thresholds import GetThresholds
from components.get_list_assets import Get_List_Assets
from components.process_data import ProcessData
from components.check_send_forecast import CheckSendForecast
from components.send_forecast import SendForecast
from components.check_send_notification import CheckSendNotification
from components.send_notification import SendNotification




# NEW Components
from components.train_forecast_prophet import ForecastProphet
from components.train_forecast_transformer import ForecastTransformer
from components.train_forecast_lstm import ForecastLSTM
from components.load_prophet_forecast import LoadAndForecastProphet
from components.load_transformer_forecast import LoadAndForecastTransformer
from components.load_lstm_forecast import LoadAndForecastLSTM
from components.download_file_minio import DownloadFileMinio
from components.merge_forecasts import MergeForecast
from components.compare_models import CheckSetModels
from components.save_model import ProduceModel

def REN_Train_Model_Pipeline(url_pilot,
    diff_time:int,
    filter_vars:list = [],
    filter_case:list = [],
    path_minio = "minio-kubeflow-renergetic.apps.dcw1-test.paas.psnc.pl",
    access_key="minio",
    secret_key="DaTkKc45Hxr1YLR4LxR2xJP2",
    min_date = "5 May 2023",
    max_date = "today",
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
    mae_threshold:float = 1000000,
    n_epochs: int = 10,
    timestamp: float = time.time()
    ):

    env_var = V1EnvVar(name='HOME', value='/tmp')
    download_data_op = comp.create_component_from_func(
        GetData, packages_to_install = ["requests", "numpy", "maya","pandas", "icecream", "tqdm", "retry"], output_component_file = "download_data_op_component.yaml")
    download_weather_open_meteo_op = comp.create_component_from_func(
        DownloadWeatherData_OpenMeteo, output_component_file= "open_meteo_component.yaml", packages_to_install=["requests", "numpy", "maya","pandas", "icecream", "tqdm", "retry", "pyarrow"]
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

    download_file_minio_op = comp.create_component_from_func(
        DownloadFileMinio, packages_to_install=["minio"], output_component_file= "download_minio_component.yaml"
    )

    train_prophet_op = comp.create_component_from_func(
        ForecastProphet, base_image= "adcarras/ren-docker-forecast:0.0.1", output_component_file= "forecast_prophet_component.yaml"
    )

    train_transformer_op = comp.create_component_from_func(
        ForecastTransformer, base_image= "adcarras/ren-docker-forecast:0.0.1", output_component_file= "forecast_transformer_component.yaml"
    )

    train_lstm_op = comp.create_component_from_func(
        ForecastLSTM, base_image= "adcarras/ren-docker-forecast:0.0.1", output_component_file= "forecast_lstm_component.yaml"
    )

    load_and_forecast_prophet_op = comp.create_component_from_func(
        LoadAndForecastProphet, base_image= "adcarras/ren-docker-forecast:0.0.1", output_component_file= "load_and_forecast_prophet.yaml"
    )

    load_and_forecast_transformer_op = comp.create_component_from_func(
        LoadAndForecastTransformer, base_image= "adcarras/ren-docker-forecast:0.0.1", output_component_file= "load_and_forecast_transformer.yaml"
    )

    load_and_forecast_lstm_op = comp.create_component_from_func(
        LoadAndForecastLSTM, base_image= "adcarras/ren-docker-forecast:0.0.1", output_component_file= "load_and_forecast_lstm.yaml"
    )

    merge_forecast_op = comp.create_component_from_func(
        MergeForecast, packages_to_install=["pandas", "pyarrow"], output_component_file="merge_forecast_component.yaml"
    )

    check_send_forecast_op = comp.create_component_from_func(
        CheckSendForecast, packages_to_install=[], output_component_file= "check_send_forecast_component.yaml"
    )
    send_forecast_op = comp.create_component_from_func(SendForecast, packages_to_install=["requests", "numpy", "maya","pandas", "icecream", "tqdm", "minio", "boto3"], output_component_file= "send_forecast_comp.yaml")

    check_send_notification_op = comp.create_component_from_func(
        CheckSendNotification, output_component_file= "check_send_notification.yaml"
    )

    send_notification_op = comp.create_component_from_func(
        SendNotification, packages_to_install=["pandas", "maya", "fuckit", "icecream"],output_component_file="send_notification.yaml"
    )

    set_models_op = comp.create_component_from_func(CheckSetModels, output_component_file="compare_and_set_models.yaml", packages_to_install=["pandas", "scikit-learn"])

    produce_model_op = comp.create_component_from_func(ProduceModel, output_component_file= "produce_model_comp.yaml")
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

        
        with dsl.ParallelFor(get_list_task.output) as asset:
            
            # PROPHET SIDE - CHANGE THE FUNCTION FOR A COMPONENT AND LOAD COMPONENT

            check_forecast_task = (check_metrics_forecast_op(download_task.outputs["output_data_metric"], asset, mae_threshold = mae_threshold)
                                   .set_memory_request('2Gi')
                                    .set_memory_limit('4Gi')
                                    .set_cpu_request('2')
                                    .set_cpu_limit('4'))
            
            with dsl.Condition(check_forecast_task.output == True):
                forecast_train_prophet_task = (
                train_prophet_op(
                    process_task.output, 
                    download_weather_open_meteo_task.output,  
                    diff_time, 
                    num_days, 
                    asset
                ).add_env_variable(env_var)
                .set_memory_request('2Gi')
                .set_memory_limit('4Gi')
                .set_cpu_request('2')
                .set_cpu_limit('4')
                )
            with dsl.Condition(check_forecast_task.output == False):
                bucket_name = "models_renergetic"
                filename = "prophet_{asset_name}.json".format(asset_name = asset)
                download_model_prophet_task = download_file_minio_op(path_minio, access_key, secret_key, bucket_name, filename)
                load_and_forecast_prophet_task = load_and_forecast_prophet_op(download_model_prophet_task.output, download_weather_open_meteo_task.output, 
                                                diff_time, num_days)
            

            merge_prophet_task = merge_forecast_op(
                forecast_train_prophet_task.outputs["forecast_data"]
            )
            merge_prophet_task = merge_forecast_op(
                load_and_forecast_prophet_task.outputs["forecast_data"]
            )

            # TRANSFORMERS SIDE - CHANGE THE FUNCTION FOR A COMPONENT AND LOAD COMPONENT

            with dsl.Condition(check_forecast_task.output == True):
                forecast_train_transformer_task = (
                train_transformer_op(
                    process_task.output, download_weather_open_meteo_task.output, diff_time, num_days, asset, n_epochs
                ).add_env_variable(env_var)
                .set_memory_request('2Gi')
                .set_memory_limit('4Gi')
                .set_cpu_request('2')
                .set_cpu_limit('4')
                )
            
            with dsl.Condition(check_forecast_task.output == False):
                bucket_name = "models_renergetic"
                filename = "transformer_{asset_name}.pt".format(asset_name = asset)
                download_model_transformer_task = download_file_minio_op(path_minio, access_key, secret_key, bucket_name, filename)
                load_and_forecast_transformer_task = load_and_forecast_transformer_op(download_model_transformer_task.output,
                                                                                      process_task.output,
                                                                                      download_weather_open_meteo_task.output,
                                                                                      diff_time, num_days, asset)
            
            merge_transformers_task = merge_forecast_op(
                forecast_train_transformer_task.outputs["forecast_data"]
            )
            merge_transformers_task = merge_forecast_op(
                load_and_forecast_transformer_task.outputs["forecast_data"]
            )

            # LSTM SIDE
            
            with dsl.Condition(check_forecast_task.output == True):
                forecast_train_lstm_task = (
                train_lstm_op(
                    process_task.output, download_weather_open_meteo_task.output, diff_time, num_days, asset, n_epochs
                ).add_env_variable(env_var)
                .set_memory_request('2Gi')
                .set_memory_limit('4Gi')
                .set_cpu_request('2')
                .set_cpu_limit('4')
                )
            
            with dsl.Condition(check_forecast_task.output == False):
                bucket_name = "models_renergetic"
                filename = "lstm_{asset_name}.pt".format(asset_name = asset)
                download_model_lstm_task = download_file_minio_op(path_minio, access_key, secret_key, bucket_name, filename)
                load_and_forecast_lstm_task = load_and_forecast_lstm_op(download_model_lstm_task.output,
                                                                                      process_task.output,
                                                                                      download_weather_open_meteo_task.output,
                                                                                      diff_time, num_days, asset)
            
            merge_lstm_task = merge_forecast_op(
                forecast_train_lstm_task.outputs["forecast_data"]
            )
            merge_lstm_task = merge_forecast_op(
                load_and_forecast_lstm_task.outputs["forecast_data"]
            )


            # CHECK METRICS

            set_model_task = set_models_op(
                process_task.output,
                merge_prophet_task.output,
                merge_lstm_task.output,
                merge_transformers_task.output
            )

            # SAVE MODEL

            with dsl.Condition(set_model_task.output == "prophet"):
                save_model_task = produce_model_op(set_model_task.output)
            with dsl.Condition(set_model_task.output == "lstm"):
                save_model_task = produce_model_op(set_model_task.output)
            with dsl.Condition(set_model_task.output == "transformers"):
                save_model_task = produce_model_op(set_model_task.output)
            



compiler.Compiler().compile(pipeline_func = REN_Train_Model_Pipeline, package_path ="Train_Model_Pipeline.yaml")