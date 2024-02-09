
import kfp.components as comp
from kfp import compiler, dsl
from kfp import dsl
from kubernetes.client.models import V1EnvVar

import time

from components.get_data import GetData
from components.download_weather_data_open_meteo import DownloadWeatherData_OpenMeteo
from components.download_data_from_influx_db import DownloadDataFromInfluxDB
from components.calculate_forecast_metrics import CheckModelInList
from components.utils import Get_List_Assets, GetListofDict, Check
from components.process_data import ProcessData




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
from components.save_model import SetModel

def REN_Train_Model_Pipeline(url_pilot:str,
    diff_time:int,
    filter_vars:list = [],
    filter_case:list = [],
    path_minio = "minio-kubeflow-renergetic.apps.dcw1-test.paas.psnc.pl",
    access_key="minio",
    secret_key="DaTkKc45Hxr1YLR4LxR2xJP2",
    min_date = "5 May 2023",
    max_date = "today",
    dict_assets : dict = {
        "electricity_meter": ["building1", "building2"],
        "heat_meter": ["building1", "building2"]
    },
    key_measurement = "energy",
    type_measurement = "simulated",
    pilot_name = "Virtual",
    hourly_aggregate = "no",
    minute_aggregate = "no",
    forecast_models = "all",
    num_days: int = 1,
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

    check_model_list_op = comp.create_component_from_func(
        CheckModelInList, output_component_file= "model_list_component.yaml"
    )

    get_list_assets_op = comp.create_component_from_func(
        Get_List_Assets, output_component_file= "get_list_component.yaml"
    )
    
    get_list_measurements_op = comp.create_component_from_func(GetListofDict, output_component_file="list_dicts_components.yaml")

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

    compare_models_op = comp.create_component_from_func(CheckSetModels, output_component_file="compare_and_set_models.yaml", packages_to_install=["pandas", "scikit-learn"])

    check_generic_op = comp.create_component_from_func(Check, output_component_file= "generic_check.yaml")

    set_model_op = comp.create_component_from_func(SetModel, packages_to_install=["maya", "minio"], output_component_file="set_models_component.yaml")

    ########## BEGIN PIPELINE DEFINITION  ###############
    
    ### STEP 1: DOWNLOAD WEATHER DATA #######################
    # In this step weather data is downloaded as it should be somewhat genereic for the same pilot. #

    download_weather_influx_task = download_weather_influx_db_op(timestamp)

    download_weather_open_meteo_task = download_weather_open_meteo_op(download_weather_influx_task.output, pilot_name, min_date)


    # LOOP OVER MEASUREMENTS #

    list_measurements_task = get_list_measurements_op(dict_assets)

    with dsl.ParallelFor(list_measurements_task.output) as measurement:

        # STEP 2: DOWNLOAD AND PROCESS TIME SERIES DATA ###
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
        get_list_task = (get_list_assets_op(measurement, dict_assets))

        
        with dsl.ParallelFor(get_list_task.output) as asset:

            # STEP 3: TRAIN ALL MODELS
            
            # PROPHET SIDE - CHANGE THE FUNCTION FOR A COMPONENT AND LOAD COMPONENT

            check_prophet_task = check_model_list_op("prophet", forecast_models)
            
            with dsl.Condition(check_prophet_task.output == True):
                forecast_train_prophet_task = (
                train_prophet_op(
                    process_task.output, 
                    download_weather_open_meteo_task.output,  
                    diff_time, 
                    num_days,
                    pilot_name,
                    measurement,
                    asset,
                    path_minio,
                    access_key,
                    secret_key
                ).add_env_variable(env_var)
                .set_memory_request('2Gi')
                .set_memory_limit('4Gi')
                .set_cpu_request('2')
                .set_cpu_limit('4')
                )
            with dsl.Condition(check_prophet_task.output == False):
                load_and_forecast_prophet_task = load_and_forecast_prophet_op(
                    download_weather_open_meteo_task.output,  # INPUT DATA
                    diff_time, num_days, # TIME SERIES CRITERIA
                    path_minio, access_key, secret_key, # MINIO CONFIG
                    pilot_name, measurement, asset) # NAME CONFIG
            

            merge_prophet_task = merge_forecast_op(
                forecast_train_prophet_task.outputs["forecast_data"],
                forecast_train_prophet_task.outputs["results"]
            )
            merge_prophet_task = merge_forecast_op(
                load_and_forecast_prophet_task.outputs["forecast_data"],
                load_and_forecast_prophet_task.outputs["results"]
            )

            # TRANSFORMERS SIDE - CHANGE THE FUNCTION FOR A COMPONENT AND LOAD COMPONENT

            check_transformers_task = check_model_list_op("transformers", forecast_models)

            with dsl.Condition(check_transformers_task.output == True):
                forecast_train_transformer_task = (
                train_transformer_op(
                    process_task.output, 
                    download_weather_open_meteo_task.output, 
                    diff_time, 
                    num_days,
                    pilot_name,
                    measurement, 
                    asset, 
                    n_epochs,
                    path_minio,
                    access_key,
                    secret_key
                ).add_env_variable(env_var)
                .set_memory_request('2Gi')
                .set_memory_limit('4Gi')
                .set_cpu_request('2')
                .set_cpu_limit('4')
                )
            
            with dsl.Condition(check_transformers_task.output == False):
                load_and_forecast_transformer_task = load_and_forecast_transformer_op(
                    process_task.output,download_weather_open_meteo_task.output,
                    diff_time, num_days,
                    path_minio, access_key, secret_key, 
                    pilot_name,measurement, asset
                    )
            
            merge_transformers_task = merge_forecast_op(
                forecast_train_transformer_task.outputs["forecast_data"],
                forecast_train_transformer_task.outputs["results"]
            )
            merge_transformers_task = merge_forecast_op(
                load_and_forecast_transformer_task.outputs["forecast_data"],
                load_and_forecast_transformer_task.outputs["results"]
            )

            # LSTM SIDE
            
            check_lstm_task = check_model_list_op("lstm", forecast_models)

            with dsl.Condition(check_lstm_task.output == True):
                forecast_train_lstm_task = (
                train_lstm_op(
                    process_task.output, download_weather_open_meteo_task.output, 
                    # TS VARS
                    diff_time, 
                    num_days,
                    # TS NAMES
                    pilot_name,
                    measurement, 
                    asset,
                    # MINIO VARS
                    path_minio,
                    access_key,
                    secret_key,
                    # MODEL OPTIONS
                    n_epochs
                ).add_env_variable(env_var)
                .set_memory_request('2Gi')
                .set_memory_limit('4Gi')
                .set_cpu_request('2')
                .set_cpu_limit('4')
                )
            
            with dsl.Condition(check_lstm_task.output == False):
                load_and_forecast_lstm_task = load_and_forecast_lstm_op(
                                                                        process_task.output, download_weather_open_meteo_task.output,
                                                                        diff_time, num_days,
                                                                        path_minio, access_key, secret_key, 
                                                                        pilot_name, measurement, asset)
            
            merge_lstm_task = merge_forecast_op(
                forecast_train_lstm_task.outputs["forecast_data"],
                forecast_train_lstm_task.outputs["results"]
            )
            merge_lstm_task = merge_forecast_op(
                load_and_forecast_lstm_task.outputs["forecast_data"],
                load_and_forecast_lstm_task.outputs["results"]
            )


            # CHECK METRICS

            compare_task = compare_models_op(
                process_task.output,
                merge_prophet_task.outputs["output_forecast"],
                merge_lstm_task.outputs["output_forecast"],
                merge_transformers_task.outputs["output_forecast"]
            )

            # SAVE MODEL

            prophet_set_check_task = check_generic_op(compare_task.output, "prophet")
            lstm_set_check_task = check_generic_op(compare_task.output, "lstm")
            transformers_set_check_task = check_generic_op(compare_task.output, "transformers")

            with dsl.Condition(prophet_set_check_task.output == True):
                set_model_task = set_model_op(forecast_train_prophet_task.outputs["results"],
                                            compare_task.output,
                                            path_minio,access_key,secret_key,
                                            pilot_name,measurement,asset)
            with dsl.Condition(lstm_set_check_task.output == True):
                save_model_task = set_model_op(forecast_train_lstm_task.outputs["results"],
                                               compare_task.output,
                                                           path_minio,
                                                           access_key,
                                                           secret_key,
                                                           pilot_name,
                                                           measurement,
                                                           asset)
            with dsl.Condition(transformers_set_check_task.output == True):
                save_model_task = set_model_op(forecast_train_transformer_task.outputs["results"],
                                               compare_task.output,
                                                           path_minio,
                                                           access_key,
                                                           secret_key,
                                                           pilot_name,
                                                           measurement,
                                                           asset)
            



compiler.Compiler().compile(pipeline_func = REN_Train_Model_Pipeline, package_path ="Train_Model_Pipeline.yaml")