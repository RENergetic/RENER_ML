import kfp.components as comp
from kfp import compiler, dsl
from kubernetes.client.models import V1EnvVar
from components.check_forecast import CheckDataAvailability, CheckModelSet
import time
from components.get_data import GetData
from components.process_data import ProcessData
from components.utils import Get_List_Assets, GetListofDict, Check
from components.report_utils import ReportError
from components.compare_maintenance import CalculateMetrics, BoolMetrics, Retrain

process_data_op = comp.create_component_from_func(
    ProcessData, packages_to_install= ["maya", "pandas", "icecream", "tqdm"], output_component_file= "process_data_op_component.yaml"
)
download_data_op = comp.create_component_from_func(
        GetData, packages_to_install = ["requests", "numpy", "maya","pandas", "icecream", "tqdm", "retry"], output_component_file = "download_data_op_component.yaml")  
check_data_availability_op = comp.create_component_from_func(CheckDataAvailability, 
                                                            output_component_file="check_data_availability_forecast.yaml", 
                                                            )
check_model_set_op = comp.create_component_from_func(
    CheckModelSet, output_component_file= "check_model_set_component.yaml", packages_to_install = ["minio"]
)
get_list_measurements_op = comp.create_component_from_func(GetListofDict, output_component_file="list_dicts_components.yaml")
get_list_assets_op = comp.create_component_from_func(
        Get_List_Assets, output_component_file= "get_list_component.yaml"
    )
error_report_op = comp.create_component_from_func(
    ReportError, 
    output_component_file = "error_report_component.yaml", 
    packages_to_install=[])
compare_op  = comp.create_component_from_func(CalculateMetrics, output_component_file="compare_and_toggle.yaml", 
                                                       packages_to_install=["pandas", "icecream", "numpy", "scikit-learn"])
bool_retrain_op = comp.create_component_from_func(BoolMetrics, output_component_file="bool_metrics.yaml", packages_to_install=[])
retrain_op = comp.create_component_from_func(Retrain, output_component_file="retrain_component.yaml")


def MonitoringPipeline(
        url_pilot,
        pilot_name,
        dict_assets : dict = {
            "electricity_meter": ["building1", "building2"],
            "heat_meter": ["building1", "building2"]
        },
        key_measurement = "energy",
        hourly_aggregate = "no",
        minute_aggregate = "no",
        min_date: str = "1 month ago",
        max_date: str = "today",
        diff_time: int = 60,
        availability_minimum: float = 50,
        num_days_check = 14,
        filter_vars: list = [],
        filter_case:list = [],
        metrics_threshold = {
            "Coverage": 50
        },
        timestamp: float = time.time()
):

    env_var = V1EnvVar(name='HOME', value='/tmp')
    
    list_measurements_task = get_list_measurements_op(dict_assets, timestamp)

    with dsl.ParallelFor(list_measurements_task.output) as measurement:
        download_real_task = (download_data_op(measurement, 
                                        min_date, 
                                        max_date, 
                                        url_pilot,
                                        pilot_name, 
                                        "real", 
                                        key_measurement, 
                                        filter_vars, 
                                        filter_case).add_env_variable(env_var)
                                .set_memory_request('2Gi')
                                .set_memory_limit('4Gi')
                                .set_cpu_request('2')
                                .set_cpu_limit('4'))
        
        download_forecast_task = (download_data_op(measurement, 
                                        min_date, 
                                        max_date, 
                                        url_pilot,
                                        pilot_name, 
                                        "forecasting", 
                                        key_measurement, 
                                        filter_vars, 
                                        filter_case).add_env_variable(env_var)
                                .set_memory_request('2Gi')
                                .set_memory_limit('4Gi')
                                .set_cpu_request('2')
                                .set_cpu_limit('4'))
        
        process_real_task = (process_data_op(download_real_task.outputs["output_data_forecast"], 
                            hourly_aggregate,
                            minute_aggregate,
                            min_date, 
                            max_date)
                            .set_memory_request('2Gi')
                                .set_memory_limit('4Gi')
                                .set_cpu_request('2')
                                .set_cpu_limit('4'))
        
        process_forecast_task = (process_data_op(download_forecast_task.outputs["output_data_forecast"], 
                            hourly_aggregate,
                            minute_aggregate,
                            min_date, 
                            max_date)
                            .set_memory_request('2Gi')
                                .set_memory_limit('4Gi')
                                .set_cpu_request('2')
                                .set_cpu_limit('4'))
        
        
        get_list_task = (get_list_assets_op(measurement, dict_assets, timestamp))
        with dsl.ParallelFor(get_list_task.output) as asset:
            check_availability_real_task = check_data_availability_op(process_real_task.output, asset, diff_time, availability_minimum, num_days_check)
            check_availability_forecast_task = check_data_availability_op(process_forecast_task.output, asset, diff_time, availability_minimum, num_days_check)
    
                
            with dsl.Condition(check_availability_real_task.output == False):
                report_error_task = error_report_op()
            
            with dsl.Condition(check_availability_forecast_task.output == False):
                report_error_task = error_report_op()
            
            with dsl.Condition(check_availability_real_task.output == True and check_availability_forecast_task.output == True):
                compare_task = compare_op(process_real_task.output, 
                                    process_forecast_task.output,
                                    asset)
                
                bool_task = bool_retrain_op(compare_task.output, metrics_threshold)

                if bool_task.output == True:
                    retrain_task = retrain_op()
                

                

compiler.Compiler().compile(pipeline_func = MonitoringPipeline, package_path ="Monitoring_Pipeline.yaml")
