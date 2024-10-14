# KubeFlow Development

This serves as a general description of the Kubeflow Development for pipelines with regards to RENergetic. 

Recommended to look as well at the [official guide for KubeFlow](https://v1-0-branch.kubeflow.org/docs/pipelines/)


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#how-to-upload-a-pipeline-in-kubeflow">Pipeline Code</a>
    </li>
    <li>
      <a href="#how-to-create-a-pipeline-function">Pipeline Functions</a>
    </li>
    <li>
      <a href="#other-considerations">Others</a>
    </li>

  </ol>
</details>

## How to upload a pipeline in KubeFlow?

In order to upload a pipeline into KubeFlow, a *.yaml* file is necessary in order to create a new pipeline or to upload a new version of an already existent pipeline. The code that generates this file in Python is as follows:

``` python
from kfp import compiler
compiler.Compiler().compile(pipeline_func = REN_Train_Model_Pipeline, package_path ="Train_Model_Pipeline.yaml")
```

The **package path** is the path where the yaml file will be saved and **pipeline_func** is the function of the pipeline. With this, any existing Pipeline Function can be turned into a yaml file for KF.

## How to create a pipeline function?

This part shows the basic code reference as how to create a pipeline function. For this, we need to differentiate between components and tasks.

- ***COMPONENTS***: Lines of code to be executed. This would be similar to a python funcion. It has been compacted to be run by KubeFlow. 
- ***TASKS***: Specific execution of components. It dictates the flow of the pipeline. 

**A pipeline in KubeFlow is a series of tasks executions.**

This is an example of a dummy pipeline:

``` python 
def DummyPipeline(
        value_1: int,
        value_2: str = "",
        value_3: int = 8
):
    dummy_op = comp.create_component_from_func(
        DummyFunction, output_component_file = "dummy_op_component.yaml")
    
    dummy_task_1 = dummy_op(value_1, "")
    dummy_task_2 = dummy_op(value_1, value_2)
    dummy_task_3 = dummy_op(value_3, dummy_task_1.output)

compiler.Compiler().compile(pipeline_func = DummyPipeline, package_path ="Dummy_Pipeline.yaml")
```


With the line:

``` python 
dummy_op = comp.create_component_from_func(
        DummyFunction, output_component_file = "dummy_op_component.yaml")
```

The code of the funcion *DummyFunction* is turned into a KubeFlow Component. Then, *dummy_task_1*, *dummy_task_2*, etc... are tasks executions of that operation. Because *dummy_task_3* has an output of *dummy_task_1* as part of the arguments of the function, *dummmy_task_3* will be executed **AFTER** *dummmy_task_1*.

The code of *DummyFunction* is: 

```python 
def DummyFunction(argument_1: int, argument_2: str) -> str:
    if argument_2 != "":
        raise ValueError
    else:
        print("Who cares")
    
    if argument_1 < 1000:
        print(argument_1 * 2)
    else:
        print(argument_1)
    
    return argument_2
```

The inputs and outpus of the function are defined, which is a recommended practice (although not always necessary)


## Other considerations

This shows the basic principles of KubeFlow Development. However, in order to understand the RENergetic Pipelines there are a couple more things to understand.

- **Custom Images**

Some of the components used custom docker images of python ([click here for more info](https://www.kubeflow.org/docs/components/notebooks/container-images/)) in order to ensure faster runs. The way this changes in the creation of the component is through the argument *base_image*. 

```python
train_lstm_op = comp.create_component_from_func(
        ForecastLSTM, base_image= "adcarras/ren-docker-forecast:0.0.1",packages_to_install=["darts==0.27.2","fuckit"], output_component_file= "forecast_lstm_component.yaml"
    )
```

- **If / Loops**

Some of the executions require if/else conditions or Loops. If/Else are set up as follows,

 ```python 
with dsl.Condition(check_set_task.output == True):
```

while For loops are set up as:

```python 
with dsl.ParallelFor(list_measurements_task.output) as measurement:
```

- ***Variables and Resources management***

Some tasks might use a lot of resources to run in parallel so it might be necessary to limit the resources used. The following code shows how to:

``` python
download_task = (download_data_op(measurement, min_date, max_date, url_pilot,pilot_name, type_measurement, key_measurement, filter_vars, filter_case).add_env_variable(env_var)
                            .set_memory_request('2Gi')
                            .set_memory_limit('4Gi')
                            .set_cpu_request('2')
                            .set_cpu_limit('4'))
```


