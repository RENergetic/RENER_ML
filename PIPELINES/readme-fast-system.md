# Fast System

In this readme the FAST System is explained alongside an example for each of the components. The FAST System is the methodology incorporated into the KubeFlow pipelines so that new developments can be made without any changes to the already developed pipelines. 

![alt text](/figures/fast_explanation.png)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Components</summary>
  <ol>
    <li>
      <a href="#forge">Forge</a>
    </li>
    <li>
      <a href="#augur">Augur</a>
    </li>
    <li>
      <a href="#surveil">Surveil</a>
    </li>
    <li>
      <a href="#surveil">Translate</a>
    </li>

  </ol>
</details>

## Forge

The forge class will allow any developers to include new processing to the KubeFlow Pipelines. This processing happens in the process task of both training and forecast pipelines and when the data is being sent and checked. This is why it is not part of the monitoring pipeline. 

Let's consider a time series with unusual high values that needed to be capped. This is a procedure not included in the pipelines, so the only way to use it is through a custom Forge Class.

``` python

import dill

class Forge:
    ''' Dynamic class for timeseries processing before model training. This is the initial forge class'''

    def __init__(self, process_type = None):
        self.process_type = process_type


def ProcessForge(self, data_, value_max = 1000):
    ''' This function inputs a pandas dataframe and return a pandas df as well'''
    data_ = data_[data_["y"] < value_max].reset_index()
    return data_


Forge.process = classmethod(ValueLimit)

forge_limit = Forge()

with open('forge_osr.pkl', 'wb') as outp:
    dill.dump(forge_limit, outp)

```

The class is saved into a pickle file that will be uploaded into a folder of the MinIO server to be used in the pipelines. With that pickle file everything needed is there. 

***IMPORTANT***: **The only requirement is that the output of the function is a pandas dataframe with two columns: "ds" as the string datetime variable and "y" as the time series value.**


## Augur

The augur dynamic class serves as the custom algorithm class for model training and forecasting. It is the central piece of the FAST System. As an example of this class, the [nhits model](https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html#usage-example) is being used. 

``` python

class Augur:
    ''' Dynamic class for timeseries forecast '''

    def __init__(self, model_name = None, type_model = None, training_date = None, metric = {}, model = None):
        self.model_name = model_name
        self.type_model = type_model
        self.training_date = training_date
        self.metric = metric
        self.model = model

def Train_NHITS(self,data_to_train,
                index_col = "pilot",
                list_futr_exog_list = [],
                format_ = "%Y-%m-%d %H:%M:%S",
                test_size = {
                    "days": 1
                }):
    
    import maya
    from datetime import datetime
    from neuralforecast import NeuralForecast
    from neuralforecast.models import  NHITS

    
    max_date = max(data_to_train[ds_column])
    date_training_cup = maya.parse(max_date)

    Y_train_df = data_to_train[Y_df.ds <= date_training_cup]
    Y_test_df = data_to_train[Y_df.ds >= date_training_cup]

    # Fit and predict with NBEATS and NHITS models
    horizon = Y_test_df.shape[0]-24
    # horizon = 10
    models = [NHITS(input_size=2*horizon, 
                    futr_exog_list = list_futr_exog_list, 
                    h=horizon, max_steps=40)]
    nf = NeuralForecast(models=models, freq='1H')
    nf.fit(df=Y_train_df)

    nf.save(path=f'model',
                    model_index=None, 
                    overwrite=True,
                    save_dataset=True)

    self.model = nf


def PredictNHITS(self, prev_data):

    if "unique_id" not in prev_data.columns:
        prev_data["unique_id"] = 1

    model_ =  self.model
    y_pred = model_.predict(prev_data).reset_index()
    y_pred = y_pred[["ds", "NHITS"]]
    y_pred.columns = ["ds", "yhat"]

    return y_pred

def LoadNHITS(self):

    from neuralforecasting.core import NeuralForecast

    nf = NeuralForecast.load("model")

    self.model = nf


Augur.train = classmethod(Train_NHITS)
Augur.predict = classmethod(PredictNHITS)
Augur.load = classmethod(LoadNHITS)
augur = Augur("NHITS", "Neural Network")
with open('augur_specific.pkl', 'wb') as outp:
    dill.dump(augur, outp)

```

The model will be saved with all the files in a folder named "model". All the files in that folder will be the ones sent to the MinIO server. Then all this files will be downloaded for the forecasting and then the model will be loaded with the custom load function. 

**REQUIREMENTS**

- The Train function doesn't require any specific returns as the only thing necessary is to save the model in a "model" folder. 
- The load function only needs to save the model as part of the attributes of the class.
- The predict function will have as input a dataframe with "ds" and "y" as columns. It may have other inputs but they need to be dealt as default values. The output will be a *pandas dataframe* with the columns "ds" and "yhat". *NOTE: The name of the value forecasted is not y but yhat.*


## Surveil

Surveil allows to create custom metrics for monitoring the models or choosing the best model in training. Let's imagine a coverage metric, similar to what a accuracy metric would be for a classification model. The prediction is considered correct if the real value is below 120% of the forecast.

```python
class Surveil:
    ''' Dynamic class for metrics '''

    def __init__(self, metric_name, type_metric):
        self.metric = metric_name
        self.type_metric = type_metric


def SurveilData(data_, name_var):

    import pandas as pd
    import numpy as np

    def CustomMetric(value, real):
        ''' 
        Custom Metric for Coverage
        '''

        return real < (value *1.2)

    list_values = []

    for index, row in data_.iterrows():
        list_values.append(CustomMetric(row[name_var], row["y"]))
    
    metric_value = np.round(sum(list_values)*100/data_.shape[0],2)

    return metric_value

Surveil.check = classmethod(SurveilData)
surveil = Surveil("coverage", "above")
with open("surveil.pkl", "wb") as outp:
    dill.dump(surveil, outp)

```

**REQUIREMENTS**

- The surveil check method will only return one numerical value. There is no limit to the format but a percentage rounded to the second value is recommended. 
- Aside from that it is required to add one of to options, "above", which means that the best value will be the highest, or "below" which means the lowest value will be best. 

## Translate

This class reunites all files to explain or add data visualizations for the forecasts. For instance, let's say it is necessary to include a plot of the forecast values to be checked later on:

```python

class Translate:

    ''' Dynamic class for translate '''

    def __init__(self, name_tr = None):
        self.name = name_tr

def PlotTranslate(data_):

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        x = data_["ds"], y = data_["yhat"], name = "forecast"
    )

    fig.write_html(
        os.path.join("translate", "image_1.html")
    )

Translate.translation = classmethod(PlotTranslate)

translator = Translate()

with open("translator.pkl", "wb") as outp:
    dill.dump(translator, outp)

```

Similar to the training process, all the files will be saved in a folder *"translate"*. After which all those files will be saved in the MinIO Server, to later be accessed.



