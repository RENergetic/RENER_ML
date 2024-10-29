# Pipelines Description

This serves as the description of all the components and functions inside the pipelines. This shows in full detail the way the pipelines work. 


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Pipelines</summary>
  <ol>
    <li>
      <a href="#training">Training</a>
    </li>
    <li>
      <a href="#forecast">Forecasting</a>
    </li>
    <li>
      <a href="#monitor">Monitor</a>
    </li>

  </ol>
</details>

## Training

The training pipeline works as follows:

![alt text](/figures/Training%20Pipeline.png)

The purple boxes are function based tasks, the blue boxes are check tasks. As shown in the *check Prophet* box task, its a boolean tasks which divides the pipeline into two. In the case of the training, it separates whether a new model is trained or it used a previous model.

The compare metric used is based on one of three options given by default: r2 ($R^2$), mae (Mean Absolute Error), RMSE (Root Mean Square Error). Setting a model means to make the model trained the default model to forecast for that measurement-asset, if the model is not set, it will still be saved into the MinIO server but the next forecasting run will not be used to make the predictions.

The import data and weather, processing is done only once per measurement, however for each asset a different model is trained. The loop is done as:

```python

with dsl.ParallelFor(get_list_task.output) as asset:

```

This way, several models can ve trained in parallel, accelerating the run, this is way, the resources are also limited, to run more models in parallel. 

## Forecast

This is the diagram of the forecast pipeline:

![alt text](/figures/Forecast%20Pipeline.png)

There are three checks in this pipeline:

- Check availability: It checks if there is enough recent data to make new forecasts.
- Check Send: This allow to make forecasts without actually send the data to the database (to test other models)
- Check Notification: This send a notification if the values of the forecast go beyond a certain threshold.

If there is not enough available data, then the forecast will not happen.

## Monitor

The monitor pipeline compares the real and the forecast data.

![alt text](/figures/Monitor%20Pipeline.png)

If the metrics are below a certain threshold, then a retraining is toggled.




