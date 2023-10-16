import numpy as np
import pandas as pd
from prophet import Prophet
import itertools
from tqdm import tqdm
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.serialize import model_to_json, model_from_json

def Train_Prophet(train_data, 
									weather_data,
                  metric = "MAE",
                  params_grid = {  
                    'changepoint_prior_scale': [0.001, 0.01],
                    'weekly_seasonality': [False, 5],
                    'daily_seasonality':  [False, 5],
                    'seasonality_prior_scale': [0.01, 0.1],
                    "seasonality_mode": ["multiplicative", "additive"]
                    }, 
                  exogenous_data = pd.DataFrame(),
                  weather_vars:list = ["all"],
                  horizon_span: str = "10 days"):

    """
    
    Trains a prophet model based on train

    Parameters
    ----------
    train_data: Data in the format ["ds", "y"], where ds is a columns in datetime or string = "%Y-%m-%d %H:%M:%S
    weather_data: Weather Data in the format ["ds", values]. The expected values to be included are: shortwave_radiation,temperature_2m,direct_radiation, diffuse_radiation, direct_normal_irradiance. If they are not in the weather data will not be added tothe model.
    metric: Metric which will be used to test the model while training, by default MAE. Options are MAE, RMSE, Coverage
    
    params_grid: Grid of parameters with which the model will be tested. There is a default grid defined.
    exogenous_data: Exogenous data
		weather_vars: This parameter allows to use only a subsample of all the variables in the weather_data if the list is empty none will be used. By default, all variables inside the weather_data will be used.
		horizon_span: Horizon span used in cross validation. As a default 10 days are used. 
		
    Returns
    -------
    best_model: The model with the best metric in training
    best_metric: the value of the metric for the best model
    tuning_results: the whole set of tests done
    """
    
    # Generate all combinations of parameters
    all_params = [dict(zip(params_grid.keys(), v)) for v in itertools.product(*params_grid.values())]
    rmses = []  # Store the RMSEs for each params here
    mae = [] # Store the MAEs for each params here
    coverage = [] # Store the Coveragess for each params here
		
		
		# Processing to merge train data and weather data 
		
    train_data = train_data[["ds", "y"]].groupby("ds").mean().reset_index(level = "ds")
    train_data = pd.merge(train_data, weather_data, on = "ds")
    
    # Processing weather_vars
    
    if weather_vars[0] == "all":
      weather_vars = weather_data.drop(["ds"], axis = 1).columns.values
		
		# Stablishing cutoff points for cross validation
    cutoffs = [pd.to_datetime(train_data.ds).quantile(0.1), pd.to_datetime(train_data.ds).quantile(0.3), pd.to_datetime(train_data.ds).quantile(0.5)]
    
    # Incluir grid search cross fit. 
    
    if metric in ["MAE", "RMSE"]:
        best_metric = float("inf")
    elif metric == "Coverage":
        best_metric = 0
    else:
        metric = "MAE"
        best_metric = float("inf")

    best_model = None
	
		# Processing Exogenous Data
    if exogenous_data.shape[0] != 0:
        try:
          train_data = pd.merge(train_data, exogenous_data, on = "ds")
        except:
          pass


    for params in tqdm(all_params):
        m = Prophet(**params)
        # m.add_country_holidays("Country") --> TO BE ADDED LATER

        for var in weather_vars:
          if var in weather_data.columns.values:
            m.add_regressor(var)
        
        if exogenous_data.shape[0] != 0:
            for col in exogenous_data.columns.values:
                if col not in ["ds", "ds_day", "ds_hour"]:
                    m.add_regressor("people", mode = "multiplicative")

        m.fit(train_data)

        df_cv = cross_validation(m, cutoffs=cutoffs, horizon=horizon_span, parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])
        mae.append(df_p["mae"].values[0])
        coverage.append(df_p["coverage"].values[0])

        if metric == "MAE":
            if mae[-1] < best_metric:
                best_metric = mae[-1]
                best_model = m
        elif metric == "RMSE":
             if rmses[-1] < best_metric:
                best_metric = rmses[-1]
                best_model = m
        elif metric == "Coverage":
            if coverage[-1] > best_metric:
                best_metric = coverage[-1]
                best_model = m
    
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    tuning_results["mae"] = mae
    tuning_results["coverage"] = coverage

    return best_model, best_metric, tuning_results

def PredictFromProphet(model, 
                       weather_data,
                       freq_hourly, 
                       days_ahead = 1):

  """
  Takes a Prophet model and makes the prediction. It

  """

  if freq_hourly <= 1:
      freq_of_hours = np.round(1/freq,0)
      freq = "{num_hours}H".format(num_hours = freq_of_hours)
      periods = np.round(days_ahead*24 / freq_of_hours,0)
  else:
      freq_in_minutes = np.round(60/freq,0)
      freq = "{num_minutes}T".format(num_minutes = freq_in_minutes)
      periods = np.round(days_ahead*24*60 / freq_in_minutes,0)
    
  future = model.make_future_dataframe(periods = periods, freq = freq, include_history = False)
  future["ds"] = future["ds"].apply(str)
  future = pd.merge(future, weather_data, on = "ds")
  forecast = model.predict(future)

  return forecast[["ds", "yhat"]]

def TrainAndPredictProphet(train_data, weather_data, freq_hourly, days_ahead, **kwargs):
  """
  
  Component to train and predict

  """

  for key, value in kwargs.items():
        if key == "metric":
          metric_name = value
        
        if key == "params_grid":
          params_ = value
        
        if key == "exogenous_data":
           exo_data = value
        
        if key == "weather_vars":
           w_var = value
        
        if key == "horizon_span":
           h_span = value

  if "params_" not in locals():
    params_ = {  
                  'changepoint_prior_scale': [0.001, 0.01],
                  'weekly_seasonality': [False, 5],
                  'daily_seasonality':  [False, 5],
                  'seasonality_prior_scale': [0.01, 0.1],
                  "seasonality_mode": ["multiplicative", "additive"]
              }
  
  if "metric" not in locals():
     metric_name = "MAE"
  
  if "exo_data" not in locals():
     exo_data = pd.DataFrame()
  
  if "w_var" not in locals():
     w_var = ["all"]
  
  if "h_span" not in locals():
     h_span = "10 days"
  

  model, metric, tuning_df = Train_Prophet(train_data, weather_data,metric = metric_name,
                                           params_grid= params_, exogenous_data=exo_data,
                                           weather_vars = w_var, horizon_span= h_span)
  preds_test = PredictFromProphet(model, weather_data, freq_hourly, days_ahead)

  return preds_test

def LoadAndForecastProphet(path_to_model, weather_data, freq_hourly, days_ahead):
  """
  
  Component to load the model and predict

  """
  with open(path_to_model, 'r') as fin:
    model = model_from_json(fin.read())  # Load model
  
  preds_test = PredictFromProphet(model, weather_data, freq_hourly, days_ahead)
  
  return preds_test


def SaveProphetModel(model, path_to_model):
  with open(path_to_model, 'w') as fout:
    fout.write(model_to_json(model))  # Save model