from kfp.components import InputPath, OutputPath


def ForecastProphet(input_data_path: InputPath(str), input_weather_path: InputPath(str),
    diff_time,
    num_days:int,
    pilot_name,
    measurement_name,
    asset_name,
    url_minio,
    access_key,
    secret_key,
    load_bool:bool,
    forecast_data_path: OutputPath(str),
    results_path: OutputPath()
    ):

    # LIBRARIES REQUIRED    

    import pandas as pd
    import json
    import numpy as np

    # FUNCTIONS

    from prophet import Prophet
    import itertools
    from tqdm import tqdm
    from minio import Minio
    import maya
    from datetime import datetime
    from prophet.diagnostics import cross_validation
    from prophet.diagnostics import performance_metrics
    from prophet.serialize import model_to_json, model_from_json
    import fuckit
    from icecream import ic

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
        exogenous_data: Exogenous data ["ds", values]
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
        
        train_data["ds"] = train_data["ds"].apply(pd.to_datetime)
        weather_data["ds"] = weather_data["ds"].apply(pd.to_datetime)

        train_data = pd.merge_asof(train_data, weather_data, on = "ds")
        
        # Processing weather_vars
        
        if weather_vars[0] == "all":
            weather_vars = weather_data.drop(["ds"], axis = 1).columns.values
            
        # Stablishing cutoff points for cross validation
        cutoffs = [pd.to_datetime(train_data.ds).quantile(0.1), pd.to_datetime(train_data.ds).quantile(0.3), pd.to_datetime(train_data.ds).quantile(0.5)]
        ic(cutoffs)
        if (max(pd.to_datetime(train_data.ds)) - cutoffs[-1]).days < 10:
            days_horizon = (max(pd.to_datetime(train_data.ds)) - cutoffs[-1]).days
            ic(days_horizon)
            if days_horizon < 2:
                cutoffs = [pd.to_datetime(train_data.ds).quantile(0.1), pd.to_datetime(train_data.ds).quantile(0.3)]
                horizon_span = "3 days"
            else:
                horizon_span = "{hor_d} days".format(hor_d = days_horizon-1)

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
                data = pd.merge(train_data, exogenous_data, on = "ds")
            except:
                pass


        for params in tqdm(all_params):
            m = Prophet(**params)
            # m.add_country_holidays("Country") --> TO BE ADDED LATER

            for var in weather_vars:
                if var in weather_data.columns.values:
                    m.add_regressor(var)

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
                            days_ahead:int = 1):

        """
        Takes a Prophet model and makes the prediction. It

        """

        print(type(days_ahead))
        print(days_ahead)

        if freq_hourly <= 1:
            freq_of_hours = np.round(1/freq_hourly,0)
            freq = "{num_hours}H".format(num_hours = freq_of_hours)
            periods = np.round(days_ahead*24 / freq_of_hours,0)
        else:
            freq_in_minutes = np.round(60/freq_hourly,0)
            freq = "{num_minutes}T".format(num_minutes = freq_in_minutes)
            periods = np.round(days_ahead*24*60 / freq_in_minutes,0)
            
        future = model.make_future_dataframe(periods = int(periods), freq = freq, include_history = False)
        future = pd.merge_asof(future, weather_data, on = "ds")
        forecast = model.predict(future)
        forecast = forecast[["ds", "yhat"]]
        forecast.columns = ["ds", "yhat_prophet"]
        forecast["ds"] = forecast["ds"].apply(str)
        return forecast
    
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

        return preds_test, model

    def SaveProphetModel(model, path_to_model):
        with open(path_to_model, 'w') as fout:
            fout.write(model_to_json(model))  # Save model
    
    def ProduceModel(url_minio, 
                     access_key, 
                     secret_key, 
                     pilot_name, 
                     measurement_name, 
                     asset_name,
                     path_to_model):
        
        # START PROCESS

        client = Minio(
            url_minio,
            access_key=access_key,
            secret_key=secret_key,
        )

        bucket_name = "{pilot_name}-{measurement}-{asset}".format(
            pilot_name = pilot_name.lower().replace("_", "-"),
            measurement = measurement_name.lower().replace("_", "-"),
            asset = asset_name.lower().replace("_","-")
        )

        print(bucket_name)

        if client.bucket_exists(bucket_name) != True:
            client.make_bucket(bucket_name)

        model_name = "prophet_model_{date}.json".format(date = datetime.strftime(maya.now().datetime(), "%Y_%m_%d"))

        # SAVE MODEL FILE(S)

        client.fput_object(bucket_name,
                        model_name,
                        file_path = path_to_model)
        
        # UPDATE MODEL CONFIGURATION

        try:
            client.fget_object(bucket_name, 
                               "asset_prophet_config.json",
                               "prophet_config.json")
            with open("prophet_config.json") as file:
                dict_config = json.load(file)
            
            list_models = dict_config["list_models"]
            dict_config = {
                "list_models": list_models.append(model_name),
                "set_model": model_name,
                "train_date": datetime.strftime(maya.now().datetime(), "%Y-%m-%d"),
                "lastUpdate": datetime.strftime(maya.now().datetime(), "%Y-%m-%d %H:%M:%S")
            }

        except:
            dict_config = {
                "list_models": [model_name],
                "set_model": model_name,
                "train_date": datetime.strftime(maya.now().datetime(), "%Y_%m_%d"),
                "lastUpdate": datetime.strftime(maya.now().datetime(), "%Y_%m_%d %H:%M:%S")
            }
        
        with open("prophet_config.json", "w") as file:
            json.dump(dict_config, file)
        
        client.fput_object(bucket_name, 
                               "asset_prophet_config.json",
                               "prophet_config.json")


        return model_name

    def DownloadModel(url_minio,
                      access_key,
                      secret_key,
                      pilot_name,
                      measurement_name,
                      asset_name):

        client = Minio(
            url_minio,
            access_key=access_key,
            secret_key=secret_key,
        )

        bucket_name = "{pilot_name}-{measurement}-{asset}".format(
            pilot_name = pilot_name.lower().replace("_", "-"),
            measurement = measurement_name.lower().replace("_", "-"),
            asset = asset_name.lower().replace("_","-")
        )


        client.fget_object(bucket_name,
                        "asset_prophet_config.json",
                        file_path = "prophet_config.json")

        with open("prophet_config.json") as file:
            config_ = json.load(file)
        
        model_name = config_["model_name"]

        client.fget_object(bucket_name,
                        model_name,
                        file_path = "prophet_model.json")

        return model_name

    # READ DATA COMING FROM PREVIOUS STEPS

    weather_data = pd.read_feather(input_weather_path)

    if load_bool == False:
        # THIS IS THE PATH OF TRAINING AND FORECASTING

        with open(input_data_path) as file:
            data_str = json.load(file)
        
        data = pd.DataFrame(data_str)
        
        freq_hourly = 60/int(diff_time)

        data = data[data.asset_name == asset_name]

        day_train_max = datetime.strftime(maya.parse(max(data.ds)).add(days = - 1 * num_days).datetime(), "%Y-%m-%d %H:%M:%S")

        data = data[data.ds < day_train_max]

        ic(day_train_max)
        ic(data.shape[0])
        
        try:
            pred_test, model = TrainAndPredictProphet(data, weather_data, freq_hourly, num_days)
            pred_test.reset_index().to_feather(forecast_data_path)
        except Exception as e:
            pred_test = pd.DataFrame()
            model_name = "No Model Trained"
            print("Training Failed")
            print(e)
            pred_test.to_csv(forecast_data_path)

        

        with fuckit:
            SaveProphetModel(model, "prophet_model.json")

            model_name = ProduceModel(
                url_minio, 
                access_key, 
                secret_key, 
                pilot_name, 
                measurement_name, 
                asset_name,
                "prophet_model.json"
            )    
            print("Model saved")

        results_dict = {
            "model_name": model_name
        }
    
    else:
        model_name = DownloadModel(
            url_minio,
            access_key,
            secret_key,
            pilot_name,
            measurement_name,
            asset_name
        )
        
        with open("prophet_model.json", 'r') as fin:
            prophet_model = model_from_json(fin.read())  # Load model
        
        latest_date = prophet_model.make_future_dataframe(periods = 1, freq = "1H", include_history= False)
        latest_date = latest_date["ds"].tolist()[0]

        num_days_extra = (maya.now() - maya.MayaDT(latest_date)).days

        num_days = num_days_extra + num_days

        forecast_ = PredictFromProphet(prophet_model, weather_data, freq_hourly = 60/ diff_time, days_ahead = num_days)

        forecast_.to_feather(forecast_data_path)

        print("Model {model_name} used for test".format(model_name = model_name))

        results_dict = {
            "model_name": model_name
        }

    with open(results_path, "w") as file:
        json.dump(results_dict, file)

