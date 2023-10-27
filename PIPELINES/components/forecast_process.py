from kfp.components import InputPath, OutputPath

def ForecastProcess(input_data_path: InputPath(str), input_weather_path: InputPath(str),
    measurement_name,
    path_minio,
    access_key,
    secret_key,
    mode,
    url_pilot,
    diff_time,
    pilot_name,
    send_forecast,
    asset_name,
    num_days,
    mode_prophet,
    daily_seasonality,
    weekly_seasonality,
    mlpipeline_metrics_path: OutputPath('Metrics'),
    forecast_data_path: OutputPath(str)
    ):

    import maya
    import json
    from icecream import ic
    import requests
    import pandas as pd
    from prophet.serialize import model_to_json
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from minio import Minio
    import boto3
    from tqdm import tqdm
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from datetime import datetime

    from darts.models import RNNModel
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.utils.timeseries_generation import datetime_attribute_timeseries



    try:
        client = Minio(
            path_minio,
            access_key=access_key,
            secret_key=secret_key,
            secure = False
        )

        list_objects = client.list_objects("test")
        for obj_ in list_objects:
            ic(obj_._object_name)
    except:
        ic("Cannot access minio server correctly - read data.")
    

    def ForecastData(data, asset_name, measurement_name, metrics_list, measures_per_hour, diff_time, weather_data ,mode_prophet,daily_seasonality, weekly_seasonality,mode = "no notifications"):
        
        # Generic Processing Functions

        def ModifyData(data, asset_name):
            data_ds = data[data.asset_name == asset_name][["time_registered", "value"]]
            try:
                last_cummulative_value = data[data.asset_name == asset_name]["value_cummulative"].tolist()[-1]
            except:
                last_cummulative_value = 0
            data_ds.columns = ["ds", "y"]
            if data_ds.shape[0] == 0:
                max_date = datetime.strftime(maya.when("now").datetime(),"%Y-%m-%d %H:%M:%S")
            else:
                max_date = max(data_ds.ds)
            ic(max_date)
            ic(last_cummulative_value)
            ic(data_ds.shape)
            ic(len(pd.unique(data_ds.y)))

            return data_ds, max_date, last_cummulative_value
        # Categorical Processing Functions

        def GetDateInfo(ds_str, time_value):
            maya_obj = maya.parse(ds_str)
            if time_value == "year":
                return str(maya_obj.year)
            elif time_value == "month":
                return str(maya_obj.month)
            elif time_value == "weekday":
                return str(maya_obj.weekday)
            elif time_value == "hour":
                return str(maya_obj.hour)
            
        def ManageData(data_ds, num_prevs = 24):
            data_ds["year"]  = data_ds["ds"].apply(GetDateInfo, time_value = "year")
            data_ds["month"]  = data_ds["ds"].apply(GetDateInfo, time_value = "month")
            data_ds["weekday"]  = data_ds["ds"].apply(GetDateInfo, time_value = "weekday")
            data_ds["hour"]  = data_ds["ds"].apply(GetDateInfo, time_value = "hour")

            for var_ in ["year", "month", "weekday", "hour"]:
                if len(pd.unique(data_ds[var_])) <= 2:
                    bin_main = list(pd.unique(data_ds[var_]))[0]
                    data_ds[var_] = (data_ds[var_] == bin_main)
            
            for i in range(1,num_prevs + 1):
                name_var = "prev_val_{i}".format(i = i)
                data_ds[name_var] = data_ds["y"].shift(i).apply(str)
            data_ds = data_ds[(num_prevs+1):]

            cat_features_names = ["year", "month", "weekday", "hour"]
            names_prevs_vars = []
            for name_var in data_ds.columns.values:
                if "prev_val" in name_var:
                    names_prevs_vars.append(name_var)
            cat_features_names = cat_features_names + names_prevs_vars


            return data_ds, names_prevs_vars, cat_features_names
        
        def Train_CatBoost(data_ds, cat_features_names):

                # Process data
                X_train, X_test, Y_train, Y_test = train_test_split(data_ds.drop(["y", "ds"], axis = 1), data_ds.y, test_size = 0.2)
                

                # Pool Creation

                train_pool = Pool(
                    data = X_train, label = Y_train, 
                    cat_features = cat_features_names
                    )
                test_pool = Pool(
                    data = X_test, label = Y_test, 
                    cat_features = cat_features_names
                    )
                
                catboost_model = CatBoostClassifier(
                    iterations = 10,
                    learning_rate = 1,
                    depth = 8
                )
                print("Model To Train")
                catboost_model.fit(train_pool)
                print("Model trained")
                yhat_test = catboost_model.predict(test_pool)
                yhat_train = catboost_model.predict(train_pool)
                print("Y hat obtained")
                accuracy_score_train = accuracy_score(Y_train, yhat_train)
                accuracy_score_test = accuracy_score(Y_test, yhat_test)
                ic(accuracy_score_train)
                ic(accuracy_score_test)
            
                return catboost_model, accuracy_score_train, accuracy_score_test

        def Save_CatBoost(catboost_model, pilot_name, measurement_name, asset_name):
            catboost_model.save_model("/tmp/catboost_model.cbm", format = "cbm")
            s3 = boto3.resource(
                service_name='s3',
                aws_access_key_id='QyvycO9kc2cm58K8',
                aws_secret_access_key='tKtUrdQzQgWfhfBwhbQF3yGbyZ43oPn92iGAT7g0',
                endpoint_url='https://s3.tebi.io'
            )

            for bucket in s3.buckets.all():
                ic(bucket.name)
            
            # Upload a new file
            data = open('/tmp/catboost_model.cbm', 'rb')
            f_name = "model_{pilot}_{domain}_{asset}_latest_catboost.cbm"\
            .format(pilot = pilot_name,domain = measurement_name, asset = asset_name)
            s3.Bucket('test-pf').put_object(Key=f_name, Body=data)

            message = "Model sent to tebi for {measurement_name} - {asset_name}".format(measurement_name = measurement_name, asset_name = asset_name)
            ic(message)

        def Predict_CatBoost(catboost_model, data_ds, last_cummulative_value, num_days, measures_per_hour, diff_time, cat_features_names, names_prevs_vars):
            data_ds["yhat"] = last_cummulative_value
            for day_ in range(num_days):
                for i in tqdm(range(24)):
                    for j in range(measures_per_hour):
                        last_row = data_ds.iloc[-1]
                        ds_obj = maya.parse(last_row["ds"]).add(minutes = int(diff_time))
                        dict_input = {
                            "year": ds_obj.year,
                            "month": ds_obj.month,
                            "weekday": ds_obj.weekday,
                            "hour": ds_obj.hour
                        }
                        for k_var in names_prevs_vars:
                            k = int(k_var[9:])
                            if k == 1:
                                dict_input[k_var] = str(last_row["y"])
                            else:
                                dict_input[k_var] = str(last_row["prev_val_{i}".format(i = k -1)])
                        dict_input = [dict_input]
                        data_input = pd.DataFrame(dict_input)
                        pred_pool = Pool(
                            data = data_input, 
                            cat_features = cat_features_names
                            )
                        pred_value = catboost_model.predict(pred_pool)[0][0]
                        dict_input[0]["y"] = pred_value
                        dict_input[0]["ds"] = datetime.strftime(ds_obj.datetime(), "%Y-%m-%d %H:%M:%S")
                        dict_input[0]["yhat"] = last_row["yhat"] + float(pred_value)
                        data_add = pd.DataFrame(dict_input)
                        data_ds = pd.concat([data_ds, data_add],ignore_index = True)
            return data_ds

        def Metrics_CatBoost(accuracy_score_train, accuracy_score_test, metrics_list):
            metrics = {
                'metrics': [
                    {
                    'name': 'accuracy_train',
                    'numberValue':  float(accuracy_score_train),
                    'format': "PERCENTAGE"
                    },
                    {
                        'name': 'accuracy_test',
                        "numberValue": float(accuracy_score_test),
                        "format": "PERCENTAGE"
                    },
                    {
                        "name": "asset_number",
                        "numberValue": asset_name,
                        "format": "RAW"
                    }
                ]}  
            
            metrics_list.append(metrics)
            return metrics_list

        # Prophet Functions
        
        def ManageDateTime(ds_obj):
            return datetime.strftime(maya.parse(ds_obj).datetime(), "%Y-%m-%d %H:%M:%S")

        def ManageDateMinute(ds_obj):
            return datetime.strftime(maya.parse(ds_obj).datetime(), "%Y-%m-%d %H:%M:00")
        def ManageDateHour(ds_obj):
            return datetime.strftime(maya.parse(ds_obj).datetime(), "%Y-%m-%d %H:00:00")

        def Train_Prophet(train_data, num_days, measures_per_hour, diff_time, weather_data, mode_prophet, daily_seasonality, weekly_seasonality):
            from prophet import Prophet

            train_data["ds"] = train_data["ds"].apply(ManageDateMinute)
            train_data = train_data[["ds", "y"]].groupby("ds").mean().reset_index(level = "ds")
            train_data["ds_hour"] = train_data["ds"].apply(ManageDateHour)
            weather_data["ds_hour"] = weather_data["ds_hour"].apply(str)

            ic(train_data["ds_hour"].tolist()[-20:])
            ic(weather_data["ds_hour"].tolist()[-20:])
            train_data = pd.merge(train_data, weather_data, on = "ds_hour")

            ic(train_data.shape[0])

            min_date = maya.parse(min(train_data.ds))
            max_date = maya.parse(max(train_data.ds))
            days_train = (max_date - min_date).days

            if days_train >= 365:
                yearly_seasonality = True
            else:
                yearly_seasonality = False
            # Define Model to be trained
            m = Prophet(daily_seasonality=daily_seasonality, weekly_seasonality=weekly_seasonality, yearly_seasonality = yearly_seasonality,changepoint_prior_scale = 0.05, seasonality_mode=mode_prophet)

            m.add_regressor('shortwave_radiation')
            m.add_regressor('temperature_2m')
            m.add_regressor("direct_radiation")
            m.add_regressor("diffuse_radiation")
            m.add_regressor("direct_normal_irradiance")

            # Train Model
            m.fit(train_data)
            future = m.make_future_dataframe(periods= 24*(3 + num_days)*measures_per_hour , freq="{minutes}T".format(minutes = diff_time))
            future["ds"] = future["ds"].apply(str)
            future["ds_hour"] = future["ds"].apply(ManageDateHour)
            future = pd.merge(future, weather_data, on = "ds_hour")
            
            forecast = m.predict(future)

            print(forecast.tail(5))

            return forecast, m

        def GetMetricsProphet(forecast,train_data, test_data, num_days, dict_asset, metrics_list, date_train):
            try:
                asset_number = dict_asset[asset_name]
            except:
                asset_number = 3
            
            forecast["ds"] = forecast["ds"].apply(str)

            try:
                forecast_test = forecast[forecast.ds >= date_train]["yhat"].tolist()
                train_data = pd.merge(train_data, forecast[["ds", "yhat"]], on = "ds")
                real_vals_train = train_data["y"].tolist()
                forecast_train = train_data["yhat"].tolist()
                r2_score_train = r2_score(real_vals_train, forecast_train)
                mae_score_train = mean_absolute_error(real_vals_train, forecast_train)
            except:
                r2_score_train = 0
                mae_score_train = 0
            
            real_vals_test = test_data["y"].tolist()

            if len(forecast_test) == len(real_vals_test):
                r2_score_test = r2_score(real_vals_test, forecast_test)
                mae_score_test = mean_absolute_error(real_vals_test, forecast_test)
                metrics = {
                    'metrics': [
                        {
                        'name': 'r2_score_test',
                        'numberValue':  float(r2_score_test),
                        'format': "PERCENTAGE"
                        },
                        {
                            'name': 'r2_score_train',
                            "numberValue": float(r2_score_train),
                            "format": "PERCENTAGE"
                        },
                        {
                            "name": "asset_number",
                            "numberValue": asset_name,
                            "format": "RAW"
                        },
                        {
                            "name": "mae_train",
                            "numberValue": mae_score_train,
                            "format": "RAW"
                        },
                        {
                            "name": "mae_test",
                            "numberValue": mae_score_test,
                            "format": "RAW"
                        }
                    ]}  
                
                metrics_list.append(metrics)
                
            else:
                ic(len(forecast_test))
                ic(len(real_vals_test))
            
            return metrics_list

        def SaveModelProphet(model, measurement_name, asset_name, pilot_name):
            with open("/tmp/model_prophet.json", 'w') as fout:
                fout.write(model_to_json(model))  # Save model
            

            date = maya.when("now").rfc2822()
            f_name = "model_{domain}_{asset}.json"\
                .format(domain = measurement_name, asset = asset_name)
            try:
                result = client.fput_object(
                    "test", f_name, "/tmp/model_prophet.json"
                )

                print(
                    "created {0} object; etag: {1}, version-id: {2}".format(
                        result.object_name, result.etag, result.version_id,
                    ),
                )
            except:
                message = "Model not saved for {measurement_name} - {asset_name}".format(measurement_name = measurement_name, asset_name = asset_name)
                ic(message)

                s3 = boto3.resource(
                    service_name='s3',
                    aws_access_key_id='QyvycO9kc2cm58K8',
                    aws_secret_access_key='tKtUrdQzQgWfhfBwhbQF3yGbyZ43oPn92iGAT7g0',
                    endpoint_url='https://s3.tebi.io'
                )

                for bucket in s3.buckets.all():
                    ic(bucket.name)
                
                # Upload a new file
                data = open('/tmp/model_prophet.json', 'rb')
                f_name = "model_{pilot}_{domain}_{asset}_latest_prophet.json"\
                .format(pilot = pilot_name,domain = measurement_name, asset = asset_name)
                s3.Bucket('test-pf').put_object(Key=f_name, Body=data)

                message = "Model sent to tebi for {measurement_name} - {asset_name}".format(measurement_name = measurement_name, asset_name = asset_name)
                ic(message)

        # LSTM Functions
        def Train_LSTM(data, split_proportion, diff_time, num_days=1, 
                    measures_per_hour=1, n_epochs=100, batch_size=16):
            
            

            if not isinstance(data, pd.DataFrame) or 'ds' not in data.columns or 'y' not in data.columns:
                raise ValueError("The input data must be a pandas DataFrame with 'ds' and 'y' columns.")

            if not (0 < split_proportion < 1):
                raise ValueError("The split_proportion must be a float between 0 and 1.")
            
            # datetime and freq must be set for the time series to be usable by darts
            data["ds"] = pd.to_datetime(data["ds"])
            data = data.set_index("ds").asfreq("{minutes}T".format(minutes=diff_time)).reset_index()

            # fill missing values with the last available value
            data = data.fillna(method='ffill')
            
            # Create a time series
            series = TimeSeries.from_dataframe(data, 'ds', 'y',fill_missing_dates=True, freq="{minutes}T".format(minutes = diff_time))

            # Create training and validation sets:
            train, val = series.split_after(pd.Timestamp(series.start_time() + pd.Timedelta(hours=int(len(series) * split_proportion))))

            # Normalize the time series (note: we avoid fitting the transformer on the validation set)
            transformer = Scaler()
            train_transformed = transformer.fit_transform(train)
            val_transformed = transformer.transform(val)
            series_transformed = transformer.transform(series)

            # predict *num_days* days ahead
            pred_ahead = 24 * (2 + num_days) * measures_per_hour

            my_model = RNNModel(
                input_chunk_length=2 * pred_ahead,
                model="LSTM",
                hidden_dim=25, 
                n_rnn_layers=1,
                dropout=0.2,
                training_length=pred_ahead,
                batch_size=batch_size,
                n_epochs=n_epochs,
                optimizer_kwargs={"lr": 1e-3},
                model_name="data_RNN",
                log_tensorboard=True,
                random_state=42,
                force_reset=True,
                save_checkpoints=True,
            )

            my_model.fit(
                train_transformed,
                val_series=val_transformed,
                verbose=True,
            )
            
            historical_forecast = my_model.historical_forecasts(
                                            series_transformed,
                                            start=pd.Timestamp(val.start_time() - pd.Timedelta(hours=1)),
                                            forecast_horizon=pred_ahead,
                                            retrain=False,
                                            verbose=True,
                                        )
            
            historical_forecast = transformer.inverse_transform(historical_forecast)
            historical_forecast = historical_forecast.pd_dataframe().reset_index()
            # rename columns
            historical_forecast.columns = ['ds', 'y']
            historical_forecast.columns.name = None

            # Predict
            forecast = my_model.predict(n=pred_ahead, series=val_transformed)

            # Inverse-transform forecasts and obtain the real predicted values
            forecast = transformer.inverse_transform(forecast)
            forecast = forecast.pd_dataframe().reset_index()
            forecast.columns.name = None

            forecast = pd.concat([historical_forecast, forecast], axis=0).reset_index(drop=True)

            # Check the dataframe if the frequency is always {diff_time} minute
            full_range = pd.date_range(forecast['ds'].iloc[0], forecast['ds'].iloc[-1], freq="{minutes}T".format(minutes = diff_time))
            assert full_range.difference(forecast['ds']).shape[0] == 0

            forecast_test = forecast["y"].tolist()[-24*measures_per_hour*(2 + num_days):-24*measures_per_hour]

            return forecast, forecast_test, my_model
        
        def GetMetricsLSTM(forecast, forecast_test, train_data, test_data, num_days, dict_assets, metrics_list):
            try:
                asset_number = dict_assets[asset_name]
            except:
                asset_number = 3
            try:
                # take the comman ds for forecast and train_data
                forecast_train = forecast[forecast['ds'].isin(train_data['ds'])].reset_index(drop=True)
                real_vals_train = train_data[train_data['ds'].isin(forecast['ds'])].reset_index(drop=True)
                r2_score_train = r2_score(real_vals_train['y'].to_list(), forecast_train['y'].to_list())
                mae_score_train = mean_absolute_error(real_vals_train['y'].to_list(), forecast_train['y'].to_list())
            except:
                r2_score_train = 0
            
            real_vals_test = test_data["y"].tolist()

            if len(forecast_test) == len(real_vals_test):
                r2_score_test = r2_score(real_vals_test, forecast_test)
                mae_score_test = mean_absolute_error(real_vals_test, forecast_test)
                metrics = {
                    'metrics': [
                        {
                        'name': 'r2_score_test',
                        'numberValue':  float(r2_score_test),
                        'format': "PERCENTAGE"
                        },
                        {
                            'name': 'r2_score_train',
                            "numberValue": float(r2_score_train),
                            "format": "PERCENTAGE"
                        },
                        {
                            "name": "asset_number",
                            "numberValue": asset_name,
                            "format": "RAW"
                        },
                         {
                            "name": "mae_train",
                            "numberValue": mae_score_train,
                            "format": "RAW"
                        },
                        {
                            "name": "mae_test",
                            "numberValue": mae_score_test,
                            "format": "RAW"
                        }
                    ]}  
                
                metrics_list.append(metrics)
                
            else:
                ic(len(forecast_test))
                ic(len(real_vals_test))
            return metrics_list

        def SaveModelLSTM(model, measurement_name, asset_name, pilot_name):
            model.save("/tmp/lstm_model.pt")
            # model_loaded = RNNModel.load("/tmp/lstm_model.pt")
            
            date = maya.when("now").rfc2822()
            f_name = "model_{domain}_{asset}.pt"\
                .format(domain = measurement_name, asset = asset_name)
            try:
                result = client.fput_object(
                    "test", f_name, "/tmp/lstm_model.pt"
                )

                print(
                    "created {0} object; etag: {1}, version-id: {2}".format(
                        result.object_name, result.etag, result.version_id,
                    ),
                )
            except:
                message = "Model not saved for {measurement_name} - {asset_name}".format(measurement_name = measurement_name, asset_name = asset_name)
                ic(message)

                s3 = boto3.resource(
                    service_name='s3',
                    aws_access_key_id='QyvycO9kc2cm58K8',
                    aws_secret_access_key='tKtUrdQzQgWfhfBwhbQF3yGbyZ43oPn92iGAT7g0',
                    endpoint_url='https://s3.tebi.io'
                )

                for bucket in s3.buckets.all():
                    ic(bucket.name)
                
                # Upload a new file
                data = open('/tmp/lstm_model.pt', 'rb')
                f_name = "model_{pilot}_{domain}_{asset}_latest_lstm.pt"\
                .format(pilot = pilot_name,domain = measurement_name, asset = asset_name)
                s3.Bucket('test-pf').put_object(Key=f_name, Body=data)

                message = "Model sent to tebi for {measurement_name} - {asset_name}".format(measurement_name = measurement_name, asset_name = asset_name)
                ic(message)


        ############

        print("Modifying Data")
        print(maya.now().rfc2822())

        data_ds, max_date, last_cummulative_value = ModifyData(data, asset_name)

        print("Start Training")
        
        if len(pd.unique(data_ds.y)) >= 20:

            date_train = datetime.strftime(maya.parse(max_date).add(days = -1).datetime(), "%Y-%m-%d")
            train_data = data_ds[data_ds.ds < date_train]
            test_data = data_ds[data_ds.ds >= date_train]

            # Train Prophet
            print("Training Prophet")
            print(maya.now().rfc2822())
            forecast_prophet, model_prophet = Train_Prophet(train_data, num_days, measures_per_hour, diff_time, weather_data, mode_prophet, daily_seasonality, weekly_seasonality)
            metrics_list_prophet = GetMetricsProphet(forecast_prophet,train_data, test_data, num_days, dict_assets, metrics_list, date_train)
            
            # Train LSTM
            print("Training LSTM")
            print(maya.now().rfc2822())

            try:
                ic(diff_time)
                forecast_lstm, forecast_test, model_lstm = Train_LSTM(data=train_data, split_proportion=0.9, diff_time=diff_time,
                                                            num_days=num_days, measures_per_hour=measures_per_hour, 
                                                            n_epochs=25)
                metrics_list_lstm = GetMetricsLSTM(forecast_lstm, forecast_test, train_data, test_data, num_days, dict_assets, metrics_list)
            except ValueError:
                metrics_list_lstm = [{
                    "name": "mae_test",
                    "numberValue": -1
                }]
            print("Finish Training")
            print(maya.now().rfc2822())
            ############

            # Compare Models

            forecast = forecast_prophet
            metrics_list = metrics_list_prophet
            
            print("Metrics Prophet")

            for metric in metrics_list:
                try:
                    print(metric["name"])
                    print(metric["numberValue"])
                except:
                    print(metrics_list)
                    break

            print("Metrics LSTM")

            for metric in metrics_list_lstm:
                try:
                    print(metric["name"])
                    print(metric["numberValue"])
                except:
                    print(metrics_list)
                    break

            ###########
            
            SaveModelProphet(model_prophet, measurement_name, asset_name, pilot_name)

            try:
                SaveModelLSTM(model_lstm, measurement_name, asset_name, pilot_name)
            except:
                print("LSTM Model Not Saved")

        elif data_ds.shape[0] < 10:
            print("Not enough values")
            forecast = data_ds

        else:
            data_ds, names_prevs_vars, cat_features_names = ManageData(data_ds)
            
            catboost_model, accuracy_score_train, accuracy_score_test = Train_CatBoost(data_ds, cat_features_names)

            forecast = Predict_CatBoost(catboost_model,
                            data_ds, 
                            last_cummulative_value,
                            num_days, measures_per_hour, diff_time,
                            cat_features_names, names_prevs_vars)

            Save_CatBoost(catboost_model, pilot_name, measurement_name, asset_name)
            metrics_list = Metrics_CatBoost(accuracy_score_train, accuracy_score_test,
                                            metrics_list)

        ic(metrics_list)
        return forecast[forecast.ds > max_date], metrics_list

    
    # Get Parameters
    
    dict_assets = {}
    measures_per_hour = int(60/int(diff_time))
    time_prediction = maya.now().epoch
    num_days = int(num_days)


    with open(input_data_path) as file:
        data_str = json.load(file)
    
    data = pd.DataFrame(data_str)
    metrics_list = []
    ic(asset_name)

    weather_data = pd.read_feather(input_weather_path)
    
    class TestError(Exception):
        "Input correctly City name or Lat AND Lon for city"
        pass

    try:
        ic(diff_time)
        forecasted_data, metrics_list = ForecastData(data, asset_name, measurement_name, metrics_list, measures_per_hour, diff_time, weather_data, mode_prophet, daily_seasonality, weekly_seasonality)
    except TestError:
        forecasted_dict = {
            "ds": [],
            "yhat": []
        }
        forecasted_data = pd.DataFrame(forecasted_dict)
        metrics_list = []
    
    
    try:
        forecasted_data.to_csv('/tmp/forecast_test_{asset_name}.csv'.format(asset_name = asset_name), index = False)
        data_to_send = open('/tmp/forecast_test_{asset_name}.csv'.format(asset_name = asset_name), 'rb')
        f_name = "forecast_test_{pilot}_{asset_name}.csv".format(pilot = pilot_name, asset_name = asset_name)
        s3 = boto3.resource(
            service_name='s3',
            aws_access_key_id='QyvycO9kc2cm58K8',
            aws_secret_access_key='tKtUrdQzQgWfhfBwhbQF3yGbyZ43oPn92iGAT7g0',
            endpoint_url='https://s3.tebi.io'
        )
        s3.Bucket('test-pf').put_object(Key=f_name, Body=data_to_send)
        message = "Data File: {f_name} Saved to Tebi".format(f_name = f_name)
        ic(message)
    except:
        message = "Unable to save data to tebi"
        ic(message)

        message = "Values for {asset_name}: {list_values}".format(asset_name = asset_name,list_values = forecasted_data.yhat.tolist())
        ic(message)

    forecasted_data["yhat"] = forecasted_data["yhat"].apply(lambda x : max(x,0))

    forecasted_data.to_csv(forecast_data_path, index = False)


    domain_ = "electricity"

    with open("/tmp/metrics_{domain}.json".format(domain = domain_), "w") as file:
        json.dump(metrics_list, file)
    
    

    s3 = boto3.resource(
                service_name='s3',
                aws_access_key_id='QyvycO9kc2cm58K8',
                aws_secret_access_key='tKtUrdQzQgWfhfBwhbQF3yGbyZ43oPn92iGAT7g0',
                endpoint_url='https://s3.tebi.io'
            )
    data = open("/tmp/metrics_{domain}.json".format(domain = domain_), 'rb')
    f_name = "metrics_{domain}_latest.json"\
    .format(domain = measurement_name)
    s3.Bucket('test-pf').put_object(Key=f_name, Body=data)
    


    message = "Forecasting done for {domain} and asset name : {asset_name}".format(domain = domain_, asset_name = asset_name)
    ic(message)
