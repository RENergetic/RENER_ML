from kfp.components import InputPath, OutputPath

def ForecastLSTM(input_data_path: InputPath(str),
    input_weather_path: InputPath(str),
    diff_time,
    num_days,
    pilot_name,
    measurement_name,
    asset_name,
    url_minio,
    access_key,
    secret_key,
    n_epochs,
    load_bool:bool,
    forecast_data_path: OutputPath(str),
    results_path: OutputPath(str)

):
    
    # LIBRARIES REQUIRED    

    import numpy as np
    import json
    import pandas as pd
    import pickle

    # FUNCTIONS

    from tqdm import tqdm
    from darts.models import RNNModel
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.utils.missing_values import fill_missing_values
    from darts.metrics import mae, rmse
    from minio import Minio
    import maya
    from datetime import datetime
    import fuckit

    def Train_LSTM(train_data,
                weather_data,
                metric='MAE',
                params_grid={'hidden_dim': [25], 'n_rnn_layers': [1]},
                diff_time: int = 60,
                n_epochs: int = 4):

        batch_size = 16
        split_proportion = 0.9

        """
        
        Trains a lstm model based on train

        For now, no regularization and no hyperparameter training is performed
        First start using the regularization functionality of darts
        Then (maybe) vary hyperparameters

        Parameters
        ----------
        train_data: Data in the format ["ds", "y"], where ds is a columns in datetime or string = "%Y-%m-%d %H:%M:%S
        diff_time: time in minutes between two consecutive measurements; missing values will be interpolated
            
        Returns
        -------
        model: The model trained

        """
        
        # all time steps must exist in the time series; this is guaranteed by setting freq when creating the time series
        series = TimeSeries.from_dataframe(train_data, 'ds', 'y', fill_missing_dates=True, freq="{minutes}T".format(minutes = diff_time))

        # no missing values should be in the series
        # when the time series adds missing dates, these have missing values --> fill the missing values after creating the time series
        series = fill_missing_values(series) # this uses interpolation (instead of filling with the last available value)

        # Create training and validation sets:
        # TODO consistent approach for all mdoels?
        # TODO make sure diff_time is considered for calculating the hours, or use an alternative approach to split the series
        train, val = series.split_after(
            pd.Timestamp(series.start_time() + pd.Timedelta(hours=int(len(series) * split_proportion * (int(diff_time)/60))))
        )

        transformer = Scaler()
        train_transformed = transformer.fit_transform(train)
        val_transformed = transformer.transform(val)
        # series_transformed = transformer.fit_transform(series)

        # if weather data is not available, train the model without it
        # TODO check if weather data covers enough time to be used as future covariates in RNNModel
        try:
            # create a weather series with the same time steps as the train series
            weather_series = TimeSeries.from_dataframe(
                weather_data, time_col='ds', fill_missing_dates=True, freq="{minutes}T".format(minutes=diff_time)
            )
            weather_series = fill_missing_values(weather_series)  # fill missing values
            weather_transformer = Scaler()  # create a scaler to normalize the weather series
            weather_transformed = weather_transformer.fit_transform(weather_series)  # fit the scaler to the weather series and transform it
        except:
            weather_transformed = None
            weather_transformer = None
            
        measures_per_hour = int(60/int(diff_time))

        # Define metric function
        if metric == 'MAE':
            metric_fn = mae
        elif metric == 'RMSE':
            metric_fn = rmse
        else:
            raise ValueError("Invalid metric. Choose 'MAE' or 'RMSE'.")
        
        # Add fixed parameters to the grid search
        params_grid['input_chunk_length'] = [24 * measures_per_hour]
        params_grid['training_length'] = [2 * 24 * measures_per_hour]
        params_grid['hidden_dim'] = [25]
        params_grid['n_rnn_layers'] = [1]
        params_grid['dropout'] = [0.1]
        params_grid['n_epochs'] = [n_epochs]
        params_grid['batch_size'] = [batch_size]
        params_grid['optimizer_kwargs'] = [{"lr": 1e-3}]

        best_model, best_params, best_metric = RNNModel.gridsearch(
            parameters=params_grid,
            series=train_transformed,
            val_series=val_transformed,
            future_covariates=weather_transformed,
            metric=metric_fn,
            verbose=True
        )

        # Add other fixed parameters to the best_params dictionary
        best_params.update(model_name="data_RNN",
                        log_tensorboard=True,
                        random_state=42,
                        force_reset=True,
                        save_checkpoints=True)
        
        best_model = RNNModel(**best_params)
        # model = RNNModel(
        #     input_chunk_length=24*measures_per_hour, # TODO use one day as input; OK?
        #     model="LSTM",
        #     hidden_dim=25, 
        #     n_rnn_layers=1,
        #     # dropout=0.2, # this does not have an effect with only one rnn_layer
        #     training_length=2*24*measures_per_hour, # should be larger than input_chunk_length according to Darts docs
        #     batch_size=batch_size,
        #     n_epochs=n_epochs,
        #     optimizer_kwargs={"lr": 1e-3},
        #     model_name="data_RNN",
        #     log_tensorboard=True,
        #     random_state=42,
        #     force_reset=True,
        #     save_checkpoints=True
        # )

        best_model.fit(
            train_transformed,
            future_covariates=weather_transformed,
            val_series=val_transformed,
            val_future_covariates=weather_transformed,
            verbose=True
        )

        return best_model, best_metric, (transformer, weather_transformer)

    def PredictFromLSTM(model, data, weather_data,
                        transformer, weather_transformer,
                        diff_time,
                        days_ahead = 1):

        """
        Takes a LSTM model and makes the prediction.

        Parameters
        ----------
        data: Data in the format ["ds", "y"], where ds is a columns in datetime or string = "%Y-%m-%d %H:%M:%S. 
                this dataset is used as input for the model; must have length >= input_chunk_length specified in RNNModel; forecasts start following the end of this dataset
        diff_time: time in minutes between two consecutive measurements; missing values will be interpolated
            
        Returns
        -------
        forecast: The model trained

        """

        measures_per_hour = int(60/int(diff_time))

        input_series = TimeSeries.from_dataframe(data, 'ds', 'y',fill_missing_dates=True, freq="{minutes}T".format(minutes = diff_time))
        input_series = fill_missing_values(input_series)
        # transformer = Scaler()
        input_series_transformed = transformer.transform(input_series)

        # TODO check if weather data covers enough time to be used as future covariates in RNNModel
        try:
            weather_series = TimeSeries.from_dataframe(
                weather_data, time_col='ds', fill_missing_dates=True, freq="{minutes}T".format(minutes=diff_time)
            )
            weather_series = fill_missing_values(weather_series)

            # weather_transformer = Scaler()
            weather_transformed = weather_transformer.transform(weather_series)
        except:
            weather_transformed = None

        forecast = model.predict(n=days_ahead*24*measures_per_hour, series = input_series_transformed, future_covariates=weather_transformed)
        forecast = transformer.inverse_transform(forecast)
        forecast = forecast.pd_dataframe().reset_index()
        forecast = forecast[['ds', 'y']]
        forecast.columns = ['ds', 'yhat_lstm']
        forecast['ds'] = forecast['ds'].apply(str)

        return forecast

    def TrainAndPredictLSTM(data, weather_data,diff_time, days_ahead, n_epochs):
        """
        
        Component to train and predict

        """

        model, metric, (transformer, weather_transformer) = Train_LSTM(data, weather_data,diff_time=diff_time, n_epochs = n_epochs)
        preds_test = PredictFromLSTM(model, data, weather_data,
                                    transformer, weather_transformer,
                                    diff_time, days_ahead)

        return preds_test, model, (transformer, weather_transformer)
    
    def SaveLSTMScaleObjects(transformer, weather_transformer):
        """
        Save the transformer and weather_transformer objects in a current directory
        """

        with open("lstm_scaler.pkl", "wb") as file:
            pickle.dump(transformer, file)

        with open("lstm_weather_scaler.pkl", "wb") as file:
            pickle.dump(weather_transformer, file)


    def SaveLSTMModel(model, path_to_model):
        model.save(path_to_model)

    def ProduceLSTMModel(url_minio, 
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

        model_name = "lstm_model_{date}.pt".format(date = datetime.strftime(maya.now().datetime(), "%Y_%m_%d"))

        if client.bucket_exists(bucket_name) != True:
            client.make_bucket(bucket_name)

        # SAVE MODEL FILE(S)

        client.fput_object(bucket_name,
                        model_name,
                        file_path = path_to_model)
        
        client.fput_object(bucket_name,
                        model_name + ".ckpt",
                        file_path = path_to_model+ ".ckpt")
        
        client.fput_object(bucket_name,
                        model_name.replace(".pt", "_") + "scaler.pkl",
                        file_path = "lstm_scaler.pkl")

        client.fput_object(bucket_name,
                        model_name.replace(".pt", "_") + "weather_scaler.pkl",
                        file_path = "lstm_weather_scaler.pkl")
        
        # UPDATE MODEL CONFIGURATION

        try:
            client.fget_object(bucket_name, 
                               "asset_lstm_config.json",
                               "lstm_config.json")
            with open("lstm_config.json") as file:
                dict_config = json.load(file)
            
            list_models = dict_config["list_models"]
            dict_config = {
                "list_models": list_models.append(model_name),
                "set_model": model_name,
                "train_date": datetime.strftime(maya.now().datetime(), "%Y_%m_%d"),
                "lastUpdate": datetime.strftime(maya.now().datetime(), "%Y_%m_%d %H:%M:%S")
            }

        except:
            dict_config = {
                "list_models": [model_name],
                "set_model": model_name,
                "train_date": datetime.strftime(maya.now().datetime(), "%Y_%m_%d"),
                "lastUpdate": datetime.strftime(maya.now().datetime(), "%Y_%m_%d %H:%M:%S")
            }
        
        with open("lstm_config.json", "w") as file:
            json.dump(dict_config, file)
        
        client.fput_object(bucket_name, 
                               "asset_lstm_config.json",
                               "lstm_config.json")
        
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
                        "asset_lstm_config.json",
                        file_path = "lstm_config.json")

        with open("lstm_config.json") as file:
            config_ = json.load(file)
        
        model_name = config_["model_name"]

        client.fget_object(bucket_name,
                        model_name,
                        file_path = "lstm_model.pt")
        client.fget_object(bucket_name,
                        model_name + ".ckpt",
                        file_path = "lstm_model.pt.ckpt")
        
        client.fget_object(bucket_name,
                        model_name.replace(".pt", "_") + "scaler.pkl",
                        file_path = "lstm_scaler.pkl")
        
        client.fget_object(bucket_name,
                        model_name.replace(".pt", "_") + "weather_scaler.pkl",
                        file_path = "lstm_weather_scaler.pkl")

        return model_name

    # READ DATA COMING FROM PREVIOUS STEPS

    with open(input_data_path) as file:
        data_str = json.load(file)
    
    data = pd.DataFrame(data_str)
    data = data[data.asset_name == asset_name]

    weather_data = pd.read_feather(input_weather_path)

    if load_bool == False:
        try:
            pred_test, model, (transformer, weather_transformer) = TrainAndPredictLSTM(data, weather_data, diff_time, num_days, n_epochs)
            pred_test.reset_index().to_feather(forecast_data_path)
        except Exception as e:
            pred_test = pd.DataFrame()
            model_name = "No model trained"
            print("Train model failed")
            print(e)
            pred_test.to_csv(forecast_data_path)
        
        pred_test.reset_index().to_feather(forecast_data_path)
        
        with fuckit:
            SaveLSTMModel(model, "lstm_model.pt")
            SaveLSTMScaleObjects(transformer, weather_transformer)

            model_name = ProduceLSTMModel(
                url_minio,
                access_key, 
                secret_key, 
                pilot_name, 
                measurement_name, 
                asset_name,
                "lstm_model.pt"
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
        lstm_model = RNNModel.load("lstm_model.pt")  # Load model
        data["ds"] = pd.to_datetime(data["ds"])
        data = data.set_index("ds").asfreq("{minutes}T".format(minutes=diff_time)).reset_index()

        # load scale objects
        with open("lstm_scaler.pkl", "rb") as file:
            transformer = pickle.load(file)

        with open("lstm_weather_scaler.pkl", "rb") as file:
            weather_transformer = pickle.load(file)

        forecast_ = PredictFromLSTM(lstm_model, data, weather_data, 
                                    transformer, weather_transformer,
                                    diff_time, days_ahead=num_days)

        forecast_.reset_index().to_feather(forecast_data_path)

        print("Model {model_name}".format(model_name = model_name))

        results_dict = {
            "model_name": model_name
        }



    with open(results_path, "w") as file:
        json.dump(results_dict, file)

