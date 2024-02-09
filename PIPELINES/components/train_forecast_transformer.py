from kfp.components import InputPath, OutputPath

def ForecastTransformer(input_data_path: InputPath(str),
    input_weather_path: InputPath(str),
    diff_time,
    num_days,
    pilot_name,
    measurement_name,
    asset_name,
    n_epochs,
    url_minio,
    access_key,
    secret_key,
    forecast_data_path: OutputPath(str),
    results_path: OutputPath(str)
):
    # LIBRARIES REQUIRED
    import numpy as np
    import json
    import pandas as pd

    # FUNCTIONS
    from tqdm import tqdm
    from minio import Minio
    import maya
    from datetime import datetime
    from darts.models import TransformerModel
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.utils.missing_values import fill_missing_values
    from darts.metrics import mae, rmse

    def Train_Transformer(train_data,
                          weather_data,
                          metric="MAE",
                          params_grid={'d_model': [64, 128],
                                       'nhead': [4, 8]},
                          diff_time: int = 60,
                          n_epochs:int = 4):
        """
        Trains a Transformer model based on the given train_data.

        Parameters
        ----------
        train_data : pandas.DataFrame
            Data in the format ["ds", "y"], where "ds" is a column in datetime or string format ("%Y-%m-%d %H:%M:%S").
        weather_data : pandas.DataFrame
            Data containing weather information.
        metric : str, optional
            The metric used for model evaluation. Default is 'MAE'.
        params_grid : dict, optional
            The grid of hyperparameters to search over during grid search. Default is {'d_model': [64, 128], 'nhead': [4, 8]}.
        diff_time : int, optional
            Time in minutes between two consecutive measurements; missing values will be interpolated. Default is 60.

        Returns
        -------
        best_model : TransformerModel
            The trained Transformer model.
        best_metric : float
            The best metric value achieved during training.
        """

        batch_size = 16
        split_proportion = 0.9

        # all time steps must exist in the time series; this is guaranteed by setting freq when creating the time series
        series = TimeSeries.from_dataframe(
            train_data, 'ds', 'y', fill_missing_dates=True, freq="{minutes}T".format(minutes=diff_time)
        )

        # no missing values should be in the series
        series = fill_missing_values(series)

        # Create training and validation sets:
        train, val = series.split_after(
            pd.Timestamp(series.start_time() + pd.Timedelta(hours=int(len(series) * split_proportion * (int(diff_time)/60))))
        )

        # Normalize the time series (note: we avoid fitting the transformer on the validation set)
        transformer = Scaler()
        train_transformed = transformer.fit_transform(train)
        val_transformed = transformer.transform(val)
        # series_transformed = transformer.transform(series)

        # if weather data is not available, train the model without it
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

        measures_per_hour = int(60 / int(diff_time))

        # Define metric function
        if metric == 'MAE':
            metric_fn = mae
        elif metric == 'RMSE':
            metric_fn = rmse
        else:
            raise ValueError("Invalid metric. Choose 'MAE' or 'RMSE'.")

        # Add fixed parameters to the grid search
        params_grid['input_chunk_length'] = [2 * 24 * measures_per_hour]
        params_grid['output_chunk_length'] = [24 * measures_per_hour]
        params_grid['num_encoder_layers'] = [3]
        params_grid['num_decoder_layers'] = [3]
        params_grid['dim_feedforward'] = [512]
        params_grid['dropout'] = [0.1]
        params_grid['activation'] = ["relu"]
        params_grid['n_epochs'] = [n_epochs]
        params_grid['batch_size'] = [batch_size]
        params_grid['optimizer_kwargs'] = [{"lr": 1e-3}]

        best_model, best_params, best_metric = TransformerModel.gridsearch(
            parameters=params_grid,
            series=train_transformed,
            val_series=val_transformed,
            past_covariates=weather_transformed,
            metric=metric_fn,
            verbose=True
        )

        # Add other fixed parameters to the best_params dictionary
        best_params.update(model_name="data_transformer",
                        log_tensorboard=True,
                        random_state=42,
                        force_reset=True,
                        save_checkpoints=True)

        # Create the best model
        best_model = TransformerModel(**best_params)

        # Fit the best model
        best_model.fit(
            train_transformed,
            past_covariates=weather_transformed,
            val_series=val_transformed,
            val_past_covariates=weather_transformed,
            verbose=True
        )

        return best_model, best_metric

    def PredictFromTransformer(model, data, weather_data, diff_time, days_ahead=1):
        """
        Takes a Transformer model and makes the prediction.

        Parameters
        ----------
        model : TransformerModel
            The trained Transformer model.
        data : pandas.DataFrame
            The input data in the format ["ds", "y"], where "ds" is a column of datetime or string in the format "%Y-%m-%d %H:%M:%S".
            This dataset is used as input for the model and must have a length greater than or equal to the input_chunk_length specified in the TransformerModel.
            Forecasts start following the end of this dataset.
        weather_data : pandas.DataFrame
            The weather data used as covariates for the model. It should have a column named "ds" representing the time and other columns representing weather features.
        diff_time : int
            The time in minutes between two consecutive measurements. Missing values will be interpolated.
        days_ahead : int, optional
            The number of days to forecast. Default is 1.

        Returns
        -------
        forecast : pandas.DataFrame
            Dataframe of the forecasted values with columns "ds" and "yhat_transformer"
            ds: string in format "%Y-%m-%d %H:%M:%S", yhat_transformer: float
        """

        measures_per_hour = int(60 / int(diff_time))

        input_series = TimeSeries.from_dataframe(
            data, 'ds', 'y', fill_missing_dates=True, freq="{minutes}T".format(minutes=diff_time)
        )
        input_series = fill_missing_values(input_series)

        transformer = Scaler()
        input_series_transformed = transformer.fit_transform(input_series)

        try:
            weather_series = TimeSeries.from_dataframe(
                weather_data, time_col='ds', fill_missing_dates=True, freq="{minutes}T".format(minutes=diff_time)
            )
            weather_series = fill_missing_values(weather_series)

            weather_transformer = Scaler()
            weather_transformed = weather_transformer.fit_transform(weather_series)
        except:
            weather_transformed = None

        forecast = model.predict(
            n=24 * days_ahead * measures_per_hour, series=input_series_transformed, past_covariates=weather_transformed
        )
        forecast = transformer.inverse_transform(forecast)
        forecast = forecast.pd_dataframe().reset_index()
        forecast = forecast[['ds', 'y']]
        forecast.columns = ['ds', 'yhat_transformer']
        forecast['ds'] = forecast['ds'].apply(str)

        return forecast

    def TrainAndPredictTransformer(data, weather_data, diff_time, days_ahead, n_epochs):
        """
        Component to train and predict
        """

        model, metric = Train_Transformer(data, weather_data, diff_time=diff_time, n_epochs = n_epochs)
        preds_test = PredictFromTransformer(model, data, weather_data, diff_time, days_ahead)

        return preds_test, model

    def SaveTransformerModel(model, path_to_model):
        """
        Component to save the model
        """
        model.save(path_to_model)

    def ProduceTransformersModel(url_minio, 
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
            pilot_name = pilot_name,
            measurement = measurement_name,
            asset = asset_name
        )

        model_name = "transformers_model_{date}.pt".format(date = datetime.strftime(maya.now().datetime(), "%Y_%m_%d"))

        if client.bucket_exists(bucket_name) != True:
            client.make_bucket(bucket_name)

        # SAVE MODEL FILE(S)

        client.fput_object(bucket_name,
                        model_name,
                        file_path = path_to_model)
        
        client.fput_object(bucket_name,
                        model_name + ".ckpt",
                        file_path = path_to_model+ ".ckpt")
        
        # UPDATE MODEL CONFIGURATION

        try:
            client.fget_object(bucket_name, 
                               "asset_transformers_config.json",
                               "transformers_config.json")
            with open("transformers_config.json") as file:
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
        
        with open("transformers_config.json", "w") as file:
            json.dump(dict_config, file)
        
        client.fput_object(bucket_name, 
                               "asset_transformers_config.json",
                               "transformers_config.json")
        
        return model_name

    # READ DATA COMING FROM PREVIOUS STEPS
    with open(input_data_path) as file:
        data_str = json.load(file)

    data = pd.DataFrame(data_str)
    data = data[data.asset_name == asset_name]

    weather_data = pd.read_feather(input_weather_path)

    print("The data we are using to train the model is:\n")
    print(data.head())

    pred_test, model = TrainAndPredictTransformer(data, weather_data, diff_time, num_days, n_epochs = n_epochs)

    pred_test.to_feather(forecast_data_path)

    SaveTransformerModel(model, "transformers_model.pt")

    model_name = ProduceTransformersModel(
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

    with open(results_path, "w"):
        json.dump(results_dict, results_path)


