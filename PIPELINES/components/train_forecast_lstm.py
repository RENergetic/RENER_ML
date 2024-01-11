from kfp.components import InputPath, OutputPath

def ForecastLSTM(input_data_path: InputPath(str),
    input_weather_path: InputPath(str),
    diff_time,
    num_days,
    asset_name,
    forecast_data_path: OutputPath(str),
    model_saved_path: OutputPath(str)

):
    
    # LIBRARIES REQUIRED    

    import numpy as np
    import json
    import pandas as pd

    # FUNCTIONS

    from tqdm import tqdm
    from darts.models import RNNModel
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.utils.missing_values import fill_missing_values
    from darts.metrics import mae, rmse

    def Train_LSTM(train_data,
                weather_data,
                metric='MAE',
                params_grid={'hidden_dim': [25], 'n_rnn_layers': [1]},
                diff_time: int = 60):

        n_epochs = 100
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
            pd.Timestamp(series.start_time() + pd.Timedelta(hours=int(len(series) * split_proportion)))
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

        return best_model, best_metric

    def PredictFromLSTM(model, data, weather_data,
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
        transformer = Scaler()
        input_series_transformed = transformer.fit_transform(input_series)

        # TODO check if weather data covers enough time to be used as future covariates in RNNModel
        try:
            weather_series = TimeSeries.from_dataframe(
                weather_data, time_col='ds', fill_missing_dates=True, freq="{minutes}T".format(minutes=diff_time)
            )
            weather_series = fill_missing_values(weather_series)

            weather_transformer = Scaler()
            weather_transformed = weather_transformer.fit_transform(weather_series)
        except:
            weather_transformed = None

        forecast = model.predict(n=days_ahead*24*measures_per_hour, series = input_series_transformed, future_covariates=weather_transformed)
        forecast = transformer.inverse_transform(forecast)
        forecast = forecast.pd_dataframe().reset_index()
        forecast = forecast[['ds', 'y']]
        forecast.columns = ['ds', 'yhat_lstm']
        forecast['ds'] = forecast['ds'].apply(str)

        return forecast


    def TrainAndPredictLSTM(data, diff_time, days_ahead):
        """
        
        Component to train and predict

        """

        model, metric = Train_LSTM(data, diff_time=diff_time)
        preds_test = PredictFromLSTM(model, data, diff_time, days_ahead)

        return preds_test, model

    def SaveLSTMModel(model, path_to_model):
        model.save(path_to_model)

    # READ DATA COMING FROM PREVIOUS STEPS

    with open(input_data_path) as file:
        data_str = json.load(file)
    
    data = pd.DataFrame(data_str)
    data = data[data.asset_name == asset_name]

    weather_data = pd.read_feather(input_weather_path)

    pred_test, model = TrainAndPredictLSTM(data, weather_data, diff_time, num_days)

    pred_test.to_feather(forecast_data_path)

    SaveLSTMModel(model, model_saved_path)
