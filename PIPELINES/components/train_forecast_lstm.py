from kfp.components import InputPath, OutputPath

def ForecastLSTM(input_data_path: InputPath(str),
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

    def Train_LSTM(train_data, 
        diff_time: int):

        n_epochs = 100
        batch_size = 16

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

        transformer = Scaler()
        series_transformed = transformer.fit_transform(series)

        measures_per_hour = int(60/int(diff_time))

        model = RNNModel(
            input_chunk_length=24*measures_per_hour, # TODO use one day as input; OK?
            model="LSTM",
            hidden_dim=25, 
            n_rnn_layers=1,
            # dropout=0.2, # this does not have an effect with only one rnn_layer
            training_length=2*24*measures_per_hour, # should be larger than input_chunk_length according to Darts docs
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer_kwargs={"lr": 1e-3},
            model_name="data_RNN",
            log_tensorboard=True,
            random_state=42,
            force_reset=True,
            save_checkpoints=True
        )

        model.fit(
            series_transformed,
            verbose=True
            # no validation set specified here (which would be used for regularization)
            # the parameter values used for RNNModel() do not seem to allow for regularization, 
            # for which the validation would have been used; so the validation set has no effect (unless the parameters of RNNModel are changed accordingly)?
        )

        return model

    def PredictFromLSTM(model, data,
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

        forecast = model.predict(n=days_ahead*measures_per_hour, series = input_series_transformed)
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

        model = Train_LSTM(data, diff_time=diff_time)
        preds_test = PredictFromLSTM(model, data, diff_time, days_ahead)

        return preds_test, model

    def SaveLSTMModel(model, path_to_model):
        model.save(path_to_model)

    # READ DATA COMING FROM PREVIOUS STEPS

    with open(input_data_path) as file:
        data_str = json.load(file)
    
    data = pd.DataFrame(data_str)
    data = data[data.asset_name == asset_name]

    pred_test, model = TrainAndPredictLSTM(data, diff_time, num_days)

    pred_test.to_feather(forecast_data_path)

    SaveLSTMModel(model, model_saved_path)
