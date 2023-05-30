from darts.models import RNNModel
from darts.models import TransformerModel
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, coefficient_of_variation, mae, mse, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import pandas as pd
############ Warning and Logging Suppression ############
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

def Train_LSTM(data, split_proportion, diff_time=60, num_days=1, 
               measures_per_hour=1, n_epochs=100, batch_size=16):
    """
    Train an LSTM model on the given time series data.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing the time series data with 'ds' and 'y' columns.
    split_proportion : float
        The proportion of the data to be used for training. The rest will be used for validation.
    diff_time : int, optional, default: 60
        The time difference between two consecutive measures in the time series data, measured in minutes.
    num_days : int, optional, default: 1
        The number of days for which the model should make predictions.
    measures_per_hour : int, optional, default: 1
        The number of measures per hour in the time series data.
    n_epochs : int, optional, default: 100
        The number of epochs to train the LSTM model.
    batch_size : int, optional, default: 16
        The batch size for training the LSTM model.

    Returns
    -------
    forecast : TimeSeries
        The forecasted values as a TimeSeries object.
    series : TimeSeries
        The original time series data as a TimeSeries object.
    val : TimeSeries
        The validation time series data as a TimeSeries object.
    my_model : RNNModel
        The trained LSTM model.
    """

    if not isinstance(data, pd.DataFrame) or 'ds' not in data.columns or 'y' not in data.columns:
        raise ValueError("The input data must be a pandas DataFrame with 'ds' and 'y' columns.")

    if not (0 < split_proportion < 1):
        raise ValueError("The split_proportion must be a float between 0 and 1.")

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

    # create month and year covariate series
    year_series = datetime_attribute_timeseries(
        pd.date_range(start=series.start_time(), freq=series.freq_str, periods=len(series) * 2),
        attribute="year",
        one_hot=False,
    )
    year_series = Scaler().fit_transform(year_series)
    month_series = datetime_attribute_timeseries(
        year_series, attribute="month", one_hot=True
    )
    covariates = year_series.stack(month_series)
    cov_train, cov_val = covariates.split_after(pd.Timestamp(val.start_time() - pd.Timedelta(hours=1)))

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
        future_covariates=cov_train,
        val_series=val_transformed,
        val_future_covariates=cov_val,
        verbose=True,
    )

    # Make forecasts
    forecast = my_model.predict(n=pred_ahead, future_covariates=covariates)

    # Inverse-transform forecasts and obtain the real predicted values
    forecast = transformer.inverse_transform(forecast)

    # Compute the mean absolute percentage error (MAPE)
    print("Test Coefficient of Variation: {:.2f}".format(coefficient_of_variation(forecast, val)))
    print("Test Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape(forecast, val)))
    print("Test Mean Absolute Error (MAE): {:.2f}".format(mae(forecast, val)))
    print("Test Mean Squared Error (MSE): {:.2f}".format(mse(forecast, val)))
    print("Test Root Mean Squared Error (RMSE): {:.2f}".format(rmse(forecast, val)))

    return forecast, series, val, my_model

def Train_Transformer(data, split_proportion, diff_time=60, num_days=1,
                      measures_per_hour=1, n_epochs=100, batch_size=16):
    """
    Train a Transformer model on the given time series data.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing the time series data with 'ds' and 'y' columns.
    split_proportion : float
        The proportion of the data to be used for training. The rest will be used for validation.
    diff_time : int, optional, default: 60
        The time difference between two consecutive measures in the time series data, measured in minutes.
    num_days : int, optional, default: 1
        The number of days for which the model should make predictions.
    measures_per_hour : int, optional, default: 1
        The number of measures per hour in the time series data.
    n_epochs : int, optional, default: 100
        The number of epochs to train the Transformer model.
    batch_size : int, optional, default: 16
        The batch size for training the Transformer model.

    Returns
    -------
    forecast : TimeSeries
        The forecasted values as a TimeSeries object.
    series : TimeSeries
        The original time series data as a TimeSeries object.
    val : TimeSeries
        The validation time series data as a TimeSeries object.
    my_model : TransformerModel
        The trained Transformer model.
    """

    if not isinstance(data, pd.DataFrame) or 'ds' not in data.columns or 'y' not in data.columns:
        raise ValueError("The input data must be a pandas DataFrame with 'ds' and 'y' columns.")

    if not (0 < split_proportion < 1):
        raise ValueError("The split_proportion must be a float between 0 and 1.")

    # Fill missing values with the last available value
    data = data.fillna(method='ffill')

    # Create a time series
    series = TimeSeries.from_dataframe(data, 'ds', 'y', fill_missing_dates=True, freq="{minutes}T".format(minutes = diff_time))
    
    # Create training and validation sets:
    train, val = series.split_after(pd.Timestamp(series.start_time() + pd.Timedelta(hours=int(len(series) * split_proportion))))
    
    # Normalize the time series (note: we avoid fitting the transformer on the validation set)
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)
    series_transformed = transformer.transform(series)

    # predict num_days days ahead
    pred_ahead = 24 * (2 + num_days) * measures_per_hour 

    my_model = TransformerModel(
        input_chunk_length = 2 * pred_ahead,
        output_chunk_length = pred_ahead,
        batch_size = batch_size,
        n_epochs = n_epochs,
        model_name = "data_transformer",
        optimizer_kwargs={"lr": 1e-3},
        d_model = 16,
        nhead = 4,
        num_encoder_layers = 2,
        num_decoder_layers = 2,
        dim_feedforward = 128,
        dropout = 0.2,    
        activation = "relu",
        random_state = 42,
        log_tensorboard = True,
        force_reset = True,
        save_checkpoints=True,
    )

    my_model.fit(
        train_transformed,
        val_series=val_transformed,
        verbose=True,
    )

    # Make forecasts
    forecast = my_model.predict(n=pred_ahead)

    # Inverse-transform forecasts and obtain the real predicted values
    forecast = transformer.inverse_transform(forecast)

    # Compute the mean absolute percentage error (MAPE) and other error metrics
    print("Test Coefficient of Variation: {:.2f}".format(coefficient_of_variation(forecast, val)))
    print("Test Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape(forecast, val)))
    print("Test Mean Absolute Error (MAE): {:.2f}".format(mae(forecast, val)))
    print("Test Mean Squared Error (MSE): {:.2f}".format(mse(forecast, val)))
    print("Test Root Mean Squared Error (RMSE): {:.2f}".format(rmse(forecast, val)))

    return forecast, series, val, my_model