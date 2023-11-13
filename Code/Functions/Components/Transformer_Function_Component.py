import pandas as pd
from darts.models import TransformerModel
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values

def Train_Transformer(train_data,
    diff_time: int):

    """

    Trains a Transformer model based on train_data.

    Parameters
    ----------
    train_data: Data in the format ["ds", "y"], where ds is a columns in datetime or string = "%Y-%m-%d %H:%M:%S
    diff_time: time in minutes between two consecutive measurements; missing values will be interpolated

    Returns
    -------
    model: The model trained

    """

    n_epochs = 50
    batch_size = 16
    split_proportion = 0.9

    # all time steps must exist in the time series; this is guaranteed by setting freq when creating the time series
    series = TimeSeries.from_dataframe(train_data, 'ds', 'y', fill_missing_dates=True, freq="{minutes}T".format(minutes=diff_time))

    # no missing values should be in the series
    series = fill_missing_values(series)

    # Create training and validation sets:
    train, val = series.split_after(pd.Timestamp(series.start_time() + pd.Timedelta(hours=int(len(series) * split_proportion))))

    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)
    # series_transformed = transformer.transform(series)

    measures_per_hour = int(60/int(diff_time))

    model = TransformerModel(
        input_chunk_length=2*24*measures_per_hour,
        output_chunk_length=24*measures_per_hour,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        activation="relu",
        batch_size=batch_size,
        n_epochs=n_epochs,
        optimizer_kwargs={"lr": 1e-3},
        model_name="data_transformer",
        log_tensorboard=True,
        random_state=42,
        force_reset=True,
        save_checkpoints=True
    )

    model.fit(
        train_transformed,
        val_series=val_transformed,
        verbose=True
    )

    return model


def PredictFromTransformer(model, input_data,
                       diff_time,
                       days_ahead = 1):

    """
    Takes a Transformer model and makes the prediction.

    Parameters
    ----------
    data: Data in the format ["ds", "y"], where ds is a columns in datetime or string = "%Y-%m-%d %H:%M:%S.
            this dataset is used as input for the model; must have length >= input_chunk_length specified in TransformerModel; forecasts start following the end of this dataset
    diff_time: time in minutes between two consecutive measurements; missing values will be interpolated
    days_ahead: number of days to forecast

    Returns
    -------
    forecast: The model trained

    """

    measures_per_hour = int(60/int(diff_time))

    input_series = TimeSeries.from_dataframe(input_data, 'ds', 'y',fill_missing_dates=True, freq="{minutes}T".format(minutes = diff_time))
    input_series = fill_missing_values(input_series)

    transformer = Scaler()
    input_series_transformed = transformer.fit_transform(input_series)

    forecast = model.predict(n=24*days_ahead*measures_per_hour, series=input_series_transformed)
    forecast = transformer.inverse_transform(forecast)
    forecast = forecast.pd_dataframe().reset_index()

    return forecast


def TrainAndPredictTransformer(train_data, input_data, diff_time, days_ahead):
    """

    Component to train and predict

    """

    model = Train_Transformer(train_data, diff_time=diff_time)
    preds_test = PredictFromTransformer(model, input_data, diff_time, days_ahead)

    return model, preds_test

def LoadAndForecastTransformer(path_to_model, input_data, diff_time, days_ahead):
    """

    Component to load the model and predict

    """
    model = TransformerModel.load(path_to_model)

    forecast = PredictFromTransformer(model, input_data, diff_time, days_ahead)

    return forecast


def SaveTransformerModel(model, path_to_model):
    """

    Component to save the model

    """
    model.save(path_to_model)
