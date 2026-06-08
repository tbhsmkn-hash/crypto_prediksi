import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from modules.optimizer import optimize_svr

FEATURE_COLUMNS = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_7",
    "Lag_14",

    "SMA_7",
    "SMA_14",

    "EMA_7",
    "EMA_14",

    "RSI_14",

    "Return_1",
    "Return_7",

    "Volatility_7"
]
#main hybrid function

def train_hybrid_model(df):
     split_index = int(
        len(df) * 0.80
    )
     train_df = df.iloc[:split_index]
     test_df = df.iloc[split_index:]
     
     train_close = train_df["Close"]
     test_close = test_df["Close"]

def train_hybrid_model(df):

    split_index = int(
        len(df) * 0.80
    )

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    train_close = train_df["Close"]
    test_close = test_df["Close"]
    # arima model
    arima_model = auto_arima(
        train_close,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    arima_train_pred = pd.Series(
        arima_model.predict_in_sample(),
        index=train_close.index
    )

    residual_train = (
        train_close -
        arima_train_pred
    )

    # fitur train
    X_train = train_df[
        FEATURE_COLUMNS
    ]
    y_train = residual_train

    # scaling
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )
    #bayesian SVR
    svr_model = optimize_svr(
        X_train_scaled,
        y_train
    )
    # train residual prediction
    train_residual_pred = svr_model.predict(
        X_train_scaled
    )

    hybrid_train_pred = (
        arima_train_pred.values +
        train_residual_pred
    )
    # -------------------------------
    # forecast test arima
    n_test = len(test_df)
    arima_test_forecast = arima_model.predict(
        n_periods=n_test
    )
    
    # test fitur
    X_test = test_df[
        FEATURE_COLUMNS
    ]

    X_test_scaled = scaler.transform(
        X_test
    )

    # Residual Forecast
    residual_test_pred = svr_model.predict(
        X_test_scaled
    )
    #Hybrid Forecast
    hybrid_test_pred = (
        arima_test_forecast +
        residual_test_pred
    )
    # Return Object
    return {

        "arima_model":
            arima_model,

        "svr_model":
            svr_model,

        "scaler":
            scaler,

        "train_actual":
            train_close,

        "test_actual":
            test_close,

        "arima_train":
            arima_train_pred,

        "hybrid_train":
            hybrid_train_pred,

        "arima_test":
            arima_test_forecast,

        "hybrid_test":
            hybrid_test_pred
    }
# forecast masa depan
def future_forecast(
    model_result,
    periods=7
):

    arima_model = model_result[
        "arima_model"
    ]

    future_pred = arima_model.predict(
        n_periods=periods
    )

    return future_pred

def create_train_test_split(
    df,
    train_size=0.80
):

    split_index = int(
        len(df) * train_size
    )

    train_df = df.iloc[:split_index]

    test_df = df.iloc[split_index:]

    return train_df, test_df

# ===================================
# SVR ONLY MODEL
# ===================================

svr_only = optimize_svr(
    X_train_scaled,
    train_close.values
)

svr_test_pred = svr_only.predict(
    X_test_scaled
)
# return dictionary
return {

    "arima_model": arima_model,
    "svr_model": svr_model,
    "svr_only_model": svr_only,
    "scaler": scaler,
    "train_actual": train_close,
    "test_actual": test_close,
    "arima_train": arima_train_pred,
    "hybrid_train": hybrid_train_pred,
    "arima_test": arima_test_forecast,
    "svr_test": svr_test_pred,
    "hybrid_test": hybrid_test_pred
}

