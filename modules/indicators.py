
import numpy as np
import pandas as pd


def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.where(
        delta > 0,
        0
    )

    loss = -delta.where(
        delta < 0,
        0
    )

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_features(df):

    df = df.copy()

    # LAG
    for lag in [1, 2, 3, 7, 14]:
        df[f"Lag_{lag}"] = df["Close"].shift(lag)

    # SMA
    df["SMA_7"] = (
        df["Close"]
        .rolling(7)
        .mean()
    )

    df["SMA_14"] = (
        df["Close"]
        .rolling(14)
        .mean()
    )

    # EMA
    df["EMA_7"] = (
        df["Close"]
        .ewm(span=7)
        .mean()
    )

    df["EMA_14"] = (
        df["Close"]
        .ewm(span=14)
        .mean()
    )

    # RSI
    df["RSI_14"] = calculate_rsi(df["Close"])

    # Return
    df["Return_1"] = (
        df["Close"]
        .pct_change(1)
    )

    df["Return_7"] = (
        df["Close"]
        .pct_change(7)
    )

    # Volatility
    df["Volatility_7"] = (
        df["Return_1"]
        .rolling(7)
        .std()
    )

    df.dropna(inplace=True)

    return df