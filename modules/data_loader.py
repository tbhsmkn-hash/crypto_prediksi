import pandas as pd
import requests
import streamlit as st

COINS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "PEPE": "pepe",
    "MOODENG": "moo-deng"
}


@st.cache_data(ttl=3600)
def get_crypto_data(coin_id: str, days: int = 365):

    url = (
        f"https://api.coingecko.com/api/v3/coins/"
        f"{coin_id}/market_chart"
        f"?vs_currency=usd&days={days}"
    )

    try:
        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            return pd.DataFrame()

        data = response.json()

        prices = data["prices"]
        volumes = data["total_volumes"]

        df_price = pd.DataFrame(
            prices,
            columns=["timestamp", "close"]
        )

        df_vol = pd.DataFrame(
            volumes,
            columns=["timestamp", "volume"]
        )

        df = pd.merge(
            df_price,
            df_vol,
            on="timestamp"
        )

        df["Date"] = pd.to_datetime(
            df["timestamp"],
            unit="ms"
        )

        df = df[[
            "Date",
            "close",
            "volume"
        ]]

        df.columns = [
            "Date",
            "Close",
            "Volume"
        ]

        df.set_index("Date", inplace=True)

        return df

    except Exception:
        return pd.DataFrame()