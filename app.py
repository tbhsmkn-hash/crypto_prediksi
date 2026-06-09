import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from modules.data_loader import (
    COINS,
    get_crypto_data
)
from modules.indicators import (
    add_features
)
from modules.hybrid_model import (
    train_hybrid_model,
    future_forecast
)
from modules.metrics import (
    calculate_metrics,
    directional_accuracy
)
from modules.visualization import (
    plot_hybrid_results,
    plot_future_forecast
)
from modules.visualization import (
    plot_model_comparison
)

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Prediksi Crypto Hybrid ARIMA-SVR",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("📈 Hybrid ARIMA-SVR Cryptocurrency Forecasting")

st.markdown("""
Sistem Prediksi Cryptocurrency menggunakan:

- Auto ARIMA
- Bayesian Optimized SVR
- Hybrid ARIMA-SVR
- CoinGecko Market Data
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Pengaturan")

selected_symbol = st.sidebar.selectbox(
    "Pilih Cryptocurrency",
    list(COINS.keys())
)

selected_coin = COINS[selected_symbol]

days_history = st.sidebar.slider(
    "Data Historis (Hari)",
    min_value=180,
    max_value=1825,
    value=365
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon",
    min_value=1,
    max_value=30,
    value=7
)

run_button = st.sidebar.button(
    "🚀 Run Prediction",
    use_container_width=True
)

# =====================================================
# MAIN PROCESS
# =====================================================

if run_button:
    with st.spinner("Downloading market data..."):
        raw_df = get_crypto_data(
            selected_coin,
            days_history
        )
    if raw_df.empty:
        st.error("Data tidak tersedia.")
        st.stop()
    st.success("Dataset berhasil dimuat")
    # =================================================
    # DATA PREVIEW
    # =================================================
    st.subheader("Dataset Preview")
    st.dataframe(
        raw_df.tail(20),
        use_container_width=True
    )

    csv_data = raw_df.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download Dataset CSV",
        data=csv_data,
        file_name=f"{selected_symbol}_dataset.csv",
        mime="text/csv"
    )

    # =================================================
    # FEATURE ENGINEERING
    # =================================================

    with st.spinner("Building features..."):
        feature_df = add_features(
            raw_df
        )

    st.subheader("Feature Engineered Dataset")

    st.dataframe(
        feature_df.tail(),
        use_container_width=True
    )

    # =================================================
    # TRAIN MODEL
    # =================================================
    with st.spinner(
        "Training Auto ARIMA + Bayesian SVR..."
    ):
        results = train_hybrid_model(
            feature_df
           # benchmark_rows = []
        )

    st.success(
        "Model berhasil dilatih"
    )

    # =================================================
    # METRICS
    # =================================================

    actual = results["test_actual"]
    hybrid_pred = results["hybrid_test"]
    metric = calculate_metrics(
        actual,
        hybrid_pred
    )

    da = directional_accuracy(
        actual.values,
        hybrid_pred
    )

    st.subheader(
        "Evaluasi Hybrid Model"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "RMSE",
        f"{metric['RMSE']:,.4f}"
    )

    col2.metric(
        "MAE",
        f"{metric['MAE']:,.4f}"
    )

    col3.metric(
        "MAPE",
        f"{metric['MAPE']:.2f}%"
    )

    col4.metric(
        "R²",
        f"{metric['R2']:.4f}"
    )

    col5.metric(
        "Direction Accuracy",
        f"{da:.2f}%"
    )

    # =================================================
    # VISUALIZATION
    # =================================================

    st.subheader(
        "Hybrid ARIMA-SVR Prediction"
    )

    fig_result = plot_hybrid_results(
        results
    )

    st.plotly_chart(
        fig_result,
        use_container_width=True
    )

    # =================================================
    # FUTURE FORECAST
    # =================================================

    st.subheader(
        f"Forecast {forecast_days} Hari Kedepan"
    )

    future_values = future_forecast(
        results,
        periods=forecast_days
    )

    future_dates = pd.date_range(
        start=raw_df.index[-1] + timedelta(days=1),
        periods=forecast_days
    )

    forecast_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Forecast": future_values
        }
    )

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    fig_future = plot_future_forecast(
        raw_df["Close"],
        future_dates,
        future_values
    )

    st.plotly_chart(
        fig_future,
        use_container_width=True
    )

    # =================================================
    # DOWNLOAD FORECAST
    # =================================================
    forecast_csv = forecast_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Forecast",
        forecast_csv,
        file_name=f"{selected_symbol}_forecast.csv",
        mime="text/csv"
    )
    arima_metric = calculate_metrics(
        results["test_actual"],
        results["arima_test"]
    )
    # arima
    arima_da = directional_accuracy(
        results["test_actual"].values,
        results["arima_test"]
    )
    benchmark_rows.append({
        "Model":"ARIMA",
        "RMSE":arima_metric["RMSE"],
        "MAE":arima_metric["MAE"],
        "MAPE":arima_metric["MAPE"],
        "R2":arima_metric["R2"],
        "DA":arima_da
    })
    # svr
    svr_metric = calculate_metrics(
        results["test_actual"],
        results["svr_test"]
    )
    svr_da = directional_accuracy(
        results["test_actual"].values,
        results["svr_test"]
    )
    benchmark_rows.append({
        "Model":"SVR",
        "RMSE":svr_metric["RMSE"],
        "MAE":svr_metric["MAE"],
        "MAPE":svr_metric["MAPE"],
        "R2":svr_metric["R2"],
        "DA":svr_da
    })
    # hybrid 
    hybrid_metric = calculate_metrics(
        results["test_actual"],
        results["hybrid_test"]
    )
    
    hybrid_da = directional_accuracy(
        results["test_actual"].values,
        results["hybrid_test"]
    )
    benchmark_rows.append({
    "Model":"Hybrid ARIMA-SVR",
    "RMSE":hybrid_metric["RMSE"],
    "MAE":hybrid_metric["MAE"],
    "MAPE":hybrid_metric["MAPE"],
    "R2":hybrid_metric["R2"],
    "DA":hybrid_da
    })

    # =================================================
    # MODEL INFO
    # =================================================

    st.subheader(
        "Model Information"
    )

    st.info(
        f"""
        Symbol : {selected_symbol}

        Observasi : {len(raw_df)}

        Feature Engineering :
        Lag, SMA, EMA, RSI,
        Return, Volatility

        Model :
        Auto ARIMA + Bayesian Optimized SVR

        Forecast Horizon :
        {forecast_days} Hari
        """
    )

else:
    st.info(
        "Pilih parameter lalu klik Run Prediction."
    )
