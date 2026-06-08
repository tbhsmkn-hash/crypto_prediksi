import plotly.graph_objects as go


def plot_hybrid_results(results):

    fig = go.Figure()

    # TRAIN ACTUAL
    fig.add_trace(
        go.Scatter(
            x=results["train_actual"].index,
            y=results["train_actual"].values,
            mode="lines",
            name="Train Actual"
        )
    )

    # TEST ACTUAL
    fig.add_trace(
        go.Scatter(
            x=results["test_actual"].index,
            y=results["test_actual"].values,
            mode="lines",
            name="Test Actual"
        )
    )

    # ARIMA TEST
    fig.add_trace(
        go.Scatter(
            x=results["test_actual"].index,
            y=results["arima_test"],
            mode="lines",
            name="ARIMA Forecast"
        )
    )

    # HYBRID TEST
    fig.add_trace(
        go.Scatter(
            x=results["test_actual"].index,
            y=results["hybrid_test"],
            mode="lines",
            name="Hybrid ARIMA-SVR"
        )
    )

    fig.update_layout(
        title="Hybrid ARIMA-SVR Prediction",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        height=650
    )

    return fig

# Forecast Visualization
def plot_future_forecast(
    historical_data,
    future_dates,
    future_values
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data.values,
            mode="lines",
            name="Historical Price"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=future_values,
            mode="lines+markers",
            name="Forecast"
        )
    )

    fig.update_layout(
        title="Future Cryptocurrency Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        height=650
    )

    return fig
# Performance Visualization
def plot_metrics(metrics_dict):

    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(
        metrics_dict
    )

    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        title="Model Performance"
    )

    return fig
#benchmark plot
def plot_model_comparison(results):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results["test_actual"].index,
            y=results["test_actual"],
            mode="lines",
            name="Actual"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results["test_actual"].index,
            y=results["arima_test"],
            mode="lines",
            name="ARIMA"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results["test_actual"].index,
            y=results["svr_test"],
            mode="lines",
            name="SVR"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results["test_actual"].index,
            y=results["hybrid_test"],
            mode="lines",
            name="Hybrid"
        )
    )

    fig.update_layout(
        title="Model Benchmark Comparison",
        height=700,
        hovermode="x unified"
    )

    return fig