import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)


def calculate_metrics(actual, pred):

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            pred
        )
    )

    mae = mean_absolute_error(
        actual,
        pred
    )

    mape = (
        mean_absolute_percentage_error(
            actual,
            pred
        ) * 100
    )

    r2 = r2_score(
        actual,
        pred
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }


def directional_accuracy(actual, pred):

    actual_dir = np.sign(
        np.diff(actual)
    )

    pred_dir = np.sign(
        np.diff(pred)
    )

    return (
        (actual_dir == pred_dir)
        .mean()
        * 100
    )