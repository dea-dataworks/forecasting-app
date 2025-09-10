import numpy as np
import pandas as pd
import pytest
from src.baselines import evaluate_forecast  # (y_true: Series, y_pred: Series) -> dict

def _get_metric(d: dict, name_prefix: str):
    name_prefix = name_prefix.lower()
    for k, v in d.items():
        if k.lower().startswith(name_prefix):
            return v
    raise KeyError(f"Metric starting with '{name_prefix}' not found in {list(d)}")

def test_basic_metrics_small_series():
    y_true = pd.Series([1, 2, 3], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    y_pred = pd.Series([1, 2, 2], index=y_true.index)
    metrics = evaluate_forecast(y_true, y_pred)

    mae = _get_metric(metrics, "mae")
    rmse = _get_metric(metrics, "rmse")
    assert mae == pytest.approx(1/3, rel=1e-6, abs=1e-6)
    assert rmse == pytest.approx((1/3) ** 0.5, rel=1e-6, abs=1e-6)

    mape = _get_metric(metrics, "mape")  # percent or fraction—just ensure valid
    assert np.isfinite(mape) and mape >= 0

def test_handles_nans_and_zeros_safely():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    y_true = pd.Series([0, 1, 2, 3], index=idx)
    y_pred = pd.Series([0, np.nan, 2, 3], index=idx)

    metrics = evaluate_forecast(y_true, y_pred)
    mae = _get_metric(metrics, "mae")
    rmse = _get_metric(metrics, "rmse")
    assert np.isfinite(mae) and np.isfinite(rmse)

