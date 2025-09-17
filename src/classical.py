from __future__ import annotations
import warnings
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Silence only ARIMA convergence warnings during fitting
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning as SMConvergenceWarning
except Exception:
    SMConvergenceWarning = None

# Optional deps
try:
    import pmdarima as pm
except Exception:
    pm = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None

HAS_PMDARIMA = pm is not None
HAS_PROPHET  = Prophet is not None

# Auto-ARIMA training: One small, focused function: fit on train only (no leakage), with simple knobs for seasonality.
def train_auto_arima(
    y_train: pd.Series,
    seasonal: bool = False,
    m: Optional[int] = None,   # season length (7 for daily with weekly seasonality, 12 for monthly with yearly seasonality, etc.)
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
    max_P: int = 2,
    max_Q: int = 2,
    max_D: int = 1,
    stepwise: bool = True,
    suppress_warnings: bool = True,
):
    """
    Fit an auto-ARIMA on the training series only.
    Returns a fitted pmdarima model, or None if pmdarima is not installed.
    """
    if pm is None:
        warnings.warn("pmdarima is not installed. Skipping ARIMA.")
        return None

    if seasonal and (m is None or m < 1):
        raise ValueError("When seasonal=True, provide a positive integer m (season length).")

    # Fit while suppressing noisy convergence warnings from statsmodels
    with warnings.catch_warnings():
        # Ignore general user-level warnings
        warnings.simplefilter("ignore", category=UserWarning)

        # Ignore statsmodels convergence warnings, if available
        if SMConvergenceWarning is not None:
            warnings.simplefilter("ignore", category=SMConvergenceWarning)

        # Ignore the scikit-learn rename warning from pmdarima
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r".*'force_all_finite'.*"
        )

        model = pm.auto_arima(
            y_train,
            seasonal=seasonal,
            m=(m or 1),
            start_p=0, start_q=0, start_P=0, start_Q=0,
            max_p=max_p, max_q=max_q, max_d=max_d,
            max_P=max_P, max_Q=max_Q, max_D=max_D,
            stepwise=stepwise,
            information_criterion="aic",
            suppress_warnings=suppress_warnings,
            error_action="ignore",
            trace=False,
        )
    return model


# Auto-ARIMA forecasting + intervals: We forecast exactly over test index (aligns outputs cleanly for plotting and metrics).
def forecast_auto_arima(
    model,
    test_index: pd.DatetimeIndex,
    alpha: float = 0.05,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Forecast over the given test_index length and return
    (y_pred, y_lower, y_upper) as Series indexed by test_index.
    """
    if model is None:
        raise ValueError("ARIMA model is None. Did training succeed?")

    n_periods = len(test_index)

    # Silence sklearn rename warning triggered inside pmdarima during predict
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r".*'force_all_finite'.*"
        )
        y_pred_np, conf = model.predict(
            n_periods=n_periods,
            return_conf_int=True,
            alpha=alpha,
        )

    y_pred = pd.Series(y_pred_np, index=test_index, name="arima_pred")
    y_low  = pd.Series(conf[:, 0], index=test_index, name="arima_low")
    y_high = pd.Series(conf[:, 1], index=test_index, name="arima_high")
    return y_pred, y_low, y_high

# Prophet training: Minimal, business-friendly defaults. We let you toggle standard seasonalities.
def train_prophet(
    y_train: pd.Series,
    weekly: bool = True,
    yearly: bool = True,
    daily: bool = False,
    interval_width: float = 0.95,
) -> Optional[object]:
    """
    Fit a Prophet model on train only.
    Expects a DatetimeIndex and a single numeric Series.
    """
    if Prophet is None:
        warnings.warn("prophet is not installed. Skipping Prophet.")
        return None

    df = pd.DataFrame({"ds": y_train.index, "y": y_train.values})

    model = Prophet(
        weekly_seasonality=weekly,
        yearly_seasonality=yearly,
        daily_seasonality=daily,
        interval_width=float(interval_width),
    )
    model.fit(df)
    return model

# Prophet forecasting + intervals: Prophet plays nicely if you just pass a future frame whose ds equals your test dates.
def forecast_prophet(
    model: object,
    test_index: pd.DatetimeIndex,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Forecast over test_index with Prophet.
    Returns (y_pred, y_lower, y_upper) as Series indexed by test_index.
    """
    if model is None:
        raise ValueError("Prophet model is None. Did training succeed?")

    future = pd.DataFrame({"ds": test_index})
    fcst = model.predict(future)

    y_pred = pd.Series(fcst["yhat"].values, index=test_index, name="prophet_pred")
    y_low  = pd.Series(fcst["yhat_lower"].values, index=test_index, name="prophet_low")
    y_high = pd.Series(fcst["yhat_upper"].values, index=test_index, name="prophet_high")
    return y_pred, y_low, y_high

# Plot: Keeps visuals consistent across models. This is for forecasts with bands.
def plot_forecast_with_ci(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: pd.Series,
    y_low: pd.Series,
    y_high: pd.Series,
    title: str = "",
):
    """
    Matplotlib line + band. Assumes DatetimeIndex everywhere.
    Returns a matplotlib Figure for Streamlit or saving.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    y_train.plot(ax=ax, label="train")
    y_test.plot(ax=ax, label="test")
    y_pred.plot(ax=ax, label="forecast")

    ax.fill_between(
        y_pred.index,
        y_low.values,
        y_high.values,
        alpha=0.2,
        label="conf. interval",
        linewidth=0,
    )
    ax.set_title(title or "Forecast")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return fig

