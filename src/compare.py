import warnings
from typing import Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from prophet import Prophet  # optional
except Exception:  # pragma: no cover
    Prophet = None  # type: ignore
from typing import Any, Dict, Tuple
import pandas as pd

def validate_horizon(horizon: int, test_len_like) -> int:
    """
    Validate the forecast horizon against the test-set length.

    Parameters
    ----------
    horizon : int
        Desired forecast horizon (number of steps ahead). Must be > 0.
    test_len_like : Any
        Either an integer test length, or any object with a defined len()
        (e.g., a list, Series, or array representing the test set).

    Returns
    -------
    int
        A safe horizon value clipped to the available test length.

    Raises
    ------
    ValueError
        If horizon <= 0 or resolved test length <= 0.
    TypeError
        If test_len_like is neither an int nor a length-like object.
    """
    # coerce/validate horizon
    try:
        h = int(horizon)
    except Exception as e:
        raise TypeError("horizon must be an integer.") from e
    if h <= 0:
        raise ValueError("Forecast horizon must be positive.")

    # resolve test length from int or length-like
    if isinstance(test_len_like, int):
        t = test_len_like
    else:
        try:
            t = len(test_len_like)
        except Exception as e:
            raise TypeError("test_len_like must be an int or have a length.") from e

    if t <= 0:
        raise ValueError("Test length must be positive.")

    return min(h, t)

def make_future_index(last_ts: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    """
    Build a future DatetimeIndex starting one step after `last_ts`.

    Parameters
    ----------
    last_ts : pd.Timestamp
        Last timestamp in the training data.
    periods : int
        Number of future steps to generate (must be > 0).
    freq : str
        Pandas offset alias (e.g., 'D', 'W', 'M', 'H') matching the series frequency.

    Returns
    -------
    pd.DatetimeIndex
        Regular future index of length `periods` at frequency `freq`.

    Raises
    ------
    ValueError
        If periods <= 0 or freq is missing/invalid.
    """
    if periods <= 0:
        raise ValueError("`periods` must be a positive integer.")
    if not freq:
        raise ValueError("`freq` must be provided (e.g., 'D', 'M', 'H').")

    step = pd.tseries.frequencies.to_offset(freq)
    start = pd.Timestamp(last_ts) + step
    return pd.date_range(start=start, periods=periods, freq=freq)

def forecast_with_adapter(model: Any, horizon: int, last_ts: pd.Timestamp, freq: str) -> pd.Series:
    """
    Produce a horizon-length forecast as a pd.Series indexed by a common future index.
    Works across supported model types by normalizing their predict APIs.

    Supported:
    - pmdarima: has .predict(n_periods=h)
    - statsmodels SARIMAXResults: has .get_forecast(steps=h).predicted_mean
    - Prophet: Prophet model with .predict(future_df) → 'yhat'
    - Callable baselines: callable that returns array-like of length h
      (tried as fn(h) then fn(h=h) )

    Parameters
    ----------
    model : Any
        Trained model or callable forecaster.
    horizon : int
        Number of steps ahead to forecast (already validated).
    last_ts : pd.Timestamp
        Last timestamp of the training series.
    freq : str
        Pandas frequency string for the series (e.g., 'D', 'W', 'M').

    Returns
    -------
    pd.Series
        Forecast values indexed by a shared future DatetimeIndex.

    Raises
    ------
    ValueError
        If the model type is unsupported or returns the wrong length.
    """
    future_idx = make_future_index(last_ts, periods=horizon, freq=freq)

    # 1) Prophet
    if Prophet is not None and isinstance(model, Prophet):
        future_df = pd.DataFrame({"ds": future_idx})
        fcst_df = model.predict(future_df)
        if "yhat" not in fcst_df:
            raise ValueError("Prophet predict() did not return 'yhat'.")
        return pd.Series(fcst_df["yhat"].to_numpy(), index=future_idx, dtype="float64")

    # Helper: safe length check + wrap
    def _wrap(values) -> pd.Series:
        try:
            arr = pd.Series(values, index=future_idx, dtype="float64")
        except Exception:
            # fall back: coerce to list then Series
            arr = pd.Series(list(values), index=future_idx, dtype="float64")
        if len(arr) != horizon:
            raise ValueError(f"Forecaster returned length {len(arr)} != horizon {horizon}.")
        return arr

    # 2) pmdarima (duck-typing by module name + predict signature)
    mod_name = getattr(getattr(model, "__class__", None), "__module__", "") or ""
    if hasattr(model, "predict") and "pmdarima" in mod_name:
        values = model.predict(n_periods=horizon)
        return _wrap(values)

    # 3) statsmodels SARIMAXResults or similar with get_forecast
    if hasattr(model, "get_forecast"):
        res = model.get_forecast(steps=horizon)
        if hasattr(res, "predicted_mean"):
            pm = res.predicted_mean
            # Ensure index matches our future_idx
            try:
                pm = pm.astype("float64")
            except Exception:
                pm = pd.Series(pm, dtype="float64")
            # If statsmodels index matches in length, just reindex by position
            if len(pm) == horizon:
                pm.index = future_idx
                return pm
        # Fallback: try to use res.mean if available
        if hasattr(res, "mean"):
            return _wrap(res.mean)
        raise ValueError("get_forecast() did not yield a usable predicted_mean/mean.")

    # 4) Callable baselines (e.g., naive, moving average) → array-like
    if callable(model):
        try:
            values = model(horizon)  # try positional
        except TypeError:
            values = model(h=horizon)  # try named
        return _wrap(values)

    raise ValueError(f"Unsupported model type for forecasting: {type(model)}")

def generate_forecasts(
    models: Dict[str, Any],
    horizon: int,
    last_ts: pd.Timestamp,
    freq: str
) -> Dict[str, pd.Series]:
    """
    Loop over trained models and produce aligned forecast Series for each.

    Parameters
    ----------
    models : dict[str, Any]
        Mapping of model name -> trained model (or callable baseline).
    horizon : int
        Steps ahead to forecast (should already be validated).
    last_ts : pd.Timestamp
        Last timestamp of the training data.
    freq : str
        Pandas frequency string (e.g., 'D', 'W', 'M').

    Returns
    -------
    dict[str, pd.Series]
        Mapping of model name -> forecast Series (float64) on the same future index.

    Notes
    -----
    - Skips models that are None or fail to forecast, with a warning.
    - Ensures each returned Series has the same DatetimeIndex and float dtype.
    """
    forecasts: Dict[str, pd.Series] = {}

    for name, model in models.items():
        if model is None:
            warnings.warn(f"{name}: model is None, skipping.")
            continue
        try:
            fcst = forecast_with_adapter(model, horizon, last_ts, freq)
            # Standardize dtype/name and ensure index length
            fcst = pd.Series(fcst.to_numpy(dtype="float64"), index=fcst.index, name=name)
            if len(fcst) != horizon:
                warnings.warn(f"{name}: forecast length {len(fcst)} != horizon {horizon}, skipping.")
                continue
            forecasts[name] = fcst
        except Exception as e:
            warnings.warn(f"{name}: skipping forecast due to error: {e}")

    if not forecasts:
        raise ValueError("No forecasts were produced. Check models and horizon.")

    return forecasts

def compute_metrics_table(
    y_true: pd.Series,
    forecasts: Dict[str, pd.Series],
    *,
    include_smape: bool = False,
    include_mase: bool = False,
    y_train_for_mase: pd.Series | None = None,
    sort_by: str = "RMSE",
    ascending: bool = True,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Compute MAE, RMSE, MAPE(%) and optional sMAPE/MASE for each model on the SAME aligned horizon.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth test values with a DatetimeIndex.
    forecasts : dict[str, pd.Series]
        Model name -> forecast Series (must share the same future index length).
    include_smape : bool, default False
        If True, compute sMAPE (%).
    include_mase : bool, default False
        If True, compute MASE using a naive one-step benchmark on `y_train_for_mase`.
    y_train_for_mase : pd.Series or None
        Training series used to compute the naive MAE denominator for MASE. Required if include_mase=True.
    sort_by : {"RMSE","MAE","MAPE%","sMAPE%","MASE"}, default "RMSE"
        Column to sort by (ignored if not present).
    ascending : bool, default True
        Sort order.
    eps : float, default 1e-12
        Small constant to stabilize denominators.

    Returns
    -------
    pd.DataFrame
        Rows = models; columns include ['MAE','RMSE','MAPE%'] plus optional ['sMAPE%','MASE'].
        Sorted with NaNs pushed to the bottom.
    """
    # Precompute MASE denominator if requested and feasible
    mase_den = None
    if include_mase:
        if y_train_for_mase is None or len(y_train_for_mase) < 2:
            mase_den = np.nan
        else:
            ytr = pd.Series(y_train_for_mase, copy=False).astype("float64")
            naive_err = (ytr.iloc[1:].to_numpy() - ytr.iloc[:-1].to_numpy())
            mase_den = float(np.mean(np.abs(naive_err))) if len(naive_err) > 0 else np.nan
            if mase_den <= eps:
                mase_den = np.nan  # undefined / zero denominator

    rows: list[dict[str, float]] = []
    for name, fcst in forecasts.items():
        idx = y_true.index.intersection(fcst.index)
        if len(idx) == 0:
            warnings.warn(f"{name}: no overlapping timestamps with y_true; skipping.")
            continue

        yt = y_true.loc[idx].astype("float64")
        yp = fcst.loc[idx].astype("float64")

        mask = yt.notna() & yp.notna()
        yt = yt[mask]
        yp = yp[mask]
        if len(yt) == 0:
            warnings.warn(f"{name}: no valid aligned points after NaN filtering; skipping.")
            continue

        err = yp - yt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(np.square(err))))

        # Safe MAPE: ignore zeros in denominator
        nonzero = np.abs(yt.to_numpy()) > eps
        if nonzero.any():
            mape = float(np.mean(np.abs(err.to_numpy()[nonzero] / yt.to_numpy()[nonzero])) * 100.0)
        else:
            mape = np.nan

        row = {"Model": name, "MAE": mae, "RMSE": rmse, "MAPE%": mape}

        if include_smape:
            denom = np.abs(yt.to_numpy()) + np.abs(yp.to_numpy()) + eps
            smape = float(np.mean(2.0 * np.abs(err.to_numpy()) / denom) * 100.0)
            row["sMAPE%"] = smape

        if include_mase:
            row["MASE"] = (mae / mase_den) if (mase_den is not None and not np.isnan(mase_den)) else np.nan

        rows.append(row)

    if not rows:
        raise ValueError("No metrics computed; check alignment and inputs.")

    df = pd.DataFrame(rows).set_index("Model")

    # Sort if the column exists; always push NaNs to the bottom
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending, na_position="last")
    else:
        df = df.sort_values(by="RMSE", ascending=True, na_position="last")

    return df

def plot_overlay(
    y_train: pd.Series,
    y_test: pd.Series,
    forecasts: Dict[str, pd.Series],
    tail: int = 200
):
    """
    One figure with: last `tail` points of train, full test, and all model forecasts overlaid.

    Parameters
    ----------
    y_train : pd.Series
        Training series (DatetimeIndex).
    y_test : pd.Series
        Test/holdout series (DatetimeIndex).
    forecasts : dict[str, pd.Series]
        Model name -> forecast Series (DatetimeIndex aligned to future).
    tail : int, default 200
        How many trailing points of train to show for context.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure (caller can render in Streamlit or save to file).
    """
    if tail is not None and tail > 0:
        y_tr = y_train.iloc[-tail:]
    else:
        y_tr = y_train

    fig, ax = plt.subplots(figsize=(10, 5))

    # Context: train tail + full test
    ax.plot(y_tr.index, y_tr.values, label="Train (tail)", alpha=0.6, linewidth=1.2)
    ax.plot(y_test.index, y_test.values, label="Test (actual)", linewidth=1.6)

    # Overlaid forecasts
    for name, fcst in forecasts.items():
        if fcst is None or len(fcst) == 0:
            continue
        ax.plot(fcst.index, fcst.values, label=name, linewidth=1.4)

    ax.set_title("Forecast Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel(y_train.name or "value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    return fig

def comparison_run(
    y_train: pd.Series,
    y_test: pd.Series,
    models: Dict[str, Any],
    horizon: int,
    freq: str
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], "plt.Figure"]:
    """
    Orchestrate Phase 5:
      1) Validate horizon against test length
      2) Generate aligned forecasts for all models
      3) Compute side-by-side metrics table
      4) Create overlay plot

    Parameters
    ----------
    y_train : pd.Series
        Training series (DatetimeIndex).
    y_test : pd.Series
        Test series (DatetimeIndex).
    models : dict[str, Any]
        Mapping model name -> trained model or callable baseline.
    horizon : int
        Requested forecast horizon (steps ahead).
    freq : str
        Pandas frequency string (e.g., 'D', 'W', 'M').

    Returns
    -------
    (metrics_df, forecasts, fig)
        metrics_df : pd.DataFrame
            Rows=models, Cols=['MAE','RMSE','MAPE%'] sorted by RMSE asc.
        forecasts : dict[str, pd.Series]
            Model name -> forecast Series (aligned future index).
        fig : matplotlib.figure.Figure
            Overlay plot for visual comparison.
    """
    # 1) validate horizon
    h = validate_horizon(y_test, horizon)

    # 2) aligned forecasts
    last_ts = pd.Timestamp(y_train.index[-1])
    forecasts = generate_forecasts(models=models, horizon=h, last_ts=last_ts, freq=freq)

    # 3) metrics on the SAME overlapping window
    metrics_df = compute_metrics_table(y_true=y_test, forecasts=forecasts)

    # 4) overlay plot
    fig = plot_overlay(y_train=y_train, y_test=y_test, forecasts=forecasts, tail=200)

    return metrics_df, forecasts, fig