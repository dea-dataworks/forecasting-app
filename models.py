import pandas as pd
import numpy as np

# --- Baselines ---

def train_test_split_ts(
    df: pd.DataFrame,
    target_col: str,
    test_size: int | float
) -> tuple[pd.Series, pd.Series]:
    """
    Split a time series into train/test sets without shuffling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by datetime, containing the target column.
    target_col : str
        Name of the column to forecast.
    test_size : int or float
        If int: number of test observations.
        If float (0 < test_size < 1): fraction of total length to use as test.

    Returns
    -------
    (y_train, y_test) : tuple of pd.Series
        Train and test target series.
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Datetime index must be sorted in increasing order.")

    n = len(df)

    # Resolve test_size
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("Float test_size must be between 0 and 1.")
        n_test = int(round(n * test_size))
    elif isinstance(test_size, int):
        n_test = test_size
    else:
        raise TypeError("test_size must be int or float.")

    if n_test <= 0 or n_test >= n:
        raise ValueError("Invalid test_size: must leave at least 1 point in both train and test.")

    # Slice
    y = df[target_col]
    y_train = y.iloc[: n - n_test]
    y_test = y.iloc[n - n_test :]

    return y_train, y_test

def naive_forecast(y_train: pd.Series, horizon: int) -> pd.Series:
    """
    Repeat the last observed value for 'horizon' future steps.

    Assumes y_train has a DatetimeIndex or PeriodIndex with an inferable frequency
    (ensured by Phase 1 regularization/validation).

    Parameters
    ----------
    y_train : pd.Series
        Training target series, indexed by datetime/period.
    horizon : int
        Number of future steps to forecast (typically len(y_test)).

    Returns
    -------
    pd.Series
        Forecast values indexed over the next 'horizon' steps.
    """
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    if y_train.empty:
        raise ValueError("y_train cannot be empty.")

    last_val = y_train.iloc[-1]
    idx = y_train.index

    # Build future index
    if isinstance(idx, pd.DatetimeIndex):
        freq = idx.freq or pd.infer_freq(idx)
        if freq is None:
            raise ValueError("Cannot infer frequency from DatetimeIndex. Ensure a regular frequency (see Phase 1).")
        start = idx[-1] + pd.tseries.frequencies.to_offset(freq)
        future_index = pd.date_range(start=start, periods=horizon, freq=freq)
    elif isinstance(idx, pd.PeriodIndex):
        if idx.freq is None:
            raise ValueError("Cannot infer frequency from PeriodIndex. Ensure a regular frequency (see Phase 1).")
        future_index = pd.period_range(start=idx[-1] + 1, periods=horizon, freq=idx.freq)
    else:
        raise TypeError("Index must be DatetimeIndex or PeriodIndex with a regular frequency.")

    return pd.Series([last_val] * horizon, index=future_index, name="naive")

def moving_average_forecast(
    y_train: pd.Series,
    horizon: int,
    window: int = 7
) -> pd.Series:
    """
    Forecast all future steps with the trailing moving-average of the last `window` points.

    Parameters
    ----------
    y_train : pd.Series
        Training target series with a DatetimeIndex or PeriodIndex (regular frequency).
    horizon : int
        Number of steps to forecast (typically len(y_test)).
    window : int, default 7
        Size of the trailing window used to compute the mean.

    Returns
    -------
    pd.Series
        Forecast values indexed over the next `horizon` steps.

    Notes
    -----
    - Uses a *trailing* window (last `window` observed values).
    - Repeats that single mean across the horizon (no peeking/rolling updates).
    """
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    if y_train.empty:
        raise ValueError("y_train cannot be empty.")
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    if len(y_train) < window:
        raise ValueError(f"y_train has only {len(y_train)} points; needs at least `window={window}`.")

    # Compute trailing mean from the last `window` observed values
    last_window = y_train.iloc[-window:]
    mean_val = last_window.mean()

    idx = y_train.index
    if isinstance(idx, pd.DatetimeIndex):
        freq = idx.freq or pd.infer_freq(idx)
        if freq is None:
            raise ValueError("Cannot infer frequency from DatetimeIndex. Ensure a regular frequency (see Phase 1).")
        start = idx[-1] + pd.tseries.frequencies.to_offset(freq)
        future_index = pd.date_range(start=start, periods=horizon, freq=freq)
    elif isinstance(idx, pd.PeriodIndex):
        if idx.freq is None:
            raise ValueError("Cannot infer frequency from PeriodIndex. Ensure a regular frequency (see Phase 1).")
        future_index = pd.period_range(start=idx[-1] + 1, periods=horizon, freq=idx.freq)
    else:
        raise TypeError("Index must be DatetimeIndex or PeriodIndex with a regular frequency.")

    return pd.Series([mean_val] * horizon, index=future_index, name=f"movavg_w{window}")


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Compute MAE, RMSE, and MAPE safely for aligned Series.

    - Drops index elements where either side is NA.
    - Requires same index (order + labels) to avoid accidental misalignment.
    - Uses epsilon in MAPE denominator to avoid div-by-zero explosions.

    Returns a plain dict with float metrics.
    """
    if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series):
        raise TypeError("y_true and y_pred must be pandas Series.")

    if not y_true.index.equals(y_pred.index):
        raise ValueError("y_true and y_pred must have the SAME index (labels and order).")

    df = pd.concat({"y_true": y_true, "y_pred": y_pred}, axis=1).dropna()
    if df.empty:
        raise ValueError("After dropping NAs, there are no overlapping observations to evaluate.")

    err = df["y_pred"] - df["y_true"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # Safe MAPE: add small epsilon to denominator; express in %
    eps = 1e-8
    mape = float(np.mean(np.abs(err) / (np.abs(df["y_true"]) + eps)) * 100.0)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def run_baseline_suite(
    y_train: pd.Series,
    y_test: pd.Series,
    window: int = 7
) -> dict:
    """
    Run baseline models (naïve + moving average) against a test set.

    Parameters
    ----------
    y_train : pd.Series
        Training portion of the time series.
    y_test : pd.Series
        Test portion (ground truth).
    window : int, default 7
        Window size for the moving-average baseline.

    Returns
    -------
    dict
        {
          "naive": {
              "y_pred": pd.Series,   # forecast aligned to y_test.index
              "metrics": {...}       # MAE/RMSE/MAPE
          },
          "movavg": {
              "y_pred": pd.Series,
              "metrics": {...},
              "window": int
          }
        }
    """
    horizon = len(y_test)

    # Naïve forecast
    naive_pred = naive_forecast(y_train, horizon)
    naive_pred.index = y_test.index  # align to test horizon
    naive_metrics = evaluate_forecast(y_test, naive_pred)

    # Moving average forecast
    movavg_pred = moving_average_forecast(y_train, horizon, window)
    movavg_pred.index = y_test.index
    movavg_metrics = evaluate_forecast(y_test, movavg_pred)

    return {
        "naive": {
            "y_pred": naive_pred,
            "metrics": naive_metrics,
        },
        "movavg": {
            "y_pred": movavg_pred,
            "metrics": movavg_metrics,
            "window": window,
        },
    }

def format_baseline_report(results: dict) -> pd.DataFrame:
    """
    Turn baseline results dict into a tidy metrics DataFrame.

    Parameters
    ----------
    results : dict
        Output of run_baseline_suite().

    Returns
    -------
    pd.DataFrame
        Rows = model name, Cols = MAE/RMSE/MAPE.
    """
    rows = []
    for model_name, res in results.items():
        row = {"model": model_name}
        row.update(res["metrics"])
        if "window" in res:
            row["window"] = res["window"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    return df
