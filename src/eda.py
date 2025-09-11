import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_raw_series(df):
    """
    Plot the raw time series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex and one numeric column.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Line plot of the series.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    if df.shape[1] != 1:
        raise ValueError("DataFrame must have exactly one value column")

    # Downsample for plotting if needed
    max_points = 5000
    if len(df) > max_points:
        step = len(df) // max_points
        df_plot = df.iloc[::step]
    else:
        df_plot = df

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_plot.index, df_plot.iloc[:, 0], label=df.columns[0])
    ax.set_title("Raw Time Series")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    return fig


def plot_rolling(df, window: int = 7, show_var: bool = False):
    """
    Plot raw series with rolling mean (and optional rolling variance).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex and exactly one numeric column.
    window : int, default=7
        Rolling window size (in samples; assumes regular spacing from Phase 1).
    show_var : bool, default=False
        If True, add rolling variance on a secondary y-axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # --- basic checks ---
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    if df.shape[1] != 1:
        raise ValueError("DataFrame must have exactly one value column")
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    if len(df) < window:
        raise ValueError(f"Series length ({len(df)}) is smaller than window ({window}).")

    # --- compute rolling stats (on full data) ---
    col = df.columns[0]
    roll_mean = df[col].rolling(window=window, min_periods=window).mean()
    if show_var:
        roll_var = df[col].rolling(window=window, min_periods=window).var(ddof=0)

    # --- downsample for plotting if needed ---
    max_points = 5000
    def downsample(s):
        if len(s) <= max_points:
            return s
        step = len(s) // max_points
        return s.iloc[::step]

    raw_plot = downsample(df[col])
    mean_plot = downsample(roll_mean)
    if show_var:
        var_plot = downsample(roll_var)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(raw_plot.index, raw_plot.values, label=col, alpha=0.6)
    ax.plot(mean_plot.index, mean_plot.values, label=f"Rolling mean (w={window})", linewidth=2)

    ax.set_title("Rolling View")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")

    if show_var:
        ax2 = ax.twinx()
        ax2.plot(var_plot.index, var_plot.values, label="Rolling variance", linestyle="--", alpha=0.7)
        ax2.set_ylabel("Variance")
        # Build a combined legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.tight_layout()
    return fig

def basic_stats(df):
    """
    Compute basic stats for a single-column time series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex and exactly one numeric column.

    Returns
    -------
    dict
        {"min": float, "max": float, "mean": float}
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    if df.shape[1] != 1:
        raise ValueError("DataFrame must have exactly one value column")

    s = df.iloc[:, 0]
    s_nonnull = s.dropna()
    if s_nonnull.empty:
        raise ValueError("Series has no non-NaN values to summarize")

    return {
        "min": float(s_nonnull.min()),
        "max": float(s_nonnull.max()),
        "mean": float(s_nonnull.mean()),
    }

def plot_decomposition(df, period: int | None = None, model: str = "additive"):
    """
    Plot seasonal-trend decomposition (Trend, Seasonal, Residual) as 3 stacked subplots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex and exactly one numeric column.
    period : int, optional
        Seasonal period. If None, try to infer from index frequency (D->7, W->52, M/MS->12, Q/QS->4, H->24, B->7).
        If it cannot be inferred, a ValueError is raised with guidance.
    model : {"additive", "multiplicative"}, default "additive"
        Decomposition model.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If input is invalid, series too short (< 2*period), cannot infer period, or decomposition fails.
    """
    # --- basic checks ---
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    if df.shape[1] != 1:
        raise ValueError("DataFrame must have exactly one value column")
    if model not in {"additive", "multiplicative"}:
        raise ValueError("model must be 'additive' or 'multiplicative'")

    s = df.iloc[:, 0].dropna()
    n = len(s)
    if n < 3:
        raise ValueError(f"Series too short ({n} points) for decomposition.")

    # --- infer period if not provided ---
    def _guess_period(idx: pd.DatetimeIndex) -> int | None:
        alias = pd.infer_freq(idx)
        if not alias:
            return None
        alias = alias.upper()
        # Map common aliases
        if alias in {"D", "B"}:
            return 7
        if alias == "W":
            return 52
        if alias in {"M", "MS"}:
            return 12
        if alias in {"Q", "QS"}:
            return 4
        if alias == "H":
            return 24
        return None

    if period is None:
        period = _guess_period(s.index)

    if period is None:
        raise ValueError(
            "Could not infer a seasonal period from the index. "
            "Try regularizing to a known frequency (e.g., Daily/Monthly) or pass a period explicitly."
        )

    # Require at least two full seasons for a stable result
    if n < 2 * period:
        raise ValueError(
            f"Series too short for decomposition: need at least 2×period = {2*period} points, have {n}."
        )

    # --- decomposition ---
    try:
        res = seasonal_decompose(s, model=model, period=period, two_sided=True, extrapolate_trend="freq")
    except Exception as e:
        raise ValueError(f"Decomposition failed: {type(e).__name__}: {e}") from e

    # --- plot (3 stacked subplots) ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(res.trend, label="Trend")
    axes[0].set_title("Trend")
    axes[0].legend(loc="upper left")

    axes[1].plot(res.seasonal, label="Seasonal")
    axes[1].set_title("Seasonal")
    axes[1].legend(loc="upper left")

    axes[2].plot(res.resid, label="Residual (noise)")
    axes[2].set_title("Residual (noise)")
    axes[2].legend(loc="upper left")

    axes[2].set_xlabel("Time")
    fig.tight_layout()
    return fig

def _auto_nlags(n: int) -> int:
    """
    Choose a safe, readable nlags: min(30, 10% of length), but < n-1.
    Ensures at least 1 lag.
    """
    if n <= 3:
        return 1
    nl = max(1, int(n * 0.10))
    nl = min(nl, 30)
    # statsmodels requires nlags < n - 1
    return max(1, min(nl, n - 2))


def plot_acf_series(df: pd.DataFrame, max_lags: int | None = None):
    """
    Plot ACF for a single-column time series with CI bands and safe lag clipping.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    if df.shape[1] != 1:
        raise ValueError("DataFrame must have exactly one value column")

    s = df.iloc[:, 0].dropna()
    n = len(s)
    if n < 3:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, f"ACF unavailable: series too short ({n} points).", ha="center", va="center")
        fig.tight_layout()
        return fig

    nlags = _auto_nlags(n) if max_lags is None else max(1, min(max_lags, n - 2))
    try:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        plot_acf(s, lags=nlags, ax=ax, zero=False)
        ax.set_title(f"ACF (nlags={nlags})")
        ax.set_xlabel("Lag")
        fig.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, f"ACF unavailable: {type(e).__name__}: {e}", ha="center", va="center")
        fig.tight_layout()
        return fig


def plot_pacf_series(df: pd.DataFrame, max_lags: int | None = None):
    """
    Plot PACF for a single-column time series with CI bands and safe lag clipping.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    if df.shape[1] != 1:
        raise ValueError("DataFrame must have exactly one value column")

    s = df.iloc[:, 0].dropna()
    n = len(s)
    if n < 3:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, f"PACF unavailable: series too short ({n} points).", ha="center", va="center")
        fig.tight_layout()
        return fig

    nlags = _auto_nlags(n) if max_lags is None else max(1, min(max_lags, n - 2))
    try:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        # yule-walker MLE is generally stable for short samples
        plot_pacf(s, lags=nlags, ax=ax, method="ywmle", zero=False)
        ax.set_title(f"PACF (nlags={nlags})")
        ax.set_xlabel("Lag")
        fig.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, f"PACF unavailable: {type(e).__name__}: {e}", ha="center", va="center")
        fig.tight_layout()
        return fig
