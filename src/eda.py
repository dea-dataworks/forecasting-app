import matplotlib.pyplot as plt
import pandas as pd

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
