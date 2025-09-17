import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.api.types import is_datetime64_any_dtype

def load_csv(path_or_buffer, encoding=None):
    """
    Load a CSV file safely into a DataFrame.

    Parameters
    ----------
    path_or_buffer : str or file-like
        Path to the CSV file, or buffer (e.g. from Streamlit upload).
    encoding : str, optional
        Encoding to use. If None, pandas will guess.

    Returns
    -------
    df : pd.DataFrame
        Parsed DataFrame.

    Raises
    ------
    ValueError
        If the file is empty or has duplicate column names.
    """
    df = pd.read_csv(
        path_or_buffer,
        encoding=encoding,
        low_memory=False,  # prevents dtype guessing issues
        na_values=["", "na", "n/a", "null", "None"]
    )
    if df.empty:
        raise ValueError("Uploaded CSV has no rows.")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names detected. Please fix and re-upload.")

    return df


def detect_datetime(df):
    """
    Detect a datetime column, parse it, and set it as the index.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df_out : pd.DataFrame
        Copy of the DataFrame with a DatetimeIndex set.
    chosen_col : str
        Name of the column used as the datetime index.

    Raises
    ------
    ValueError
        If no suitable datetime column is found,
        or if duplicates/parse errors occur.
    """
    candidate_names = ("ds", "date", "datetime", "timestamp", "time")

    # Step 1: already datetime dtype
    dt_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]

    # Step 2: try parsing others (if none already datetime)
    if not dt_cols:
        scores = {}
        for c in df.columns:
            if df[c].dtype == object or c.lower() in candidate_names:
                try:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    ok = parsed.notna().mean()
                    if ok > 0.95:  # at least 95% parseable
                        scores[c] = ok
                except Exception:
                    pass
        dt_cols = sorted(scores, key=scores.get, reverse=True)

    if not dt_cols:
        raise ValueError("No parseable datetime column found.")

    # Step 3: prefer friendly names if multiple
    chosen = dt_cols[0]
    for name in candidate_names:
        if name in [c.lower() for c in dt_cols]:
            chosen = [c for c in dt_cols if c.lower() == name][0]
            break

    # Step 4: parse and set index
    df_out = df.copy()
    if not is_datetime64_any_dtype(df_out[chosen]):
        df_out[chosen] = pd.to_datetime(df_out[chosen], errors="coerce")

    n_bad = int(df_out[chosen].isna().sum())
    if n_bad > 0:
        total = len(df_out)
        pct = round(n_bad / total * 100, 2)
        raise ValueError(
            f"Datetime parsing failed for column '{chosen}': {n_bad}/{total} rows invalid ({pct}%)."
        )

    df_out = df_out.set_index(chosen).sort_index()

    if df_out.index.has_duplicates:
        dup_n = df_out.index.duplicated().sum()
        raise ValueError(f"Found {dup_n} duplicate timestamps. Please fix your data.")

    return df_out, chosen

def validate_frequency(idx: pd.DatetimeIndex) -> dict:
    """
    Inspect a DatetimeIndex for monotonicity, infer frequency,
    and (if known) count missing timestamps (gaps) on the expected grid.

    Parameters
    ----------
    idx : pd.DatetimeIndex

    Returns
    -------
    report : dict
        Keys: is_monotonic, freq, gaps, expected_points, gap_ratio
    """
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Index must be a pandas DatetimeIndex.")

    # 1) Monotonic (strictly increasing is our standard)
    is_mono = idx.is_monotonic_increasing and not idx.has_duplicates

    # 2) Try native inference first
    freq = pd.infer_freq(idx)
    if freq is None:
        # Fallback: modal delta heuristic (accept if it explains ≥90% of steps)
        deltas = idx.to_series().diff().dropna()
        if not deltas.empty:
            modal = deltas.mode().iloc[0]
            coverage = (deltas == modal).mean()
            if coverage >= 0.90:
                try:
                    freq = to_offset(modal).freqstr
                except Exception:
                    freq = None

    # 3) If we have a frequency, build the full grid and count gaps
    gaps = expected = gap_ratio = None
    if freq is not None and len(idx) > 1:
        full = pd.date_range(idx.min(), idx.max(), freq=freq)
        # difference() returns those timestamps in full that are not in idx
        gaps = int(len(full.difference(idx)))
        expected = int(len(full))
        gap_ratio = float(gaps / expected) if expected > 0 else 0.0

    return {
        "is_monotonic": bool(is_mono),
        "freq": freq,
        "gaps": gaps,
        "expected_points": expected,
        "gap_ratio": gap_ratio,
    }

def regularize_and_fill(df, freq: str, fill: str = "ffill"):
    """
    Regularize a time series to a fixed frequency and fill missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex.
    freq : str
        Frequency string (e.g., 'D', 'M', 'H').
    fill : {'ffill', 'interpolate', 'none'}
        Strategy for filling missing values.

    Returns
    -------
    pd.DataFrame
        Regularized DataFrame on the specified frequency.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # Step 1: regularize the grid
    reg = df.resample(freq).asfreq()

    # Step 2: apply filling
    if fill == "ffill":
        reg = reg.ffill()
    elif fill == "interpolate":
        reg = reg.interpolate(method="time", limit_direction="both")
    elif fill == "none":
        pass  # leave NaN
    else:
        raise ValueError("fill must be one of: 'ffill', 'interpolate', 'none'.")

    return reg

def summarize_dataset(df):
    """
    Summarize a time-indexed DataFrame for quick UI display.

    Returns a dict with shape/coverage/missing info.
    Requires df.index to be a DatetimeIndex (set by detect_datetime).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex. Run detect_datetime() first.")

    report = validate_frequency(df.index)

    freq = report["freq"] or "Unknown"
    gaps = None if report["gaps"] is None else int(report["gaps"])
    expected = report["expected_points"]
    gap_ratio = None if report["gap_ratio"] is None else round(report["gap_ratio"], 4)

    top_missing = (
        df.isna().mean()
          .sort_values(ascending=False)
          .head(5)
          .round(3)
          .to_dict()
    )

    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "start": df.index.min().isoformat(),
        "end": df.index.max().isoformat(),
        "freq": freq,
        "gaps": gaps,
        "expected_points": None if expected is None else int(expected),
        "gap_ratio": gap_ratio,
        "top_missing": top_missing,
    }
