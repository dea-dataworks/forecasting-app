from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd
import io
from matplotlib.figure import Figure
from datetime import datetime

ArrayLike = Union[pd.Series, Sequence[float], np.ndarray]

def build_forecast_table(
    index: pd.DatetimeIndex,
    y_pred: ArrayLike,
    lower_ci: Optional[ArrayLike] = None,
    upper_ci: Optional[ArrayLike] = None,
) -> pd.DataFrame:
    """
    Build a tidy forecast table with optional confidence intervals.

    Output columns:
      - 'date' (from DatetimeIndex)
      - 'forecast' (point predictions)
      - 'lower' (optional)
      - 'upper' (optional)

    Guardrails:
      - Enforces DatetimeIndex.
      - Aligns/validates lengths; drops CI columns if misaligned.
    """
    # --- helpers (local to avoid external deps) ---
    def _to_1d_series(x: ArrayLike, name: str, n: int) -> pd.Series:
        if isinstance(x, pd.Series):
            s = x.reset_index(drop=True)
        elif isinstance(x, np.ndarray):
            s = pd.Series(x.ravel())
        else:
            s = pd.Series(list(x) if x is not None else [])
        if len(s) != n:
            raise ValueError(f"{name} length {len(s)} != expected length {n}")
        return s

    # --- inputs & basic checks ---
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas.DatetimeIndex")
    n = len(index)

    yhat = _to_1d_series(y_pred, "y_pred", n)

    # Try to coerce CIs; if lengths mismatch, we skip them gracefully.
    lower_s = upper_s = None
    if lower_ci is not None:
        try:
            lower_s = _to_1d_series(lower_ci, "lower_ci", n)
        except Exception:
            lower_s = None
    if upper_ci is not None:
        try:
            upper_s = _to_1d_series(upper_ci, "upper_ci", n)
        except Exception:
            upper_s = None

    # --- build frame ---
    df = pd.DataFrame(
        {
            "date": pd.Index(index, name="date"),
            "forecast": yhat.astype(float),
        }
    )
    if lower_s is not None:
        df["lower"] = lower_s.astype(float)
    if upper_s is not None:
        df["upper"] = upper_s.astype(float)

    # Column order: date, forecast, (lower), (upper)
    cols = ["date", "forecast"]
    if "lower" in df.columns:
        cols.append("lower")
    if "upper" in df.columns:
        cols.append("upper")
    return df[cols]

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame into UTF-8 encoded CSV bytes.
    Keeps 'date' as a column, not index.
    """
    # Reset index just in case user passed a DF with date as index
    if df.index.name is not None or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(drop=True)

    csv_str = df.to_csv(index=False)
    return csv_str.encode("utf-8")

def figure_to_png_bytes(fig, dpi: int = 120) -> bytes:
    """
    Serialize a Matplotlib Figure to PNG bytes.

    - Uses tight bounding box to avoid excess padding.
    - Preserves figure facecolor so dark/light themes look right in the PNG.
    """
    import io

    if fig is None:
        raise ValueError("figure_to_png_bytes: 'fig' is None")

    if hasattr(fig, "canvas") and hasattr(fig.canvas, "draw"):
        fig.canvas.draw()

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    buf.seek(0)
    return buf.getvalue()


def make_default_filenames(base: str = "forecast") -> dict:
    """
    Return timestamped filenames for CSV and PNG as a dict:
      {"csv": "<base>_YYYYMMDD-HHMMSS.csv", "png": "<base>_YYYYMMDD-HHMMSS.png"}

    `base` should be short and safe (no spaces); caller can inject target/horizon.
    """
    import re
    from datetime import datetime

    # slugify `base`
    s = str(base).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]", "", s) or "export"

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    stem = f"{s}_{ts}"
    return {
        "csv": f"{stem}.csv",
        "png": f"{stem}.png",
    }



