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
    Create a tidy forecast table with aligned dates and values.

    Columns:
      - date (datetime64)
      - forecast (float)
      - lower_ci (float, optional)
      - upper_ci (float, optional)

    Requirements:
      - `index` must be a DatetimeIndex
      - All provided arrays must match len(index)
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas DatetimeIndex")

    n = len(index)

    def _to_1d(a: ArrayLike, name: str) -> np.ndarray:
        arr = a.values if isinstance(a, pd.Series) else np.asarray(a)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-D")
        if len(arr) != n:
            raise ValueError(f"{name} length ({len(arr)}) must match index length ({n})")
        return arr.astype(float)

    y_arr = _to_1d(y_pred, "y_pred")
    df = pd.DataFrame({
        "date": pd.to_datetime(index, utc=False),  # keep tz if present
        "forecast": y_arr,
    })

    if lower_ci is not None:
        df["lower_ci"] = _to_1d(lower_ci, "lower_ci")
    if upper_ci is not None:
        df["upper_ci"] = _to_1d(upper_ci, "upper_ci")

    # Optional: ensure column order if both CIs present
    cols = ["date", "forecast"] + [c for c in ["lower_ci", "upper_ci"] if c in df.columns]
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

def figure_to_png_bytes(fig: Figure, dpi: int = 120) -> bytes:
    """
    Convert a Matplotlib Figure into PNG bytes for download.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def make_default_filenames(base: str = "forecast") -> dict:
    """
    Build timestamped filenames for CSV and PNG exports.
    Example: {'csv': 'forecast_20250909_101530.csv', 'png': 'forecast_20250909_101530.png'}
    """
    # Avoid spaces or funky chars in filenames
    safe_base = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base.strip())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "csv": f"{safe_base}_{ts}.csv",
        "png": f"{safe_base}_{ts}.png",
    }

