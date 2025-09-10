import pandas as pd
import pytest
from src.compare import make_future_index  # expected: (last_timestamp, steps, freq)

def test_daily_future_index_length_and_start():
    idx = pd.date_range("2024-01-01", periods=31, freq="D")
    last_ts = idx[-1]
    fut = make_future_index(last_ts, 7, "D")  # pass a scalar last timestamp + string freq
    expected = pd.date_range(last_ts + pd.Timedelta(days=1), periods=7, freq="D")
    assert len(fut) == 7
    assert fut.equals(expected)

def test_rejects_non_timestamp_input():
    with pytest.raises((TypeError, ValueError)):
        _ = make_future_index(["2024-01-31"], 5, "D")  # not a scalar timestamp
