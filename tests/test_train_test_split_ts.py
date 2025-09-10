import pandas as pd
from src.baselines import train_test_split_ts  # (df, target_col, test_size)

def test_split_sizes_and_order_with_fraction():
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({"y": range(100)}, index=idx)
    y_train, y_test = train_test_split_ts(df, "y", 0.2)  # positional
    assert len(y_train) == 80 and len(y_test) == 20
    assert y_train.index.max() < y_test.index.min()
    # Recombine to original series
    recombined = pd.concat([y_train, y_test])
    assert recombined.equals(df["y"])

def test_split_with_integer_count():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"y": range(30)}, index=idx)
    y_train, y_test = train_test_split_ts(df, "y", 7)  # positional
    assert len(y_train) == 23 and len(y_test) == 7
    assert y_train.index.max() < y_test.index.min()

