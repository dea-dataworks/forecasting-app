Phase 1 — Data Input & Validation

- Implemented core functions in data_input.py:

        -load_csv() → safe CSV reader with basic validation.
        -detect_datetime() → detect/parse datetime column, set as index, enforce uniqueness + sorting.
        -validate_frequency() → infer time series frequency, check monotonicity, report gaps.
        -regularize_and_fill() → optional resampling to fixed grid with fill strategies (ffill, interpolate, none).
        -summarize_dataset() → structured summary (rows, cols, start/end, freq, gaps, top missing).

- Guardrails:
        -Reject empty files, duplicate columns, duplicate timestamps.
        -Fail loud on unparseable datetimes (no silent drops).
        
- Output: clear, dictionary-style summaries ready for display in Streamlit.

Phase 2 — EDA (Exploration)

- Implemented core functions in eda.py:

        - plot_raw_series() → line plot of raw time series with downsampling guard.
        - plot_rolling() → rolling mean (windowed smoothing) with optional variance/std overlay.
        - basic_stats() → quick numeric summary (min, max, mean).

- Guardrails:
        -Require DatetimeIndex, single numeric column.
        -Downsample large series for plotting (>5k points).
        -Error if rolling window > series length, or if series has no valid values.

- Output: figures and dictionary-style stats ready for Streamlit display.

Phase 3: Baselines

- Added baseline model functions in models.py:

                -train_test_split_ts() → chronological train/test split with guardrails.
                -naive_forecast() → repeat last observed value over test horizon.
                -moving_average_forecast() → repeat mean of trailing window values.
                -evaluate_forecast() → compute MAE, RMSE, MAPE with safe handling.
                -run_baseline_suite() → convenience wrapper to run/compare baselines.
                -format_baseline_report() → optional tidy DataFrame for display.
- Guardrails: monotonic index enforced, no leakage, safe handling of NAs and zero-division in MAPE.
- Output: dictionary of predictions + metrics, with optional report table.