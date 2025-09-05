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
