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

Phase 4: Classical Models

- Added new module classical.py with forecasting functions:

        -train_auto_arima() → fit auto-ARIMA with optional seasonality.
        -forecast_auto_arima() → generate forecasts + confidence intervals aligned to test horizon.
        -train_prophet() → fit Prophet model (if installed) with trend + seasonalities.
        -forecast_prophet() → produce Prophet forecasts + uncertainty intervals.
        -plot_forecast_with_ci() → unified line plot with shaded confidence bands.
- Guardrails: optional imports (skip cleanly if library missing), no train/test leakage, validated seasonal period, consistent Series outputs indexed to test set.
- Output: forecast series with confidence intervals and standardized plots for comparison.

Phase 5: Comparison Dashboard

- Added new module compare.py with comparison utilities:
        -validate_horizon() → ensure requested horizon fits test set.  
        -make_future_index() → build consistent future DatetimeIndex.  
        -forecast_with_adapter() → normalize outputs from ARIMA, Prophet, baselines.  
        -generate_forecasts() → batch forecasts for all models with safe skipping.  
        -compute_metrics_table() → side-by-side MAE/RMSE/MAPE leaderboard.  
        -plot_overlay() → overlay plot of train tail, test, and all forecasts.  
        -comparison_run() → orchestrator: horizon check → forecasts → metrics → plot.
- Guardrails: horizon clipping, safe NaN handling, graceful skip of failing/None models, standardized float Series with aligned indexes.
- Output: model leaderboard table and overlay chart for direct performance comparison in the dashboard.

