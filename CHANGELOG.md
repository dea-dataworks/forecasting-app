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

Phase 6: Outputs

- Added new module outputs.py with export utilities:
        -build_forecast_table() → assemble forecast + CIs into tidy DataFrame.
        -dataframe_to_csv_bytes() → convert DataFrame to UTF-8 CSV bytes.
        -figure_to_png_bytes() → serialize Matplotlib Figure as PNG bytes.
        -make_default_filenames() → generate timestamped names for CSV/PNG.
- Guardrails: enforce datetime index, check array lengths, ensure UTF-8 encoding, avoid temp files.
- Output: forecast values (CSV) and plots (PNG) ready for download in Streamlit.

Phase 7: Streamlit Shell & Wiring

- Added app.py with Streamlit scaffolding:
        -sidebar_nav() → basic sidebar navigation (Data, EDA, Models, Compare).
        -load_sample_button() → button to load sample data into session state.
        -render_data_page() / render_placeholder_page() → page placeholders with “coming soon” messages.
        -import_smoke_test() → check that core modules import without errors.
        -main() → orchestrator: sets page config, initializes state keys, runs sidebar, routes pages.
- Guardrails: no backend logic wired yet, imports tested to avoid runtime errors, safe handling if no data loaded.
- Output: functional Streamlit shell with navigation and placeholders, ready for Phase 9 integration.

Phase 8: Streamlit Styling & Density Toggle

- Enhanced UI styling in src/ui.css:

        -Integrated Inter font (with fallbacks).
        -Standardized padding, table row height, and spacing.
        -Added brand color constant in src/theme.py for consistent plots.
- Added compact/expanded view toggle:
        -Sidebar radio to switch density.
        -_inject_density_css() helper injects CSS + body class.
        -get_density_cfg() (planned) or inline dict supplies plot/table sizing tokens.
- Guardrails: 
        -Session state persists density choice across tabs.
        -CSS injection fails safe with warning (no crash if file missing).
        -Transparent plot backgrounds ensure Light/Dark mode compatibility.
- Output: consistent font, colors, and density-aware plots/tables across all app pages..

Phase 9: Integration & Full App

- Added full Streamlit front end in app.py, wired to backend modules.
- Implemented app shell:
        - Page setup with `st.set_page_config`.
        - Sidebar navigation across Data, EDA, Models, and Compare pages.
        - Session bootstrapping for df, train, test, freq, summary.
        - CSS injection with density toggle (compact/expanded).
- Data page:
        - CSV upload with `load_csv()` and datetime detection via `detect_datetime()`.
        - Frequency report with `validate_frequency()`.
        - Optional regularization with `regularize_and_fill()`.
        - Target column selection + train/test split with `train_test_split_ts()`.
        - Dataset summary using `summarize_dataset()` and preview table.
- EDA page:
        - Raw series plot (`plot_raw_series()`).
        - Rolling mean/variance plot (`plot_rolling()`).
        - Basic stats (`basic_stats()`).
- Models page:
        - Baseline models (naïve, moving average) via `run_baseline_suite()`.
        - Metrics table (`format_baseline_report()`).
        - Overlay chart with train/test and forecasts.
        - Export options: forecast CSV + overlay PNG.
- Compare page:
        - Horizon slider with validation (`validate_horizon()`).
        - Forecast alignment (`make_future_index()`, `generate_forecasts()`).
        - Leaderboard metrics (`compute_metrics_table()`).
        - Overlay plot (`plot_overlay()`).
        - Export options: per-model CSV + overlay PNG.
- Guardrails:
        - Errors surfaced as warnings, no app crashes.
        - Horizon clipped to test size.
        - Graceful skip if no models or data available.
- Output: working end-to-end forecasting app (baseline models) with uploads, EDA, modeling, comparison, and exports.


