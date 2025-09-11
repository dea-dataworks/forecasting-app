v0.2

Phase 12: Exports

- Enhanced build_forecast_table() to include lower/upper columns when provided.
- Confirmed figure_to_png_bytes() integration with overlay plots.git l
- Standardized make_default_filenames() for timestamped CSV/PNG outputs.

Phase 13: Metrics & Compare polish

src/compare.py
        Upgraded compute_metrics_table():
        Added NaN/zero-denominator safety across metrics.
        Optional metrics: sMAPE and MASE (with naive one-step denominator).
        Configurable sort (sort_by, default RMSE, NaNs sink last).
app.py
        Compare page:
                Display H = … steps above leaderboard.
                Added checkboxes for sMAPE/MASE and a Sort by selector.
                Metrics recompute automatically when H or toggles change.
                Output: Stable, sortable leaderboard (MAE/RMSE/MAPE; optional sMAPE/MASE) clearly tied to the current horizon.


Phase 14: Plots

Extended plot_overlay() in compare.py:
        Added optional lower/upper dicts for confidence interval shading.
        Added ci_model parameter to select one model for CI display.
        Added density mode (compact / expanded) for consistent sizing and labels.
        Downsampling guard: always use train tail instead of full history.
Updated app.py Models & Compare pages:
        Wired density toggle from session state into overlay plots.
        Added CI model selector where intervals are available.
Output: Unified overlay chart with optional CI bands, readable in both modes.

Phase 15: Streamlit Polish

Updated comparison_run in compare.py:
        Added density parameter and passed through to plot_overlay for consistent compact/expanded rendering.
        Corrected argument order in validate_horizon call.
Updated app.py to forward sidebar density setting into comparison_run.
Removed redundant manual rerun on density toggle.
Guardrails: no deprecation warnings, stable single-click density switch, overlay plots persist after toggling.

Phase 16.1: Frequency & gaps (rename + clarify)

Added to_human_freq() helper to show plain-English labels (e.g., Daily, Weekly).
Renamed metrics:
        “Frequency” → Sampling frequency
        “Gaps” → Missing timestamps (index gaps)
        “Expected pts” → Expected timestamps
        “Gap ratio” → % of missing timestamps
Moved and expanded glossary to standalone “ⓘ Definitions” panel below metrics.
Added inline help= text for frequency override and fill method widgets.

Phase 16.2: Regularize to fixed frequency

Replaced manual text input with dropdown of common frequencies (Daily, Weekly, Monthly, Quarterly, Yearly, Hourly).
Added Advanced… option to enter custom pandas alias.
Added checkbox (default off) to toggle regularization; shows gentle hint if gaps are detected.
Improved Fill method dropdown with plain-language labels and descriptions.
Aligned Frequency + Fill controls in the same row for cleaner UI.
Status line now explicitly shows: Selected target: <col> | dtype | missing %.

Phase 16.3: Select target column (promote & explain)

Renamed selector to “Target (Y) column” and added help text clarifying its role.
Added a compact status line showing dtype and % missing for the chosen column.
Kept Target section above Train/Test to emphasize importance.
Status line now explicitly shows: Selected target: <col> | dtype | missing %.

Phase 16.4: Train/Test split (plain-English & dynamic feedback)

Reworked Train/Test split controls with radio for Percentage or Count.
Added plain-English help text clarifying each option.
Slider now interprets percentage directly, with live caption: “H = … rows (≈ …%)”.
Added one-liner note: “last H points, no shuffling, simulates forecasting the future.”
Validation added to nudge instead of crash when H ≤ 0 or H ≥ dataset length.

Phase 16.5: Data preview (lighter by default)

Preview moved into its own collapsed expander.
Default shows first 5 rows, with dropdown toggle for 10/20.
Added “Show last 5” button inside expander for quick split sanity-check.
Reduces scrolling noise while keeping preview accessible when needed.

Phase 16.6 — Unified Dataset Summary

Merged Dataset summary and Sampling frequency & gaps into a single expander.
Clean 2-column layout now shows: Rows, Cols, Start, End, Sampling freq, Missing timestamps, Expected timestamps, % missing timestamps.
Removed redundant "Top missing" info to keep the panel focused.

Phase 16 EDA page
Data page: preserved parsed DatetimeIndex when returning from EDA (no more "No parseable datetime column" error).
Sidebar density toggle: removed st.rerun() callback to avoid "no-op" warning; reruns now handled automatically.
Datetime parsing: removed deprecated infer_datetime_format=True in detect_datetime() (pandas infers automatically).

Phase 17.2 — ACF/PACF Diagnostics
Added plot_acf_series() and plot_pacf_series() in eda.py.
Integrated ACF/PACF plots into EDA tab after decomposition.
Auto‐clipped nlags (≤30 or 10% of series length) with short‐series guardrails.

Phase 17.3 – Layout polish

Merged raw series + rolling view into one main plot with optional overlay.
Wrapped decomposition into an expandable section with short explanatory text.
Wrapped ACF/PACF into a separate expander with cue text.
Updated page caption to reflect simplified flow.

Phase 18

- Added
        - `eda.py`: new helper `infer_season_length_from_freq(freq)` to guess seasonal period `m` from dataset frequency.
        - `app.py`: Confidence level slider (0.50–0.99) wired to ARIMA (`alpha = 1 - level`).
        - `classical.py`: `train_prophet(...)` now accepts `interval_width` argument; wired to the same Confidence slider.
        - `app.py`: fitted ARIMA/Prophet models now stashed in `st.session_state["models"]`.
        - `app.py`: Train tail slider on Models page to control overlay plot window.
        - Models page: CSV export now includes yhat, lower, upper, plus model + level; filenames show horizon + CI.
        - Compare page: Added combined CSV export across all models with consistent index and CI annotation.
        - Models: added Prophet seasonality checkboxes (weekly/yearly/daily) and ARIMA seasonal period (m) override (Auto/7/12/24/52). Uses shared CI slider. 
        - Added spinners with elapsed time display for ARIMA and Prophet training on Models page.
        - Fixed CSV export errors by updating `build_forecast_table` to accept `lower`/`upper` args alongside legacy `lower_ci`/`upper_ci`.
        - Reworked CSV download UI on Models page: model selector on top, aligned side-by-side buttons below.

















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
- UX polish: 
        -Density toggle on_change rerun; init missing session keys; consistent recoverable errors via st.warning (helper warn()); swapped st.dataframe(..., use_container_width=True) → width="stretch"; hid UI diagnostics.
        -Thin vertical slice: Data page end-to-end (upload → datetime → freq/gaps → optional regularize → pick target → split); EDA minimal (raw, rolling, stats); Models baselines with CSV/PNG exports; Compare overlay + leaderboard with horizon slider and friendly warnings.
        Note: ARIMA/Prophet deferred to next phase; baselines only for demo.
- Guardrails:
        - Errors surfaced as warnings, no app crashes.
        - Horizon clipped to test size.
        - Graceful skip if no models or data available.
- Output: working end-to-end forecasting app (baseline models) with uploads, EDA, modeling, comparison, and exports.

Phase 10: Classical Models (ARIMA & Prophet)

- Added/updated classical.py with optional-dep guards and model APIs:
        -train_auto_arima() → fit auto-ARIMA (seasonal optional, m inferred when sensible).
        -forecast_auto_arima() → predictions + 95% CIs aligned to y_test.index.
        -train_prophet() → fit Prophet with common seasonalities.
        -forecast_prophet() → predictions + CIs aligned to y_test.index.
- Models page integration:
        -Checkbox toggles for ARIMA and Prophet (disabled if libs not installed).
        -On run, trains on y_train only (no leakage) and merges outputs into baseline_results.
        -Metrics table and overlay include ARIMA/Prophet when enabled.
- Compare page: automatically picks up ARIMA/Prophet from baseline_results for overlay + leaderboard.
- Guardrails: optional deps safely skipped with friendly st.warning; aligned indices; safe metrics (NaNs/zeros handled); baseline-only demo remains fully functional if libs are missing.
- Output: end-to-end app now supports ARIMA/Prophet alongside baselines, ready for CI bands and further tuning next.

Phase 11: Release Prep & QA 

- Fixed deprecated Streamlit params in app.py (use_container_width → width="stretch").
- Added a tests/ with 3–4 tiny pytest checks. Used a tiny synthetic series.
- Cleaned redundant imports in compare.py.
- Added root requirements.txt with core deps; marked pmdarima/prophet as optional.
- Completed README.md with setup instructions, features, and troubleshooting.
- Removed/annotated empty modules (utils.py, ml_models.py).
- Guardrails: confirmed downloads (CSV/PNG) work, Compare page exports selected models.
- Output: repo is clean, documented, and shippable as v0.2.
