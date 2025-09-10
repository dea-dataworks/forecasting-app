from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional
from src.data_input import (load_csv,detect_datetime,validate_frequency,regularize_and_fill,summarize_dataset,)
from src.baselines import train_test_split_ts
from src.eda import plot_raw_series, plot_rolling, basic_stats
from src.baselines import run_baseline_suite, format_baseline_report
from src.compare import (validate_horizon, make_future_index, generate_forecasts,compute_metrics_table,plot_overlay,)
from src.classical import (HAS_PMDARIMA, HAS_PROPHET,train_auto_arima, forecast_auto_arima,train_prophet, forecast_prophet,)
from src.outputs import (build_forecast_table,dataframe_to_csv_bytes,figure_to_png_bytes,make_default_filenames,)



# --- 1) Page setup ------------------------------------------------------------
def setup_page() -> None:
    st.set_page_config(
        page_title="Forecasting App",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# --- 2) Session state init ----------------------------------------------------
def init_state_keys() -> None:
    for k in ("df", "train", "test", "freq", "summary",
              "target_col","datetime_col","baseline_results","compare_cache"):
        st.session_state.setdefault(k, None)
    st.session_state.setdefault("density", "expanded")

# --- 3) Sidebar navigation ----------------------------------------------------
def sidebar_nav() -> str:
    st.sidebar.title("Navigation")
    def _on_density_change():
        st.rerun()
    
    page = st.sidebar.radio(
        "Go to",
        options=("Data", "EDA", "Models", "Compare"),
        label_visibility="collapsed",
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.radio("Density", 
                    options=["expanded", "compact"],
                    key="density", horizontal=True,
                    help="Compact = tighter padding, slightly smaller fonts & plots",
                    on_change=_on_density_change,
                    )
    return page

# --- 4) Import smoke test (no logic calls) -----------------------------------
def import_smoke_test() -> None:
    try:
        # Import ONLY to verify paths; do not call heavy functions yet.
        from src import data_input, eda, baselines, classical, compare  # noqa: F401
    except Exception as e:
        st.sidebar.warning(f"⚠️ Import check: {type(e).__name__}: {e}")

def warn(context: str, e: Exception):
    st.warning(f"{context}: {type(e).__name__}: {e}")

# --- Inject CSS + a body class based on the toggle ---------------------------
def _inject_density_css(density_value: str) -> None:
    # Load tiny stylesheet; warn if missing but don't break the app
    try:
        css_path = Path(__file__).with_name("ui.css")
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.warning(f"UI CSS not found: {e}")

    # Set a body class so CSS can target compact/expanded
    st.markdown(
        f"""
        <script>
        const clsCompact = 'density-compact';
        const clsExpanded = 'density-expanded';
        document.body.classList.remove(clsCompact, clsExpanded);
        document.body.classList.add('{ "density-compact" if density_value=="compact" else "density-expanded" }');
        </script>
        """,
        unsafe_allow_html=True,
    )

# --- UI helpers  ---
_FREQUENCY_LABELS = {
    "B": "Business daily",
    "D": "Daily",
    "W": "Weekly",
    "M": "Monthly (month-end)",
    "MS": "Monthly (month-start)",
    "Q": "Quarterly (quarter-end)",
    "QS": "Quarterly (quarter-start)",
    "A": "Annual (year-end)",
    "AS": "Annual (year-start)",
    "H": "Hourly",
    "T": "Minutely",
    "S": "Secondly",
}
def to_human_freq(alias: str | None) -> str:
    if not alias:
        return "-"
    return _FREQUENCY_LABELS.get(str(alias), str(alias))

# --- 5) Sample data loader (button) ------------------------------------------
def load_sample_button() -> None:
    st.subheader("Data")
    st.caption("Use this to render something immediately before wiring CSV upload.")
    if st.button("Load sample data", type="primary"):
        dates = pd.date_range("2022-01-01", periods=60, freq="D")
        vals = pd.Series(100 + pd.Series(range(60)).rolling(3, min_periods=1).mean().values, index=dates)
        st.session_state.df = pd.DataFrame({"value": vals})
        st.success("Sample data loaded.")

# --- 6) Page renderers --------------------------------------------------------
# --- DATA Page ---
def render_data_page() -> None:
    st.markdown("### Data")
    st.caption("Upload → detect datetime → (optional) regularize → pick target → split train/test")

    # --- A) Upload ---
    up = st.file_uploader("Upload a CSV file", type=["csv"], accept_multiple_files=False)
    if up is not None:
        try:
            df_raw = load_csv(up)  # safe parse (no dup columns / empty)  [Phase 1]
            st.success("File loaded.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return
    else:
        # fall back to whatever is already in session (e.g., sample)
        df_raw = st.session_state.get("df")

    if df_raw is None:
        st.info("Upload a CSV to continue, or use the sample button above.")
        load_sample_button()
        return

    # --- B) Detect datetime index (first mutation) ---
    try:
        df_idx, chosen_col = detect_datetime(df_raw)  # sets DatetimeIndex, sorted
    except Exception as e:
        st.error(f"Datetime detection failed: {e}")
        return
    st.session_state.df = df_idx
    st.session_state.datetime_col = chosen_col

    # --- C) Frequency report (read-only) ---
    try:
        freq_report = validate_frequency(df_idx.index)
    except Exception as e:
        st.warning(f"Frequency inference issue: {e}")
        freq_report = {"freq": None, "gaps": None, "expected_points": None, "gap_ratio": None, "is_monotonic": False}

    with st.expander("Sampling frequency & gaps", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sampling frequency", to_human_freq(freq_report.get("freq")))
        c2.metric("Missing timestamps (index gaps)", str(freq_report.get("gaps")) if freq_report.get("gaps") is not None else "—")
        c3.metric("Expected timestamps", str(freq_report.get("expected_points")) if freq_report.get("expected_points") is not None else "—")
        gr = freq_report.get("gap_ratio")
        c4.metric("% of missing timestamps", f"{gr:.2%}" if isinstance(gr, float) else "—")

        if not freq_report.get("is_monotonic", False):
            st.warning("Index is not strictly increasing or has duplicates. Fix your data if modeling fails.")
        if freq_report.get("freq") is None:
            st.info("No clear frequency detected. You can regularize below to help models.")

    with st.expander("ⓘ Definitions", expanded=False):
        st.markdown(
            "- **Timestamp**: the date/time used as the index.\n"
            "- **Sampling frequency**: Most common spacing between timestamps (e.g., Daily, Weekly).\n"
            "- **Missing timestamps**: Dates absent from the index.\n"
            "- **Expected timestamps**: Total number of timestamps between first and last date, if none were missing.\n"
            "- **% of missing timestamps**: Missing ÷ expected, as a percentage."
            )


    # with st.expander("Sampling frequency & gaps", expanded=True):
    #     left, right = st.columns([4, 1])

    #     with left:
    #         c1, c2, c3, c4 = st.columns(4)
    #         c1.metric("Sampling frequency", to_human_freq(freq_report.get("freq")))
    #         c2.metric("Missing timestamps (index gaps)", str(freq_report.get("gaps")) if freq_report.get("gaps") is not None else "—")
    #         c3.metric("Expected timestamps", str(freq_report.get("expected_points")) if freq_report.get("expected_points") is not None else "—")
    #         gr = freq_report.get("gap_ratio")
    #         c4.metric("% of missing timestamps", f"{gr:.2%}" if isinstance(gr, float) else "—")

    #     with right:
    #         with st.expander("ⓘ Definitions", expanded=False):
    #             st.markdown(
    #                 "- **Timestamp**: the date/time used as the index.\n"
    #                 "- **Sampling frequency**: Most common spacing between timestamps (e.g., Daily, Weekly).\n"
    #                 "- **Missing timestamps**: Dates absent from the index.\n"
    #                 "- **Expected timestamps**: Total number of timestamps between first and last date, if none were missing.\n"
    #                 "- **% of missing timestamps**: Missing ÷ expected, as a percentage."
    #             )

        if not freq_report.get("is_monotonic", False):
            st.warning("Index is not strictly increasing or has duplicates. Fix your data if modeling fails.")
        if freq_report.get("freq") is None:
            st.info("No clear frequency detected. You can regularize below to help models.")

    # --- D) Optional regularization (second mutation) ---
    # Checkbox (default OFF). If there are gaps, nudge the user subtly.
    do_reg = st.checkbox("Regularize to a fixed frequency", value=False)
    if (freq_report.get("gaps") or 0) > 0:
        st.caption(f"ⓘ We found {freq_report.get('gaps')} missing timestamps. Consider regularizing.")

    # When enabled, show Frequency + Fill on the same row
    freq_fill_col1, freq_fill_col2 = st.columns([2, 1])

    # Mapping for human labels → pandas alias
    _freq_menu = {
        "Daily (D)": "D",
        "Weekly (W)": "W",
        "Monthly (M)": "M",
        "Quarterly (Q)": "Q",
        "Yearly (A)": "A",
        "Hourly (H)": "H",
        "Advanced…": "ADV",
    }

    # Choose sensible default based on detected freq; fall back to Advanced…
    _detected = (freq_report.get("freq") or "").upper()
    _default_label = next((lbl for lbl, al in _freq_menu.items() if al == _detected), "Advanced…")
    freq_label = freq_fill_col1.selectbox(
        "Frequency",
        options=list(_freq_menu.keys()),
        index=list(_freq_menu.keys()).index(_default_label),
        disabled=not do_reg,
        help="Pick a common frequency. Use **Advanced…** for a custom pandas alias (e.g., MS, QS, 15T).",
    )

    # If Advanced…, expose a tiny text input right below (still in the left column)
    custom_alias = ""
    if do_reg and _freq_menu[freq_label] == "ADV":
        custom_alias = freq_fill_col1.text_input(
            "Custom alias",
            value=_detected if _detected not in _freq_menu.values() else "",
            placeholder="e.g., MS, QS, 15T",
            help="Pandas offset alias. Examples: MS (month-start), QS (quarter-start), 15T (15 minutes).",
        ).strip()

    # Fill method with friendly labels → internal values
    _fill_map = {
        "Forward fill — carry last known value forward.": "ffill",
        "Interpolate (linear) — estimate between neighbors.": "interpolate",
        "Leave as NaN — keep blanks.": "none",
    }
    fill_label = freq_fill_col2.selectbox(
        "Fill method",
        options=list(_fill_map.keys()),
        index=0,
        disabled=not do_reg,
    )
    fill = _fill_map[fill_label]

    # Apply regularization if requested
    if do_reg:
        # Resolve final alias: menu pick or advanced text
        chosen_alias = _freq_menu[freq_label]
        if chosen_alias == "ADV":
            if not custom_alias:
                st.warning("Enter a custom frequency alias (e.g., MS, QS, 15T).")
                st.stop()
            chosen_alias = custom_alias

        try:
            df_idx = regularize_and_fill(df_idx, chosen_alias, fill=fill)
            st.session_state.df = df_idx
            # Refresh the frequency report after regularization
            freq_report = validate_frequency(df_idx.index)
        except Exception as e:
            st.error(f"Regularization failed: {e}")
            st.stop()


    # --- E) Target selection ---
    # Promote & explain: this choice drives the rest of the app.
    num_cols = [c for c in df_idx.columns if pd.api.types.is_numeric_dtype(df_idx[c])]
    if not num_cols:
        st.error("No numeric columns found. Please upload data with at least one numeric value column.")
        st.stop()

    target_col = st.selectbox(
        "Target (Y) column",
        options=num_cols,
        index=0,
        help="This is the series we’ll forecast. Other columns remain available for reference."
    )
    st.session_state.target_col = target_col

    # Tiny status line under the selector
    _col_dtype = str(df_idx[target_col].dtype)
    _missing_pct = float(df_idx[target_col].isna().mean() * 100.0)
    st.caption(f"dtype: {_col_dtype} • missing: {_missing_pct:.1f}%")


    # --- F) Train/Test split ---
    st.subheader("Train/Test split")
    split_type = st.radio("Split type", options=["fraction", "count"], horizontal=True)
    if split_type == "fraction":
        frac = st.slider("Test size (fraction of rows)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
        test_size = float(frac)
    else:
        max_count = max(1, min(365, len(df_idx) // 2))
        cnt = st.slider("Test size (last N rows)", min_value=1, max_value=max_count, value=min(30, max_count))
        test_size = int(cnt)

    try:
        y_train, y_test = train_test_split_ts(df_idx, target_col=target_col, test_size=test_size)
        st.session_state.train = y_train
        st.session_state.test = y_test
        st.session_state.freq = freq_report.get("freq")
    except Exception as e:
        st.error(f"Split failed: {e}")
        return

    # --- G) Summary + preview ---
    try:
        summary = summarize_dataset(df_idx)
        st.session_state.summary = summary
    except Exception as e:
        summary = None
        st.warning(f"Summary unavailable: {e}")

    with st.expander("📋 Dataset summary", expanded=True):
        if summary:
            left, right = st.columns(2)
            left.write(f"**Rows:** {summary['rows']:,}  \n**Cols:** {summary['cols']}  \n**Start:** {summary['start']}  \n**End:** {summary['end']}")
            right.write(
                    f"**Sampling freq:** {to_human_freq(summary['freq']) if summary['freq'] else '—'}  "
                    f"\n**Missing timestamps:** {summary['gaps']}  "
                    f"\n**% missing timestamps:** {summary['gap_ratio']}  "
                    f"\n**Top missing:** {summary['top_missing']}"
                )
        st.dataframe(df_idx.head(10), width="stretch")
        st.caption(f"Target: `{target_col}` · Train: {len(y_train):,} · Test: {len(y_test):,}")

    st.success("Data is ready. You can open EDA or Models.")

# --- EDA page ---
def render_eda_page() -> None:
    st.markdown("### EDA")
    st.caption("Raw plot → Rolling view → Basic stats")

    df = st.session_state.get("df")
    target_col = st.session_state.get("target_col")

    if df is None or target_col is None:
        st.info("Load data first on the **Data** page (upload, choose target, split).")
        return

    y_df = df[[target_col]]  # keep as DataFrame for our plotting helpers

    # --- 1) Raw series (downsample handled inside helper) ---
    st.subheader("Raw series")
    try:
        fig_raw = plot_raw_series(y_df)
        st.pyplot(fig_raw)
    except Exception as e:
        st.warning(f"Could not render raw plot: {e}")

    # --- 2) Rolling view ---
    st.subheader("Rolling view")
    c1, c2 = st.columns([2, 1])
    window = c1.slider("Window (points)", min_value=3, max_value=90, value=7, step=1)
    show_var = c2.checkbox("Show variance/std bands", value=False)

    try:
        fig_roll = plot_rolling(y_df, window=window, show_var=show_var)
        st.pyplot(fig_roll)
    except Exception as e:
        st.warning(f"Could not render rolling plot: {e}")

    # --- 3) Basic stats ---
    st.subheader("Basic stats")
    try:
        stats = basic_stats(y_df)
        m1, m2, m3 = st.columns(3)
        m1.metric("Min", f"{stats['min']:.3f}")
        m2.metric("Mean", f"{stats['mean']:.3f}")
        m3.metric("Max", f"{stats['max']:.3f}")
    except Exception as e:
        st.warning(f"Could not compute basic stats: {e}")

# --- MODELS Page ---
def render_models_page() -> None:
    st.markdown("### Models")
    st.caption("Baselines first (naïve & moving average). Export results for later.")

    y_train = st.session_state.get("train")
    y_test = st.session_state.get("test")
    df = st.session_state.get("df")
    target_col = st.session_state.get("target_col")

    if any(x is None for x in (y_train, y_test, df, target_col)):
        st.info("Finish the **Data** page first (upload, pick target, split).")
        return

    # --- Controls ---
    c1, c2 = st.columns([2, 1])
    window = c1.slider("Moving average window (for baseline)", 3, 60, 7, 1)
    run_button = c2.button("Run baselines", width="stretch")

    # --- Classical model toggles (visible even before first run) ---
    c3, c4 = st.columns([1, 1])
    use_arima = c3.checkbox("ARIMA (auto)", value=False, disabled=not HAS_PMDARIMA,
                            help="Auto-ARIMA via pmdarima" + ("" if HAS_PMDARIMA else " — not installed"))
    use_prophet = c4.checkbox("Prophet", value=False, disabled=not HAS_PROPHET,
                            help="Additive model with seasonality" + ("" if HAS_PROPHET else " — not installed"))

    # Early gate remains after toggles so they are visible on first visit
    if not run_button and "baseline_results" not in st.session_state:
        st.stop()

    # --- Run baselines ---
    # try:
    #     results = run_baseline_suite(y_train=y_train, y_test=y_test, window=window)
    #     st.session_state["baseline_results"] = results
    # except Exception as e:
    #     st.warning(f"Baseline run failed: {e}")
    #     return

    # results = st.session_state["baseline_results"]
    # Optional: auto-refresh if the MA window changed

    window_changed = st.session_state.get("baseline_window") != window

    # Only compute baselines when user clicks the button, first time, or window changed
    if run_button or "baseline_results" not in st.session_state or window_changed:
        try:
            results = run_baseline_suite(y_train=y_train, y_test=y_test, window=window)
            st.session_state["baseline_results"] = results
            st.session_state["baseline_window"] = window  # remember the window we used
        except Exception as e:
            st.warning(f"Baseline run failed: {e}")
            return  # avoid using undefined `results`

    # Use the cached results for tables/plots below
    results = st.session_state["baseline_results"]


    # ARIMA (safe defaults)
    if use_arima:
        try:
            freq = st.session_state.get("freq")
            # naive season length guess (optional, safe to skip)
            m = 7 if freq in ("D", "B") else (12 if freq in ("MS", "M") else None)
            seasonal = m is not None
            arima_model = train_auto_arima(y_train, seasonal=seasonal, m=m)
            yhat, lo, hi = forecast_auto_arima(arima_model, test_index=y_test.index)
            results["ARIMA"] = {"y_pred": yhat, "lower": lo, "upper": hi}
        except Exception as e:
            st.warning(f"ARIMA failed: {type(e).__name__}: {e}")

    # Prophet
    if use_prophet:
        try:
            prophet_model = train_prophet(y_train)
            yhat, lo, hi = forecast_prophet(prophet_model, test_index=y_test.index)
            results["Prophet"] = {"y_pred": yhat, "lower": lo, "upper": hi}
        except Exception as e:
            st.warning(f"Prophet failed: {type(e).__name__}: {e}")

    st.session_state["baseline_results"] = results  

    # --- Metrics table ---
    try:
        table = format_baseline_report(results)
        st.subheader("Metrics")
        st.dataframe(table, width="stretch")
    except Exception as e:
        st.warning(f"Could not format metrics table: {e}")
        table = None

    # --- Overlay plot (train tail, test, and baseline forecasts) ---
    try:
        # Build forecasts dict from results: {'name': y_pred_series, ...}
        # Build forecasts dict from results: {'name': y_pred_series, ...}
        forecasts = {
            name: res["y_pred"]
            for name, res in results.items()
            if isinstance(res, dict) and "y_pred" in res
        }

        # Optional CI dicts if present in results (baseline models may not have these)
        lower = {
            name: res["lower"]
            for name, res in results.items()
            if isinstance(res, dict) and "lower" in res
        }
        upper = {
            name: res["upper"]
            for name, res in results.items()
            if isinstance(res, dict) and "upper" in res
        }

        # CI selection: only offer models that have both lower & upper
        ci_candidates = sorted(set(lower.keys()) & set(upper.keys()) & set(forecasts.keys()))
        ci_options = ["(none)"] + ci_candidates
        ci_default = 0
        ci_model = st.selectbox("Show CI band from", options=ci_options, index=ci_default)
        ci_model = None if ci_model == "(none)" else ci_model

        # Density from session (set on the sidebar toggle)
        density = st.session_state.get("density", "expanded")

        fig = plot_overlay(
            y_train=y_train,
            y_test=y_test,
            forecasts=forecasts,
            lower=lower if lower else None,
            upper=upper if upper else None,
            ci_model=ci_model,
            density=density,
            tail=200,
        )
        st.subheader("Overlay")
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Could not render overlay: {e}")
        fig = None

    # --- Exports ---
    st.subheader("Exports")
    with st.expander("Download predictions as CSV", expanded=True):
        left, right = st.columns(2)
        # Let user choose which baseline to export (naïve or movavg)
        model_names = list(results.keys())
        chosen = left.selectbox("Model", options=model_names, index=0)
        try:
            y_pred = results[chosen]["y_pred"]
            forecast_df = build_forecast_table(index=y_test.index, y_pred=y_pred)
            csv_bytes = dataframe_to_csv_bytes(forecast_df)
            fn = make_default_filenames(base=f"{chosen}_forecast")
            left.download_button("Download CSV", data=csv_bytes, file_name=fn["csv"], mime="text/csv", width="stretch")
        except Exception as e:
            left.warning(f"CSV export unavailable: {e}")

        # PNG of the overlay figure
        try:
            if fig is not None:
                png = figure_to_png_bytes(fig)
                fn = make_default_filenames(base="baseline_overlay")
                right.download_button("Download PNG", data=png, file_name=fn["png"], mime="image/png", width="stretch")
            else:
                right.info("Run baselines to enable plot export.")
        except Exception as e:
            right.warning(f"PNG export unavailable: {e}")

    st.success("Baselines ready. You can move to **Compare** or enable ARIMA/Prophet later.")

# --- COMPARE page ---
def render_compare_page() -> None:
    st.markdown("### Compare")
    st.caption("Pick horizon → generate forecasts → see metrics + overlay → export")

    y_train = st.session_state.get("train")
    y_test = st.session_state.get("test")
    freq = st.session_state.get("freq")
    baseline_results = st.session_state.get("baseline_results", {})

    if y_train is None or y_test is None:
        st.info("Finish the **Data** and **Models** pages first.")
        return

    # ---- Controls
    c1, c2 = st.columns([2, 1])
    max_h = len(y_test)
    h = c1.slider("Horizon (steps into test set)", 1, max_h, max_h)
    run_btn = c2.button("Compare models", width="stretch")

    # ---- Assemble models we have
    # Start with baselines (built from stored y_pred Series)
    models = {}
    for name, res in baseline_results.items():
        if isinstance(res, dict) and "y_pred" in res:
            # adapt stored predictions as a callable that returns the tail for chosen horizon
            def _mk_callable(series):
                def _call(hh, *_args, **_kwargs):
                    return series.iloc[:hh]
                return _call
            models[name] = _mk_callable(res["y_pred"])

    # (Optional) add ARIMA / Prophet later if you store trained models in session
    # e.g., if "arima_model" in st.session_state: models["arima"] = lambda h, **kw: forecast_arima(...)

    if not models:
        st.info("No models available yet. Run baselines on the **Models** page.")
        return

    # ---- Validate horizon & make future index (shared across models)
    try:
        H = validate_horizon(h, y_test)
    except Exception as e:
        st.warning(f"Horizon invalid: {e}")
        return

    try:
        future_idx = make_future_index(last_ts=y_train.index[-1], periods=H, freq=freq)
    except Exception as e:
        st.warning(f"Could not build future index: {e}")
        return

    if not run_btn and "compare_cache" not in st.session_state:
        st.stop()

    # ---- Generate aligned forecasts
    try:
        forecasts = generate_forecasts(models=models, horizon=H, last_ts=y_train.index[-1], freq=freq)
        st.session_state["compare_cache"] = {"forecasts": forecasts, "H": H, "future_idx": future_idx}
    except Exception as e:
        st.warning(f"Forecast generation failed: {e}")
        return

    forecasts = st.session_state["compare_cache"]["forecasts"]

    # ---- Metric options (toggles + sort)
    copt1, copt2, copt3 = st.columns([1, 1, 2])
    use_smape = copt1.checkbox("sMAPE", value=False, help="Symmetric MAPE (%)")
    use_mase  = copt2.checkbox("MASE", value=False, help="Scaled by naive MAE")
    sort_choices = ["RMSE", "MAE", "MAPE%"] + (["sMAPE%"] if use_smape else []) + (["MASE"] if use_mase else [])
    sort_by = copt3.selectbox("Sort by", options=sort_choices, index=0, help="Leaderboard order")

    # ---- Metrics table
    try:
        # y_true = the first H points of y_test
        y_true = y_test.iloc[:H]
        metrics_df = compute_metrics_table(
            y_true=y_true,
            forecasts=forecasts,
            include_smape=use_smape,
            include_mase=use_mase,
            y_train_for_mase=y_train if use_mase else None,
            sort_by=sort_by,
            ascending=True,
        )
        st.subheader("Leaderboard")
        st.caption(f"H = {H} steps")
        st.dataframe(metrics_df, width="stretch")
    except Exception as e:
        st.warning(f"Could not compute metrics: {e}")
        metrics_df = None

    # ---- Overlay plot
    # ---- Overlay plot
    try:
        # Offer CI only for models that have lower+upper aligned to y_test[:H]
        lower: dict[str, pd.Series] = {}
        upper: dict[str, pd.Series] = {}

        ci_candidates = sorted(set(lower.keys()) & set(upper.keys()) & set(forecasts.keys()))
        ci_options = ["(none)"] + ci_candidates
        ci_default = 0
        ci_model = st.selectbox("Show CI band from", options=ci_options, index=ci_default, key="compare_ci_model")
        ci_model = None if ci_model == "(none)" else ci_model

        density = st.session_state.get("density", "expanded")

        fig = plot_overlay(
            y_train=y_train,
            y_test=y_test.iloc[:H],
            forecasts=forecasts,
            lower=lower if lower else None,
            upper=upper if upper else None,
            ci_model=ci_model,
            density=density,
            tail=200,
        )
        st.subheader("Overlay")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not render overlay: {e}")
        fig = None

    # ---- Exports
    st.subheader("Exports")
    with st.expander("Download a model’s forecast as CSV", expanded=True):
        left, right = st.columns(2)
        model_names = list(forecasts.keys())
        chosen = left.selectbox("Model", options=model_names, index=0)
        try:
            y_pred = forecasts[chosen]
            fc_df = build_forecast_table(index=y_pred.index, y_pred=y_pred)
            csv_bytes = dataframe_to_csv_bytes(fc_df)
            fn = make_default_filenames(base=f"{chosen}_compare_H{H}")
            left.download_button("Download CSV", data=csv_bytes, file_name=fn["csv"], mime="text/csv", width="stretch")
        except Exception as e:
            left.warning(f"CSV export unavailable: {e}")

        try:
            if fig is not None:
                png = figure_to_png_bytes(fig)
                fn = make_default_filenames(base=f"compare_overlay_H{H}")
                right.download_button("Download PNG", data=png, file_name=fn["png"], mime="image/png", width="stretch")
            else:
                right.info("Run comparison to enable plot export.")
        except Exception as e:
            right.warning(f"PNG export unavailable: {e}")

    st.success("Comparison ready. Add ARIMA/Prophet later to broaden the race.")

# DELETE
def render_placeholder_page(name: str) -> None:
    st.markdown(f"### {name}")
    st.info(f"{name} page is coming soon. Shell only for Phase 7.")

# --- 7) Main ------------------------------------------------------------------
def main() -> None:
    setup_page()
    init_state_keys()
    import_smoke_test()

    page = sidebar_nav()    
    _inject_density_css(st.session_state["density"])
                        
    st.title("📈 Forecasting App")

    if page == "Data":
        render_data_page()
    elif page == "EDA":
        render_eda_page()
    elif page == "Models":
        render_models_page()
    elif page == "Compare":
        render_compare_page()

if __name__ == "__main__":
    main()

# .\.venv-forecasting\Scripts\activate