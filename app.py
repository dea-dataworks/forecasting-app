from __future__ import annotations
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from src.data_input import (load_csv,detect_datetime,validate_frequency,regularize_and_fill,summarize_dataset,)
from src.baselines import train_test_split_ts
from src.eda import (plot_raw_series, plot_rolling, basic_stats, plot_decomposition,
                    plot_acf_series, plot_pacf_series, infer_season_length_from_freq)
from src.baselines import run_baseline_suite, format_baseline_report
from src.compare import (validate_horizon, make_future_index, generate_forecasts,compute_metrics_table,plot_overlay,build_combined_forecast_table)
from src.classical import (HAS_PMDARIMA, HAS_PROPHET,train_auto_arima, forecast_auto_arima,train_prophet, forecast_prophet,)
from src.outputs import (build_forecast_table,dataframe_to_csv_bytes,figure_to_png_bytes,make_default_filenames,)



# --- 1) Page setup ------------------------------------------------------------
def setup_page() -> None:
    st.set_page_config(
        page_title="Forecasting App",
        page_icon="📈",
        layout="centered",
        initial_sidebar_state="expanded",
    )

# --- 2) Session state init ----------------------------------------------------
def init_state_keys() -> None:
    for k in ("df", "train", "test", "freq", "summary",
              "target_col","datetime_col","baseline_results","compare_cache"):
        st.session_state.setdefault(k, None)
    st.session_state.setdefault("density", "expanded")

    # NEW: stash for fitted models + default train-tail for overlay plots
    st.session_state.setdefault("models", {})        # {"arima": ..., "prophet": ...}
    st.session_state.setdefault("train_tail", 200)   # visual window for overlay plots

# --- 3) Sidebar navigation ----------------------------------------------------
def sidebar_nav() -> str:
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        options=("Data", "EDA", "Models", "Compare"),
        label_visibility="collapsed",
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.radio("Compact Mode", 
                    options=["expanded", "compact"],
                    key="density", horizontal=True,
                    help="Compact = tighter padding, slightly smaller fonts & plots",
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
    # 1) Base stylesheet (shared rules)
    try:
        css_path = Path(__file__).with_name("ui.css")
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.warning(f"UI CSS not found: {e}")

    # 2) Compact overrides (conditionally injected; no JS)
    if str(density_value).lower() == "compact":
        try:
            compact_path = Path(__file__).with_name("ui_compact.css")
            if compact_path.exists():
                compact_css = compact_path.read_text(encoding="utf-8")
                st.markdown(f"<style>{compact_css}</style>", unsafe_allow_html=True)
            else:
                # Minimal inline fallback so you see an effect immediately
                st.markdown(
                    """
                    <style>
                      .block-container { padding-top: 0.75rem; padding-bottom: 0.75rem; }
                      section[data-testid="stSidebar"] { padding-top: 0.25rem; }
                      .stMarkdown p { margin-bottom: 0.25rem; }
                      .stButton button, .stDownloadButton button { padding: 0.25rem 0.5rem; }
                      .stDataFrame table { font-size: 0.9rem; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.sidebar.warning(f"Compact CSS load failed: {e}")

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
    # st.subheader("Data")
    # st.caption("Use this to render something immediately before wiring CSV upload.")
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
    if isinstance(df_raw.index, pd.DatetimeIndex):
        # Already parsed earlier (e.g., after coming back from EDA); don't re-detect
        df_idx = df_raw
        chosen_col = st.session_state.get("datetime_col")
    else:
        try:
            df_idx, chosen_col = detect_datetime(df_raw)  # sets DatetimeIndex, sorted
        except Exception as e:
            st.error(f"Datetime detection failed: {e}")
            return
        st.session_state.datetime_col = chosen_col

    st.session_state.df = df_idx


    # --- C) Frequency report (read-only) ---
    try:
        freq_report = validate_frequency(df_idx.index)
    except Exception as e:
        st.warning(f"Frequency inference issue: {e}")
        freq_report = {"freq": None, "gaps": None, "expected_points": None, "gap_ratio": None, "is_monotonic": False}

    # Compute summary early so we can render it inside the "Sampling frequency & gaps" expander
    try:
        summary = summarize_dataset(df_idx)
        st.session_state.summary = summary
    except Exception as e:
        summary = None
        st.warning(f"Summary unavailable: {e}")

    with st.expander("Sampling frequency & gaps", expanded=True):
        if summary:
            left, right = st.columns(2)
            left.write(
                f"**Rows:** {summary['rows']:,}  "
                f"\n**Columns:** {summary['cols']}  "
                f"\n**Start:** {summary['start']}  "
                f"\n**End:** {summary['end']}"
            )
            gap_txt = (
                    f"{summary['gap_ratio']:.2%}" if isinstance(summary.get('gap_ratio'), (int, float))
                    else (summary.get('gap_ratio') or '—')
        )
        right.write(
            f"**Sampling frequency:** {to_human_freq(summary['freq']) if summary['freq'] else '—'}  "
            f"\n**Missing timestamps:** {summary['gaps']}  "
            f"\n**Expected timestamps:** {summary['expected_points']}  "
            f"\n**% missing timestamps:** {gap_txt}"
        )

        # Keep helpful nudges under the summary
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

    # Subheader to make the target choice visually prominent
    st.subheader("Target selection")

    target_col = st.selectbox(
        "Select target (y) column",
        options=num_cols,
        index=0,
        help="This is the series we’ll forecast. Other columns remain available for reference."
    )
    st.session_state.target_col = target_col

    # Clearer status line about the chosen target
    _col_dtype = str(df_idx[target_col].dtype)
    _missing_pct = float(df_idx[target_col].isna().mean() * 100.0)
    st.caption(
        f"Selected target: **{target_col}**  |  dtype = {_col_dtype}  |  missing = {_missing_pct:.1f}%"
    )

    # --- F) Train/Test split ---
    st.subheader("Train/Test split")

    # Split type with plain-English help
    split_type = st.radio(
        "Select split type",
        options=["percentage", "count"],
        horizontal=True,
        help="• Percentage : Reserve the last X% of rows for testing.\n"
            "• Count: Reserve the last N rows for testing."
    )

    n_rows = len(df_idx)

    if split_type == "percentage":
        perc = st.slider(
            "Test size (% of rows)",
            min_value=5, max_value=50, value=20, step=5
        )
        pct = float(perc)
        frac = pct / 100.0
        test_size = float(frac)
        H = max(1, int(round(n_rows * frac)))
    else:
        # Allow up to n_rows-1 but cap early to avoid invalid selections
        max_count = max(1, n_rows - 1)
        default_cnt = min(40, max_count)
        cnt = st.slider("Test size (last N rows)", min_value=1, max_value=max_count, value=default_cnt)
        test_size = int(cnt)
        H = test_size
        pct = (H / n_rows) * 100.0 if n_rows else 0.0

    # Live interpretation right under the widget
    st.caption(f"**H = {H} rows (≈ {pct:.0f}% of data)**")

    # One-liner note about how we split
    st.caption("We always take the **last H points** as test (no shuffling) to simulate forecasting the future.")

    # Validation nudges (friendly, no crashes)
    if H <= 0:
        st.warning("Pick a positive test size for H.")
        st.stop()
    if H >= n_rows:
        st.warning(f"Test set (H={H}) cannot be the entire dataset. Reduce H below {n_rows}.")
        st.stop()

    try:
        y_train, y_test = train_test_split_ts(df_idx, target_col=target_col, test_size=test_size)
        st.session_state.train = y_train
        st.session_state.test = y_test
        st.session_state.freq = freq_report.get("freq")
    except Exception as e:
        st.error(f"Split failed: {e}")
        return
    
    # New: lightweight preview, collapsed by default + last-5 peek
    with st.expander("Data preview (first 5 rows)", expanded=True):
        n_show = st.selectbox("Rows to show", options=[5, 10, 20], index=0)
        st.dataframe(df_idx.head(n_show), width="stretch")

        if st.button("Show last 5", help="Peek at the last rows to sanity-check the split."):
            st.dataframe(df_idx.tail(5), width="stretch")

    # Keep this quick status line visible
    st.caption(f"Target: `{target_col}` · Train: {len(y_train):,} · Test: {len(y_test):,}")
    rows = len(df_idx)
    freq_human = to_human_freq(st.session_state.get('freq'))
    st.success(
        f"✅ Data ready — {rows:,} rows • {freq_human} • Target: {target_col} • H = {H} "
        f"(train {len(y_train):,} / test {len(y_test):,})"
    )


# --- EDA page ---
def render_eda_page() -> None:
    st.markdown("### EDA")
    st.caption("One main plot (toggle rolling overlay) • Quick stats • Decomposition & ACF/PACF on demand")

    df = st.session_state.get("df")
    target_col = st.session_state.get("target_col")

    if df is None or target_col is None:
        st.info("Load data first on the **Data** page (upload, choose target, split).")
        return

    y_df = df[[target_col]]  # keep as DataFrame for our plotting helpers

    # --- Main plot: raw + optional rolling overlay ---
    st.subheader("Main plot")
    overlay = st.checkbox("Show rolling overlay", value=False)

    if not overlay:
        st.caption("Shows the series as-is.")
        try:
            fig = plot_raw_series(y_df)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not render main plot: {e}")
    else:
        c1, c2 = st.columns([2, 1])
        window = c1.slider("Rolling window (points)", min_value=3, max_value=90, value=7, step=1)
        show_var = c2.checkbox("Variance bands", value=False, help="Show local variability around the rolling mean.")
        st.caption("Rolling mean smooths short-term noise; bands (optional) hint at local variability.")
        try:
            fig = plot_rolling(y_df, window=window, show_var=show_var)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not render main plot: {e}")

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

    # --- 4) Decomposition ---
    with st.expander("Decomposition", expanded=False):
        st.caption("Trend shows long-term direction. Seasonal captures repeated patterns; residual is leftover noise.")
        period_choice = st.selectbox(
            "Seasonal period",
            options=["auto", 7, 12, 24, 52],
            index=0,
            help="Auto = guess from frequency. Common choices: 7 (weekly seasonality for daily data), 12 (monthly seasonality), 24 (hourly seasonality), 52 (weekly seasonality for weekly data)."
        )
        period_arg = None if period_choice == "auto" else int(period_choice)

        try:
            fig_dec = plot_decomposition(y_df, period=period_arg)
            st.pyplot(fig_dec)
        except Exception as e:
            st.warning(f"Decomposition unavailable: {e}")

    # --- 5) Autocorrelation diagnostics (ACF / PACF) ---
    with st.expander("Autocorrelation (ACF) & Partial (PACF)", expanded=False):
        st.caption("ACF highlights repeating patterns across lags; PACF shows direct effects at each lag. We auto-clip nlags to keep it readable (≤30 or 10% of length).")
        try:
            col1, col2 = st.columns(2)
            with col1:
                fig_acf = plot_acf_series(y_df)
                st.pyplot(fig_acf)
            with col2:
                fig_pacf = plot_pacf_series(y_df)
                st.pyplot(fig_pacf)
        except Exception as e:
            st.warning(f"ACF/PACF unavailable: {e}")

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
    
    # Confidence level (shared UI; for now used by ARIMA only)
    ci_level = st.slider("Confidence level", min_value=0.50, max_value=0.99, value=0.95, step=0.01,
                         help="For interval bands. 0.95 → alpha = 0.05")
    st.session_state["ci_level"] = ci_level
    alpha = 1.0 - ci_level

    # Early gate remains after toggles so they are visible on first visit
    if not run_button and "baseline_results" not in st.session_state:
        st.stop()

    # --- Run baselines ---
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
    # ARIMA (safe defaults)
    if use_arima:
        try:
            freq = st.session_state.get("freq")
            m = infer_season_length_from_freq(freq)
            seasonal = m is not None
            arima_model = train_auto_arima(y_train, seasonal=seasonal, m=m)

            # NEW — stash fitted model
            st.session_state.setdefault("models", {})
            st.session_state["models"]["arima"] = arima_model

            # CI: alpha = 1 - level
            yhat, lo, hi = forecast_auto_arima(
                arima_model,
                test_index=y_test.index,
                alpha=alpha
            )
            results["ARIMA"] = {"y_pred": yhat, "lower": lo, "upper": hi}
        except Exception as e:
            st.warning(f"ARIMA failed: {type(e).__name__}: {e}")

    # Prophet (uses the same Confidence level as ARIMA; Prophet expects interval_width in 0–1)
    if use_prophet:
        try:
            ci_level = st.session_state.get("ci_level", 0.95)
            prophet_model = train_prophet(y_train, interval_width=ci_level)

            # NEW — stash fitted model
            st.session_state.setdefault("models", {})
            st.session_state["models"]["prophet"] = prophet_model

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

        # NEW: Train tail slider (visual window only)
        max_tail = max(50, len(y_train))
        tail = st.slider("Train tail (points)", min_value=50, max_value=max_tail, value=int(st.session_state.get("train_tail", 200)))
        st.session_state["train_tail"] = tail

        fig = plot_overlay(
            y_train=y_train,
            y_test=y_test,
            forecasts=forecasts,
            lower=lower if lower else None,
            upper=upper if upper else None,
            ci_model=ci_model,
            density=density,
            tail=tail,   # ← use the slider value
        )

        st.subheader("Overlay")
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Could not render overlay: {e}")
        fig = None

    # --- Residual diagnostics ---
    with st.expander("Residual diagnostics", expanded=False):
        models_dict = st.session_state.get("models", {}) or {}
        available = []
        if "arima" in models_dict:
            available.append("ARIMA")
        if "prophet" in models_dict:
            available.append("Prophet")

        if not available:
            st.info("Train **ARIMA** or **Prophet** to see residual diagnostics.")
        else:
            # Optional seasonality hint from inferred freq
            freq = st.session_state.get("freq")
            m_hint = infer_season_length_from_freq(freq)
            if m_hint:
                st.caption(f"Seasonality hint: m = {m_hint} (from freq = {freq}).")

            choice = st.selectbox("Model", options=available, index=0, key="resid_model_choice")
            y_train = st.session_state.get("train")

            resid = None
            if choice == "ARIMA":
                arima_model = models_dict.get("arima")
                try:
                    # pmdarima-style in-sample prediction
                    fitted_vals = pd.Series(arima_model.predict_in_sample(), index=y_train.index)
                except Exception:
                    # fallbacks
                    try:
                        fv = getattr(arima_model, "fittedvalues", None)
                        if fv is None:
                            raise AttributeError("no fittedvalues")
                        fv = pd.Series(fv)
                        if len(fv) == len(y_train):
                            fitted_vals = fv.set_axis(y_train.index)
                        else:
                            # align last part to train index
                            fitted_vals = pd.Series(fv.values[-len(y_train):], index=y_train.index)
                    except Exception as e:
                        st.warning(f"ARIMA fitted values unavailable ({type(e).__name__}: {e}). Using 1-step lag as fallback.")
                        fitted_vals = y_train.shift(1)
                resid = (y_train - fitted_vals).dropna()

            else:  # Prophet
                prophet_model = models_dict.get("prophet")
                try:
                    ds_df = pd.DataFrame({"ds": y_train.index})
                    fcst = prophet_model.predict(ds_df)
                    fitted_vals = pd.Series(fcst["yhat"].values, index=y_train.index)
                    resid = (y_train - fitted_vals).dropna()
                except Exception as e:
                    st.warning(f"Prophet fitted values unavailable: {e}")
                    resid = pd.Series(dtype="float64")

            if resid is None or len(resid) < 3:
                st.info("Residuals unavailable or too short to diagnose.")
            else:
                # Residual line
                try:
                    fig_r, ax_r = plt.subplots(figsize=(10, 3))
                    ax_r.plot(resid.index, resid.values, linewidth=1)
                    ax_r.axhline(0, linestyle="--", linewidth=1)
                    ax_r.set_title(f"{choice} residuals")
                    ax_r.set_xlabel("Time")
                    ax_r.set_ylabel("Residual")
                    fig_r.tight_layout()
                    st.pyplot(fig_r)
                except Exception as e:
                    st.warning(f"Residual plot unavailable: {e}")

                # ACF / PACF on residuals (clip lags: ≤30 or 10% of length)
                try:
                    nlags = min(30, max(1, len(resid) // 10))
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_acf = plot_acf_series(resid.to_frame("resid"), max_lags=nlags)
                        st.pyplot(fig_acf)
                    with c2:
                        fig_pacf = plot_pacf_series(resid.to_frame("resid"), max_lags=nlags)
                        st.pyplot(fig_pacf)
                except Exception as e:
                    st.warning(f"Residual ACF/PACF unavailable: {e}")
    # --- Model cards (new) ---
    st.subheader("Model cards")
    models_dict = st.session_state.get("models", {}) or {}
    level = float(st.session_state.get("ci_level", 0.95))

    c1, c2 = st.columns(2)

    # ARIMA card
    with c1:
        if "arima" in models_dict:
            am = models_dict["arima"]

            order = (
                getattr(am, "order", None)
                or getattr(getattr(am, "model_", None), "order", None)
                or getattr(getattr(am, "model", None), "order", None)
            )
            sorder = (
                getattr(am, "seasonal_order", None)
                or getattr(getattr(am, "model_", None), "seasonal_order", None)
                or getattr(getattr(am, "model", None), "seasonal_order", None)
            )

            aic = None
            cand = getattr(am, "aic", None)
            if callable(cand):
                try:
                    aic = float(cand())
                except Exception:
                    aic = None
            elif cand is not None:
                try:
                    aic = float(cand)
                except Exception:
                    aic = None
            if aic is None:
                cand2 = getattr(getattr(am, "model_", None), "aic", None)
                if cand2 is not None:
                    try:
                        aic = float(cand2)
                    except Exception:
                        aic = None

            aic_txt = f"{aic:.2f}" if isinstance(aic, (int, float)) else "—"
            st.markdown(
                "**ARIMA**  \n"
                f"(p,d,q): {order or '—'}  \n"
                f"(P,D,Q,m): {sorder or '—'}  \n"
                f"AIC: {aic_txt}  \n"
                f"CI level: {int(level*100)}%"
            )
        else:
            st.info("ARIMA not trained.")

    # Prophet card
    with c2:
        if "prophet" in models_dict:
            pm = models_dict["prophet"]

            enabled = []
            try:
                seas = getattr(pm, "seasonalities", {}) or {}
                for name in ("daily", "weekly", "yearly"):
                    if name in seas:
                        enabled.append(name)
            except Exception:
                for name in ("daily", "weekly", "yearly"):
                    flag = getattr(pm, f"{name}_seasonality", None)
                    if flag:
                        enabled.append(name)

            seas_txt = ", ".join(enabled) if enabled else "—"

            cp_n = None
            try:
                cp = getattr(pm, "changepoints", None)
                if cp is not None:
                    try:
                        cp_n = len(cp)
                    except Exception:
                        cp_n = None
            except Exception:
                pass

            cp_mode = "auto"
            try:
                explicit = getattr(pm, "changepoints", None)
                if explicit is not None and hasattr(explicit, "__iter__") and len(explicit) > 0:
                    ncp = getattr(pm, "n_changepoints", None)
                    if not ncp and cp_n:
                        cp_mode = "manual"
            except Exception:
                pass
            cp_txt = f"{cp_mode} (n={cp_n})" if cp_n is not None else f"{cp_mode} (n=—)"

            st.markdown(
                "**Prophet**  \n"
                f"seasonalities: {seas_txt}  \n"
                f"changepoints: {cp_txt}  \n"
                f"CI level: {int(level*100)}%"
            )
        else:
            st.info("Prophet not trained.")

    # --- Exports ---
    st.subheader("Exports")

    with st.expander("Download predictions as CSV", expanded=True):
        left, right = st.columns(2)

        # Helper: ensure timezone-naive index for CSVs
        def _naive_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
            try:
                # If tz-aware, drop tz; otherwise returns as-is
                return idx.tz_convert(None)
            except Exception:
                try:
                    return idx.tz_localize(None)
                except Exception:
                    return idx

        # Build a lookup of available model results with optional CI
        # results[name] = {"y_pred": Series, "lower": Series?, "upper": Series?}
        model_names = [n for n, r in results.items() if isinstance(r, dict) and "y_pred" in r]
        chosen = left.selectbox("Model", options=model_names, index=0)

        # Per-model CSV (includes CI columns if present)
        try:
            r = results[chosen]
            idx = _naive_index(y_test.index)
            level = float(st.session_state.get("ci_level", 0.95))
            H = len(y_test)
            ci_pct = int(round(level * 100))

            # Pass lower/upper only if present
            lower = r.get("lower", None)
            upper = r.get("upper", None)

            fc_df = build_forecast_table(
                index=idx,
                y_pred=r["y_pred"],
                lower=lower,
                upper=upper,
            )
            # Add metadata columns if you want them in the CSV
            fc_df["model"] = chosen
            fc_df["level"] = ci_pct

            csv_bytes = dataframe_to_csv_bytes(fc_df)
            fn = make_default_filenames(base=f"forecast_{chosen.lower()}_h{H}_ci{ci_pct}")
            left.download_button(
                "Download CSV (selected model)",
                data=csv_bytes,
                file_name=fn["csv"],
                mime="text/csv",
                width="stretch",
            )
        except Exception as e:
            left.warning(f"CSV export unavailable: {e}")

        # Combined CSV (all models in one table; adds a 'model' column)
        try:
            level = float(st.session_state.get("ci_level", 0.95))
            ci_pct = int(round(level * 100))
            idx = _naive_index(y_test.index)
            H = len(y_test)

            frames = []
            for name in model_names:
                r = results[name]
                fc = build_forecast_table(
                    index=idx,
                    y_pred=r["y_pred"],
                    lower=r.get("lower"),
                    upper=r.get("upper"),
                )
                fc["model"] = name
                fc["level"] = ci_pct
                frames.append(fc)

            if frames:
                combined = pd.concat(frames, axis=0)
                csv_bytes_all = dataframe_to_csv_bytes(combined)
                fn_all = make_default_filenames(base=f"forecast_allmodels_h{H}_ci{ci_pct}")
                right.download_button(
                    "Download CSV (combined)",
                    data=csv_bytes_all,
                    file_name=fn_all["csv"],
                    mime="text/csv",
                    width="stretch",
                )
            else:
                right.info("No forecasts available to combine.")
        except Exception as e:
            right.warning(f"Combined CSV export unavailable: {e}")

    # Keep PNG export of the overlay below CSVs
    with st.expander("Download overlay plot (PNG)", expanded=False):
        try:
            if fig is not None:
                png = figure_to_png_bytes(fig)
                fn = make_default_filenames(base="baseline_overlay")
                st.download_button("Download PNG", data=png, file_name=fn["png"], mime="image/png", width="stretch")
            else:
                st.info("Run baselines to enable plot export.")
        except Exception as e:
            st.warning(f"PNG export unavailable: {e}")


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

    with st.expander("Download comparisons as CSV (combined)", expanded=True):
    # Use the first H points of y_test for a common, tz-naive index
        idx = y_test.iloc[:H].index
        try:
            # CI level (if/when you add per-model lower/upper later)
            level = float(st.session_state.get("ci_level", 0.95))
            ci_pct = int(round(level * 100))

            combined_df = build_combined_forecast_table(
                forecasts=forecasts,
                index=idx,
                level=ci_pct,
                # lower=lower_dict,  # optional (future)
                # upper=upper_dict,  # optional (future)
            )

            csv_bytes = dataframe_to_csv_bytes(combined_df)
            fn = make_default_filenames(base=f"compare_allmodels_h{H}_ci{ci_pct}")
            st.download_button(
                "Download CSV (combined forecasts)",
                data=csv_bytes,
                file_name=fn["csv"],
                mime="text/csv",
                width="stretch",
            )
        except Exception as e:
            st.warning(f"Combined CSV export unavailable: {e}")

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
    _inject_density_css(str(st.session_state.get("density", "expanded")).lower())
                        
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