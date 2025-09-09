from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional
from src.data_input import (
    load_csv,
    detect_datetime,
    validate_frequency,
    regularize_and_fill,
    summarize_dataset,
)
from src.baselines import train_test_split_ts
from src.eda import plot_raw_series, plot_rolling, basic_stats
from src.baselines import run_baseline_suite, format_baseline_report
from src.compare import (
    validate_horizon,
    make_future_index,
    generate_forecasts,
    compute_metrics_table,
    plot_overlay,
)
from src.outputs import (
    build_forecast_table,
    dataframe_to_csv_bytes,
    figure_to_png_bytes,
    make_default_filenames,
)

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
    for k in ("df", "train", "test", "freq", "summary"):
        st.session_state.setdefault(k, None)
    st.session_state.setdefault("density", "expanded")

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
    st.sidebar.radio(
    "Density",
    options=["expanded", "compact"],
    key="density",
    horizontal=True,
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

    with st.expander("ℹ️ Detected frequency & gaps", expanded=True):
        cols = st.columns(4)
        cols[0].metric("Frequency", freq_report.get("freq") or "Unknown")
        cols[1].metric("Gaps", str(freq_report.get("gaps")) if freq_report.get("gaps") is not None else "—")
        cols[2].metric("Expected pts", str(freq_report.get("expected_points")) if freq_report.get("expected_points") is not None else "—")
        gr = freq_report.get("gap_ratio")
        cols[3].metric("Gap ratio", f"{gr:.2%}" if isinstance(gr, float) else "—")
        if not freq_report.get("is_monotonic", False):
            st.warning("Index is not strictly increasing or has duplicates. Fix your data if modeling fails.")
        if freq_report.get("freq") is None:
            st.info("No clear frequency detected. You can regularize below to help models.")

    # --- D) Optional regularization (second mutation) ---
    reg_col1, reg_col2 = st.columns([2, 1])
    do_reg = reg_col1.checkbox("Regularize to a fixed frequency", value=bool(freq_report.get("freq")))
    picked_freq = reg_col1.text_input(
        "Frequency (pandas offset alias, e.g., D, W, M, H)",
        value=freq_report.get("freq") or "",
        placeholder="e.g., D for daily",
        disabled=not do_reg,
    )
    fill = reg_col2.selectbox("Fill method", ["ffill", "interpolate", "none"], disabled=not do_reg, index=0)

    if do_reg:
        if not picked_freq:
            st.warning("Choose a frequency string to regularize.")
            return
        try:
            df_idx = regularize_and_fill(df_idx, picked_freq, fill=fill)
            st.session_state.df = df_idx
            # refresh report after regularization
            freq_report = validate_frequency(df_idx.index)
        except Exception as e:
            st.error(f"Regularization failed: {e}")
            return

    # --- E) Target selection ---
    num_cols = [c for c in df_idx.columns if pd.api.types.is_numeric_dtype(df_idx[c])]
    if not num_cols:
        st.error("No numeric columns found. Please upload data with at least one numeric value column.")
        return
    target_col = st.selectbox("Select target column", options=num_cols, index=0)
    st.session_state.target_col = target_col

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
            right.write(f"**Freq:** {summary['freq']}  \n**Gaps:** {summary['gaps']}  \n**Gap ratio:** {summary['gap_ratio']}  \n**Top missing:** {summary['top_missing']}")
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
    run_button = c2.button("Run baselines", use_container_width=True)

    if not run_button and "baseline_results" not in st.session_state:
        st.stop()

    # --- Run baselines ---
    try:
        results = run_baseline_suite(y_train=y_train, y_test=y_test, window=window)
        st.session_state["baseline_results"] = results
    except Exception as e:
        st.error(f"Baseline run failed: {e}")
        return

    results = st.session_state["baseline_results"]

    # --- Metrics table ---
    try:
        table = format_baseline_report(results)
        st.subheader("Metrics")
        st.dataframe(table, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not format metrics table: {e}")
        table = None

    # --- Overlay plot (train tail, test, and baseline forecasts) ---
    try:
        # Build forecasts dict from results: {'name': y_pred_series, ...}
        forecasts = {name: res["y_pred"] for name, res in results.items() if isinstance(res, dict) and "y_pred" in res}
        fig = plot_overlay(y_train=y_train, y_test=y_test, forecasts=forecasts, tail=200)
        st.subheader("Overlay")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not render overlay: {e}")
        fig = None

    # --- Exports ---
    st.subheader("Exports")
    with st.expander("Download baseline predictions as CSV", expanded=True):
        left, right = st.columns(2)
        # Let user choose which baseline to export (naïve or movavg)
        model_names = list(results.keys())
        chosen = left.selectbox("Model", options=model_names, index=0)
        try:
            y_pred = results[chosen]["y_pred"]
            forecast_df = build_forecast_table(index=y_test.index, y_pred=y_pred)
            csv_bytes = dataframe_to_csv_bytes(forecast_df)
            fn = make_default_filenames(base=f"{chosen}_forecast")
            left.download_button("Download CSV", data=csv_bytes, file_name=fn["csv"], mime="text/csv", use_container_width=True)
        except Exception as e:
            left.warning(f"CSV export unavailable: {e}")

        # PNG of the overlay figure
        try:
            if fig is not None:
                png = figure_to_png_bytes(fig)
                fn = make_default_filenames(base="baseline_overlay")
                right.download_button("Download PNG", data=png, file_name=fn["png"], mime="image/png", use_container_width=True)
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
    run_btn = c2.button("Compare models", use_container_width=True)

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
        H = validate_horizon(y_test, h)
    except Exception as e:
        st.error(f"Horizon invalid: {e}")
        return

    try:
        future_idx = make_future_index(last_ts=y_train.index[-1], periods=H, freq=freq)
    except Exception as e:
        st.error(f"Could not build future index: {e}")
        return

    if not run_btn and "compare_cache" not in st.session_state:
        st.stop()

    # ---- Generate aligned forecasts
    try:
        forecasts = generate_forecasts(models=models, horizon=H, last_ts=y_train.index[-1], freq=freq)
        st.session_state["compare_cache"] = {"forecasts": forecasts, "H": H, "future_idx": future_idx}
    except Exception as e:
        st.error(f"Forecast generation failed: {e}")
        return

    forecasts = st.session_state["compare_cache"]["forecasts"]

    # ---- Metrics table
    try:
        # y_true = the first H points of y_test
        y_true = y_test.iloc[:H]
        metrics_df = compute_metrics_table(y_true=y_true, forecasts=forecasts)
        st.subheader("Leaderboard")
        st.dataframe(metrics_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute metrics: {e}")
        metrics_df = None

    # ---- Overlay plot
    try:
        fig = plot_overlay(y_train=y_train, y_test=y_test.iloc[:H], forecasts=forecasts, tail=200)
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
            left.download_button("Download CSV", data=csv_bytes, file_name=fn["csv"], mime="text/csv", use_container_width=True)
        except Exception as e:
            left.warning(f"CSV export unavailable: {e}")

        try:
            if fig is not None:
                png = figure_to_png_bytes(fig)
                fn = make_default_filenames(base=f"compare_overlay_H{H}")
                right.download_button("Download PNG", data=png, file_name=fn["png"], mime="image/png", use_container_width=True)
            else:
                right.info("Run comparison to enable plot export.")
        except Exception as e:
            right.warning(f"PNG export unavailable: {e}")

    st.success("Comparison ready. Add ARIMA/Prophet later to broaden the race.")


def render_placeholder_page(name: str) -> None:
    st.markdown(f"### {name}")
    st.info(f"{name} page is coming soon. Shell only for Phase 7.")


# --- UI diagnostics (temporary) ----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def _ui_diagnostics_block():
    # Local tiny config (kept here to avoid more helpers right now)
    _cfg = {
        "compact":  dict(plot_w=880, plot_h=340, title=16, label=12.5, legend=12, table_rows=10, row_px=26),
        "expanded": dict(plot_w=960, plot_h=380, title=16.5, label=13.5, legend=13, table_rows=8,  row_px=32),
    }
    density = st.session_state.get("density", "expanded")
    cfg = _cfg["compact" if density == "compact" else "expanded"]

    with st.expander("🧪 UI Diagnostics (temporary)"):
        st.write(f"**Density:** `{density}` · Plot target: {cfg['plot_w']}×{cfg['plot_h']} px · "
                 f"Table ~{cfg['table_rows']} rows")

        # --- Plot smoke test ---
        # px → inches at 100 DPI
        w_in, h_in = cfg["plot_w"] / 100.0, cfg["plot_h"] / 100.0
        x = pd.date_range("2022-01-01", periods=120, freq="D")
        y = pd.Series(np.sin(np.linspace(0, 8, len(x))) * 10 + 100, index=x)

        fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=100)
        ax.plot(y.index, y.values)
        ax.set_title("Density-sized plot", fontsize=cfg["title"])
        ax.set_xlabel("Time", fontsize=cfg["label"])
        ax.set_ylabel("Value", fontsize=cfg["label"])
        ax.tick_params(axis="both", labelsize=max(10, int(cfg["label"] - 1)))
        fig.patch.set_alpha(0.0)   # transparent for Light/Dark
        ax.set_facecolor("none")
        st.pyplot(fig, width="stretch")

        # --- Table smoke test ---
        df_demo = pd.DataFrame({"value": np.round(y.tail(30).values, 2)}, index=y.tail(30).index)
        header_px = 42
        rows_to_show = min(len(df_demo), cfg["table_rows"])
        height_px = int(header_px + rows_to_show * cfg["row_px"])
        st.dataframe(df_demo, width="stretch", height=height_px)
        st.caption(f"Approx table height: {height_px}px (rows shown ≈ {rows_to_show})")


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