from __future__ import annotations
import streamlit as st
import pandas as pd

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
        with open("src/ui.css", "r", encoding="utf-8") as f:
            css = f.read()
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
def render_data_page() -> None:
    load_sample_button()
    df = st.session_state.df
    if df is not None:
        st.markdown("### Preview")
        st.dataframe(df.head(10), width="stretch")
        st.caption(f"Rows: {len(df):,} · Columns: {len(df.columns)} · Start: {df.index.min()} · End: {df.index.max()}")

        # NEW: quick visual proof for density + fonts + plot/table sizing
        _ui_diagnostics_block()

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
        render_placeholder_page("EDA")
    elif page == "Models":
        render_placeholder_page("Models")
    elif page == "Compare":
        render_placeholder_page("Compare")

if __name__ == "__main__":
    main()


# .\.venv-forecasting\Scripts\activate