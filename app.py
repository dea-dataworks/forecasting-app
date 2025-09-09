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

# --- 3) Sidebar navigation ----------------------------------------------------
def sidebar_nav() -> str:
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        options=("Data", "EDA", "Models", "Compare"),
        label_visibility="collapsed",
        index=0,
    )

# --- 4) Import smoke test (no logic calls) -----------------------------------
def import_smoke_test() -> None:
    try:
        # Import ONLY to verify paths; do not call heavy functions yet.
        from src import data_input, eda, baselines, classical, compare  # noqa: F401
    except Exception as e:
        st.sidebar.warning(f"⚠️ Import check: {type(e).__name__}: {e}")

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
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Rows: {len(df):,} · Columns: {len(df.columns)} · Start: {df.index.min()} · End: {df.index.max()}")

def render_placeholder_page(name: str) -> None:
    st.markdown(f"### {name}")
    st.info(f"{name} page is coming soon. Shell only for Phase 7.")

# --- 7) Main ------------------------------------------------------------------
def main() -> None:
    setup_page()
    init_state_keys()
    import_smoke_test()

    page = sidebar_nav()
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