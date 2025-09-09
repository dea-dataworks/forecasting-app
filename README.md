# Time-Series Forecasting App (Streamlit)

Upload a CSV, detect the datetime index, explore the series, run baselines and classical models (optional ARIMA/Prophet), compare results, and download forecasts/plots — all in one lightweight app.

> **Status:** v0.2 “Release Prep & QA” – Phases 1–10 implemented.  
> **Python:** 3.12 recommended.  
> **Streamlit:** 1.39+ (uses the new `width="stretch"` param).

---

## Features

- **Data input & validation**
  - Read CSV safely, detect/parse datetime, enforce uniqueness & sort, infer frequency, optionally regularize to a fixed grid.
- **EDA**
  - Raw plot, rolling window plot, quick stats (min/mean/max).
- **Baselines**
  - Chronological split (no shuffling), Naive, Moving Average, metrics (MAE, RMSE, MAPE with safe handling).
- **Classical models** *(optional deps; app skips gracefully if missing)*
  - Auto-ARIMA via `pmdarima`.
  - Prophet with common seasonalities.
- **Compare**
  - Horizon validation, aligned forecasts, overlay chart, metrics leaderboard.
- **Outputs**
  - Downloadable CSV of forecasts and PNG of plots.

---

## Quickstart

```bash
# 1) Create & activate a clean venv (Python 3.12+)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install core requirements
pip install -r requirements.txt

# 3) (Optional) Enable classical models
# These are heavier; skip if you just want baselines.
pip install pmdarima
pip install prophet  # may require a C++ toolchain; see Troubleshooting

# 4) Run the app
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

---

## How to use the app

1. **Data tab**
   - Upload a CSV. Choose/confirm the datetime column if needed.
   - Review the inferred frequency and (optionally) regularize/fill gaps.
   - Pick the value column (single numeric series).
2. **EDA tab**
   - Inspect the raw line plot, rolling mean, and quick stats.
3. **Models tab**
   - Choose a **chronological split** ratio.
   - Run **Baselines** (always available).
   - Optionally toggle **ARIMA** and/or **Prophet** (only if installed).
   - Download forecast CSV/plot.
4. **Compare tab**
   - Select which models to include.
   - Set a horizon (validated against the test set).
   - View the overlay plot + leaderboard table.
   - Download the comparison outputs.

> Tip: The UI includes a **density toggle** (compact/expanded). Streamlit reruns on every widget change; session state is used so your data and trained models persist.

---

## Project structure

```
.
├─ app.py                 # Streamlit UI (nav, pages, wiring)
├─ ui.css                 # Optional CSS tweaks for density/spacing
├─ requirements.txt       # Core deps (Streamlit, pandas, numpy, matplotlib)
├─ CHANGELOG.md           # Phase-by-phase notes
└─ src/
   ├─ __init__.py
   ├─ data_input.py       # CSV load, datetime detect, freq check, regularize, summary
   ├─ eda.py              # Raw & rolling plots, basic stats
   ├─ baselines.py        # Split, naive, moving average, metrics
   ├─ classical.py        # Auto-ARIMA + Prophet (optional imports guarded)
   ├─ compare.py          # Horizon validate, forecast adapter, overlay, leaderboard
   └─ outputs.py          # Forecast table builder, CSV/PNG serializers, filenames
```

---

## Configuration & notes

- **Streamlit version:** Uses the new `width="stretch"` replacement for deprecated `use_container_width`.  
  Make sure you’re on **Streamlit ≥ 1.39**.
- **Optional dependencies:**
  - `pmdarima` for ARIMA (pip-installable on Python 3.12; Python 3.13 support may lag).
  - `prophet` for Prophet (requires a C++ toolchain via `cmdstanpy` under the hood).
- **Design guardrails:**
  - No data leakage: chronological split only.
  - DatetimeIndex enforced; functions validate shapes/lengths.
  - Metrics handle NaNs/zero-division safely.

---

## Troubleshooting

**Prophet install fails (Windows)**
- Install Microsoft C++ Build Tools, then:
  ```bash
  pip install prophet
  ```
- If build is still heavy, consider skipping Prophet for now; baselines and ARIMA still work.

**`pmdarima` fails to build**
- Use **Python 3.12** in a fresh venv. On 3.13, wheels may be unavailable at times.
- Upgrade pip & build tools: `pip install --upgrade pip setuptools wheel`.

**Streamlit warning about `use_container_width`**
- This repo already uses `width="stretch"`. Ensure Streamlit **≥ 1.39**:
  ```bash
  pip install "streamlit>=1.39,<2"
  ```

---


