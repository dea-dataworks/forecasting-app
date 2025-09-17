## [0.2.0] – 2025-09-16

### Added
- Support for ARIMA and Prophet models with configurable seasonalities.
- Confidence intervals for forecasts, with export to CSV and PNG.
- Compare page with sortable leaderboard, horizon selection, and overlay plots.
- Confidence level slider for ARIMA/Prophet and seasonal controls for both models.
- Dataset summary panel showing frequency, gaps, and missing values.

### Changed
- Unified dataset summary and Train/Test split into clearer layouts with plain-English help.
- Regularization options simplified with dropdowns for common frequencies and fill methods.
- UI polish: density toggle, collapsible previews, improved navigation, and spinners with elapsed-time feedback.
- Export filenames standardized for consistency across pages.

### Fixed
- Safer handling of NaNs and zero denominators in metrics (MAE, RMSE, MAPE, sMAPE, MASE).
- Forecast export bugs in CSV/PNG downloads.
- Datetime parsing and frequency detection errors in some datasets.
