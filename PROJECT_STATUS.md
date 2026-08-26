# Project Status

Last verified: 2026-08-26 America/Los_Angeles.

Latest GitHub commit: `6c1d5a2 Add migrations explainability and profile configuration`.

## Completed And Verified

- Data source audit: `DATA_SOURCES.md`.
- Phase 1 data ingestion with real providers and fallback behavior.
- SQLite local database and PostgreSQL-ready SQLAlchemy models.
- Feature engineering pipeline with daily USD/TWD, USD, rates, risk, Taiwan market, Asia FX, and completeness features.
- Conservative TWSE backfill script.
- Walk-forward backtesting framework with expanding yearly splits.
- 1D, 5D, and 20D models:
  - Logistic Regression.
  - XGBoost.
  - Time-series direction baseline.
  - Ensemble weights based on walk-forward performance.
- Prediction persistence with probability up/down, expected return, confidence, risk score, and input snapshot.
- Prediction interval output using historical forward-return quantiles.
- Tail risk baseline for 5D USD/TWD shock probabilities.
- TWD Risk Score, Opportunity Score, market regime detection, and estimated CBC intervention risk.
- Exchange planner with staged recommendations and exchange plan persistence.
- Profile configuration script for default USD need/payment settings.
- LINE Messaging API client, Traditional Chinese daily report, alert dedupe, and webhook helper for user ID discovery.
- Immediate alert candidates:
  - TWD depreciation warning.
  - Sudden FX move.
  - Good exchange opportunity.
  - Macro event notice.
- FastAPI backend and Streamlit dashboard.
- Economic event CSV importer with release-time UTC normalization.
- Model monitoring for matured predictions.
- Scheduler script.
- Model explanation block based on logistic coefficients and XGBoost feature importance, labeled as non-causal.
- Strategy A/B/C first-pass comparison using real walk-forward probabilities.
- Dockerfile, `.dockerignore`, Makefile, GitHub Actions test workflow, Alembic baseline migration, and readiness checker.

## Latest Local Verification

```text
pytest: 31 passed
readiness: core checks passed
alembic current: 20260826_0001 (head)
features: 2608 rows
predictions: 3 rows
fx_prices: 2628 rows
market_data: 70472 rows
foreign_flows: 264 rows
```

Docker build was not verified locally because Docker CLI is not installed on this machine.

## Current Snapshot

```text
USD/TWD: 31.824
Bank USD Sell: 31.910
TWD Risk Score: 37 / 100
Opportunity Score: 52 / 100
Recommendation: WAIT
CBC Intervention Risk: LOW (ESTIMATED)
5D USD/TWD > +1% empirical tail risk: 6%
5D USD/TWD > +2% empirical tail risk: 1%
```

## Needs User Input Before Full Production Use

- `LINE_CHANNEL_ACCESS_TOKEN`
- `LINE_USER_ID`
- `LINE_CHANNEL_SECRET`
- Exact bank your family uses for TWD to USD conversion.
- Your real planning values:
  - `MONTHLY_USD_NEED`
  - `NEXT_PAYMENT_DATE`
  - `TARGET_USD_AMOUNT`
  - `USD_ALREADY_HELD`
  - `TWD_AVAILABLE`

## External Data Limitations

- Bank of Taiwan direct programmatic access returned challenge validation in local tests. Land Bank is currently used as public fallback.
- Yahoo Finance `USDCNH=X` long history is unreliable; it should be replaced with a licensed or more reliable FX provider before heavy model reliance on CNH.
- Consensus forecasts for CPI/FOMC/NFP/etc. are not fabricated. They require official/manual entry or a reliable licensed provider.
- Fed expectations are intentionally optional until a provider with acceptable terms is selected.
- Taiwan monthly fundamentals are not yet automated; they should be imported by release timestamp to avoid look-ahead bias.

## Next Hardening Targets

- Add explicit Alembic revisions for future schema changes.
- Add calibrated tail-risk classifiers once more clean historical data is available.
- Consider SHAP after model/data stability improves; current explainability is coefficient/gain based and non-causal.
- Deploy API/dashboard/scheduler to a real environment after secrets and bank settings are provided.
