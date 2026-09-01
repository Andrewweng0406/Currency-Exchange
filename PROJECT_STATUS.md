# Project Status

Last verified: 2026-08-29 America/Los_Angeles.

Latest committed baseline before this audit: `e5e51f7 Update verified project status`.

## Session Update 2026-08-26

- Added a Hua Nan Commercial Bank provider (`app/providers/hncb.py`) and made it the primary `bank` provider in `config/settings.yaml`, since that is the bank the user's family actually uses; Land Bank stays as the public fallback (Bank of Taiwan is still registered but no longer primary/fallback since the family doesn't bank there). Confirmed the JSON endpoint (`https://www.hncb.com.tw/hncb/rest/exRate/all`) once during development, then it became unreachable from this machine (same bot-mitigation pattern as Bank of Taiwan). On 2026-08-26 the full ingest succeeded through the fallback chain and persisted the latest bank quote from Land Bank, not Hua Nan. The exact Hua Nan spot-row field names are inferred from standard Taiwan bank rate-page conventions, not exhaustively verified against a captured spot response — worth double-checking against a real response once the endpoint is reachable again.
- Fixed a `data-quality` report bug: the `features:daily_v1` coverage check was counting all rows in the `features` table regardless of `feature_set`, not just `daily_v1` rows. Now filters correctly.
- Combined `_coverage()`'s three per-dataset queries (count/min/max) into one query.
- Added `GET /data-quality` and `make data-quality` for data coverage/feature missingness audits.
- Created `.env` from `.env.example` locally (not committed; still needs real values, see below).

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
- Exchange timing advisor with staged internal recommendations and exchange plan persistence.
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
- Concise family-friendly LINE daily report deployed to Railway. The family report now focuses on timing and does not show exchange amounts or percentages.
- Production scheduler support behind `SCHEDULER_ENABLED=true`, plus `scripts/run_daily_pipeline.py` for one-off daily automation.
- Data-quality summary now feeds the LINE daily report and alert engine. Core data failures produce conservative timing guidance plus a deduped `DATA_QUALITY_WARNING`; nonblocking historical limitations show as "部分資料有限".
- Feature engineering now limits market-data forward-fill freshness through `features.max_market_stale_days`, so old TAIEX/TSMC/CNH/DXY/rate values cannot silently pass as current model inputs.
- TAIEX/TSMC historical fallback through Yahoo Finance is available for periods when TWSE official endpoints are rate-limited or unavailable; source remains explicitly labeled.
- Strategy backtest now includes an explicit `model_timing_once` pass/fail summary so the project cannot claim savings unless the validation actually supports it.
- Strategy threshold walk-forward tuning now tests each year with policies selected only from prior years, and is exposed through CLI, API, and dashboard.
- Family-facing LINE and dashboard reasons now rank plain-language market inputs such as DXY, US 2Y, foreign flow, Taiwan equities, Asia FX, and VIX instead of exposing model-internal feature names.

## Latest Local Verification

```text
pytest: 65 passed
readiness: core checks passed
alembic current: 20260826_0001 (head)
ingest_phase1.py: 17 providers succeeded, 0 failed, 72,574 rows written/updated
data-quality summary: OK after strict stale-data checks, CNY proxy ingestion, and TAIEX/TSMC Yahoo fallback backfill
strategy backtest 2023+: fixed policy did not beat fixed_day_once locally; walk-forward tuned policy improves average cost but still fails full pass criteria when volatility worsens
features: 2609 rows
predictions: 3 rows
fx_prices: 2630 rows
market_data: 70479 rows
foreign_flows: 265 rows
latest bank quote: Land Bank fallback, USD spot selling 31.899 at 2026-08-26T23:40:01Z
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

- ~~`LINE_CHANNEL_ACCESS_TOKEN`~~ — configured in Railway.
- ~~`LINE_USER_ID`~~ — configured in Railway.
- ~~`LINE_CHANNEL_SECRET`~~ — configured in Railway.
- ~~Exact bank your family uses for TWD to USD conversion~~ — resolved: Hua Nan Commercial Bank, wired up 2026-08-26.
- Your real planning values (deferred by user for now, run `python scripts/configure_profile.py` when ready):
  - `MONTHLY_USD_NEED`
  - `NEXT_PAYMENT_DATE`
  - `TARGET_USD_AMOUNT`
  - `USD_ALREADY_HELD`
  - `TWD_AVAILABLE`

## External Data Limitations

- Bank of Taiwan direct programmatic access returned challenge validation in local tests. Land Bank is currently used as public fallback.
- Yahoo Finance `USDCNH=X` long history is unreliable; it should be replaced with a licensed or more reliable FX provider before heavy model reliance on CNH.
- FRED `DEXCHUS` USD/CNY is now used only as an explicitly labeled onshore China FX proxy/fallback for Asia FX pressure when CNH history is unavailable.
- Consensus forecasts for CPI/FOMC/NFP/etc. are not fabricated. They require official/manual entry or a reliable licensed provider.
- Fed expectations are intentionally optional until a provider with acceptable terms is selected.
- Taiwan monthly fundamentals are not yet automated; they should be imported by release timestamp to avoid look-ahead bias.

## Next Hardening Targets

- Add explicit Alembic revisions for future schema changes.
- Add calibrated tail-risk classifiers once more clean historical data is available.
- Consider SHAP after model/data stability improves; current explainability is coefficient/gain based and non-causal.
- Enable and monitor Railway scheduler in production with `SCHEDULER_ENABLED=true`.
- Replace Yahoo CNH history with a more reliable offshore CNH FX provider before treating CNH-specific signals as high-confidence.
