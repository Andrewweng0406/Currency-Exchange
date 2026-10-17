# TWD FX Monitor

Production-oriented USD/TWD probability forecasting and exchange risk management for a student in the US whose family converts TWD to USD. The project is built around probability, confidence, risk control, auditability, and LINE Messaging API alerts. It does not claim exact future FX prices.

## Prospective Validation

[The prospective validation ledger](docs/FORWARD_VALIDATION_2026-10-16.md) captures complete input/model/quote evidence and freezes first observed outcomes in separate tables. View `GET /forward-validation` or run `python scripts/run_forward_validation.py --status-only`. [Production rollout](docs/PRODUCTION_VALIDATION_2026-10-17.md) now runs the sequential pipeline daily at 22:30 UTC, with PostgreSQL and persistent model/evidence storage. LINE notifications require explicit enablement; prospective collection does not automatically activate model timing.

## 2026-10-16 Official Data And Quantitative Corrections

The [second quantitative review](docs/QUANT_REVIEW_2026-10-16.md) supersedes prior snapshots. The active feature/model versions are `daily_v3` / `phase4_ensemble_v3`: 4,669 official CBC trading dates from 2008, a verified Hua Nan USD spot parser and quote timestamp, release-aware H.10 alignment, explicit 50% and historical-rate baselines, immutable first forecasts, and baseline-relative live monitoring. Historical v1/v2 records remain for audit.

A corrected monthly comparison has approximately NT$268 mean savings per USD 10,000, but the block interval spans -NT$877 to +NT$1,401. Its verdict is `MIXED`, not `PASS`, and model timing remains disabled. Research sigmoid calibration has not been promoted. No remote deployment or LINE send was performed during the review.

Reproduce the read-only diagnostic with `python scripts/review_quant_models.py`. Current quote capture/backfill evidence is recorded by `scripts/verify_fx_sources.py`; the one-off local `--apply` path backs up SQLite before writing.

## Earlier 2026-10-16 Audit Update

Earlier v2 engineering findings and numbers are preserved in the [first audit](docs/AUDIT_2026-10-16.md) and [historical analysis](docs/HISTORICAL_ANALYSIS_2026-10-16.md). They are archived research snapshots; the active v3 data, definitions, and results are described above.

For a new read-only historical diagnostic, run `python scripts/analyze_history.py --as-of 2026-10-16 --feature-set daily_v3`. Current CSV/HTML output is separated by date and feature set under `data/analysis/`; it does not overwrite model artifacts or database rows.

## Current Status

Implemented:

- Data-source audit in `DATA_SOURCES.md`.
- SQLite local development database with PostgreSQL-ready SQLAlchemy models.
- Real data ingestion for CBC USD/TWD close, Hua Nan USD spot selling (Land Bank fallback), FRED market/macro series, TWSE foreign flow, TAIEX, 2330, and Yahoo Finance supplemental market series.
- Provider timeout/retry handling and structured JSON logs.
- Conservative TWSE historical backfill script for TAIEX, 2330, and foreign-flow trading dates.
- Daily feature engineering pipeline with technical, macro-market, Asia FX, TWSE, TSMC, and data-completeness features.
- Stale-data guardrails: market features only forward-fill for a configurable short window so old prices cannot masquerade as current data.
- Unit tests for config, TWSE parsing, indicators, and feature building.
- Walk-forward backtesting, 1D/5D/20D ensemble models, risk scoring, exchange planner, LINE alerts, API, dashboard, scheduler, economic event import, model monitoring, and strategy comparison.

## 1. Create Python Environment

```bash
cd twd-fx-monitor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use:

```bash
make install
```

## 2. Configure Environment

```bash
cp .env.example .env
```

For local development, the default SQLite database is enough:

```bash
DATABASE_URL=sqlite:///data/twd_fx_monitor.db
```

Optional OpenAI AI risk interpreter:

```bash
OPENAI_ENABLED=false
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.6
```

For PostgreSQL:

```bash
docker compose up -d postgres
DATABASE_URL=postgresql+psycopg://twdfx:twdfx_dev_password@localhost:5432/twdfx
```

PostgreSQL will require adding a driver such as `psycopg[binary]` to your environment.

## 3.1 Database Migrations

Alembic is configured for production schema tracking:

```bash
alembic upgrade head
```

The current revision is a baseline migration because the first local SQLite database was created before Alembic was introduced. New schema changes should be added as explicit Alembic revisions.

## 3. Run Phase 1 Ingestion

```bash
python scripts/ingest_phase1.py
```

The command exits with code `2` if one or more providers fail, but successful providers still write data. This is deliberate: provider failure should be visible without destroying the rest of the update.

If FRED times out for US 2Y or US 10Y, Phase 1 automatically falls back to the official U.S. Treasury Daily Treasury Par Yield Curve XML feed for recent 2Y/10Y observations. VIX, S&P 500, and Nasdaq also have Yahoo Finance fallback symbols configured for continuity when FRED is unavailable.

Refresh one configured FRED series without running the full ingestion pipeline:

```bash
python scripts/ingest_fred_series.py --series-key usd_cny --timeout-seconds 60
```

If the production runtime can reach the database but times out when downloading a large FRED CSV, stream the official CSV from a machine with working FRED access:

```bash
curl -fsS "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXCHUS" \
  | python scripts/ingest_fred_series.py --series-key usd_cny --csv-path -
```

## 4. Run Tests

```bash
pytest
```

Or:

```bash
make test
```

## 5. Backfill TWSE History

This uses official TWSE endpoints conservatively. It first backfills monthly TAIEX/2330 data, then uses known TAIEX trading dates to avoid unnecessary weekend/holiday requests for foreign-flow data.

```bash
python scripts/backfill_twse_history.py --years 1 --sleep-seconds 0.5
```

When TWSE is rate-limited or unavailable, use Yahoo Finance as a clearly labeled fallback for TAIEX/2330 history:

```bash
python scripts/backfill_twse_history.py --years 10 --no-include-foreign-flow --no-twse-enabled --sleep-seconds 0
```

The script is resumable: existing foreign-flow dates are skipped.

## 6. Build Features

```bash
python scripts/build_features.py
```

The feature set is stored in the `features` table as `daily_v3`. It includes:

- USD/TWD returns, rolling highs/lows, moving averages, volatility, RSI, ATR, Bollinger position, momentum, and rate of change.
- DXY/broad USD index returns.
- US 2Y/10Y changes and 2s10s spread.
- VIX, S&P 500, Nasdaq, USD/CNH, USD/KRW, USD/JPY returns and volatility.
- `CHINA_FX_PROXY_*`, an explicitly labeled China FX proxy that uses FRED USD/CNY (`DEXCHUS`) history first and only falls back to CNH when CNY is unavailable.
- TAIEX, 2330, TSM ADR returns and TSMC volume z-score.
- Foreign-flow 1D/3D/5D/20D sums and z-score.
- `DATA_COMPLETENESS` and per-source missing-data flags for confidence logic.

## 7. LINE Official Account Setup

LINE Notify is discontinued and is not used. This project uses LINE Official Account plus LINE Messaging API.

High-level setup:

1. Create or open a LINE Official Account.
2. Enable Messaging API in the LINE Developers console.
3. Create a long-lived channel access token.
4. Get your LINE user ID from webhook events or a small helper endpoint that will be added in the LINE phase.
5. Set these in `.env`:

```bash
LINE_CHANNEL_ACCESS_TOKEN=...
LINE_USER_ID=...
LINE_CHANNEL_SECRET=...
```

Never commit `.env`.

To discover your `LINE_USER_ID`, deploy or expose the API endpoint below as the LINE webhook URL, send a message to the Official Account, then inspect the response/logs:

```text
POST /line/webhook
```

The endpoint verifies the LINE signature when `LINE_CHANNEL_SECRET` is set and returns any `source.userId` values present in webhook events.

## 8. Common Commands

```bash
python scripts/run_backtest_phase3.py
python scripts/train_models.py --horizon all
python scripts/model_feature_importance.py --top-n 10
python scripts/backtest_strategy.py --start-year 2023 --target-usd 10000
python scripts/backtest_strategy.py --start-year 2023 --target-usd 10000 --tune
python scripts/send_line_report.py
python scripts/run_daily_pipeline.py
uvicorn app.api.main:app --reload
streamlit run dashboard/main.py
python scripts/run_scheduler.py
```

## 9. Production Deployment Notes

Use PostgreSQL in production, not SQLite. Store all secrets in environment variables.

On Railway, the API process can also start the production scheduler when this service variable is set:

```text
SCHEDULER_ENABLED=true
LINE_REPORT_ENABLED=false
```

With the scheduler enabled, the running API container keeps the LINE webhook alive and also runs the daily jobs in UTC:

- 22:30 UTC (Taipei 06:30 the following day): ingest, build features, train all horizons, capture/evaluate prospective forecasts, then evaluate model health.
- Notifications run only when `LINE_REPORT_ENABLED=true`; collection defaults to no messages.
- A failed step stops dependent steps; no fixed clock interval is assumed to guarantee completion.
- CBC and bank observations are required. Supplemental provider failures are logged as `DEGRADED`; freshness and advice guards remain active.
- Run only one scheduler process/replica to avoid duplicate jobs across API workers.
- Mount persistent storage at `/app/models/artifacts` to retain models and prospective evidence across deploys. PostgreSQL retains the ledger and original history.
- Container startup applies additive Alembic migrations before serving requests. Each pipeline step has a 30-minute timeout.

Check production scheduler state:

```text
GET /scheduler/status
GET /pipeline/status
```

`/pipeline/status` reads persistent execution records: the latest run, its trigger (`manual` or `scheduled`), current step, results, and last complete success. `FAILED`, `INTERRUPTED`, `STALE`, and `STATE_ERROR` require investigation; `NO_RECORD` means no record under this instrumentation, not proof that the older pipeline never ran. Runs use a shared process lock on the evidence volume, so overlapping manual and scheduled executions cannot run two pipelines. Records are stored under `models/artifacts/operations/` and survive deployments. This operational status does not certify forecasting skill.

For one-off production-style execution, run:

```bash
python scripts/run_daily_pipeline.py
```

The production scheduler should use conservative data frequencies:

- FRED and TWSE: daily.
- Bank quote: during Taiwan bank business hours or before decision reports.
- Yahoo supplemental market data: daily unless intraday alerting is explicitly enabled.
- Macro/event data: aligned to official release timestamps.

## 10. Phase 3 Backtesting Framework

Current Phase 3 command:

```bash
python scripts/run_backtest_phase3.py
```

It currently reports:

- Feature row range and count.
- Modeling feature count.
- Expanding walk-forward yearly splits.
- Forward targets for 1D, 5D, and 20D horizons.
- Tail-risk target columns for USD/TWD rises above 0.5%, 1%, 1.5%, and 2%.
- Classification metrics scaffolding: accuracy, precision, recall, F1, ROC-AUC, Brier score, calibration error, and TWD depreciation recall.
- Regression metrics scaffolding: MAE and RMSE.
- Smoke-test comparison of fixed-date exchange vs equal-tranche exchange.
- Strategy C comparison using real walk-forward 5D model probabilities and exchange opportunity scores. No synthetic prediction signals are used.

## 11. Phase 4 Models

Current model command:

```bash
python scripts/train_models.py --horizon all
```

After training, inspect the current model drivers:

```bash
python scripts/model_feature_importance.py --top-n 10
```

For each horizon (`1d`, `5d`, `20d`) it trains:

- Logistic Regression as an interpretable baseline.
- XGBoost for nonlinear relationships.
- A time-series direction baseline that uses USD/TWD momentum and historical up-rate behavior.

The ensemble weights are computed from walk-forward validation metrics rather than fixed by hand. Model artifacts are saved locally under `models/artifacts/` and are intentionally not committed. Latest predictions and model performance are persisted to the database for monitoring.

Run strategy comparison:

```bash
python scripts/backtest_strategy.py --start-year 2023 --target-usd 10000
python scripts/backtest_strategy.py --start-year 2023 --target-usd 10000 --tune
```

The `--tune` mode performs walk-forward threshold selection: each test year uses only prior years to choose the policy thresholds, then evaluates the next year. This avoids picking parameters with knowledge of the future.

The current model-timing strategy uses real walk-forward 5D probabilities and exchange opportunity scores. It is intentionally conservative and not tuned to make the backtest look pretty. On the current local dataset, the 2023+ fixed-policy comparison for monthly USD 10,000 needs produced:

- Fixed day once: average rate 31.4847, worst rate 33.1858, maximum regret NT$31,617.
- Equal tranches: average rate 31.4927, worst rate 32.9907, maximum regret NT$14,868.
- Model timing once: average rate 31.4945, worst rate 33.0453, maximum regret NT$31,617.

The strategy report includes `verdict`, `should_use_for_timing`, `key_findings_zh`, and `summary.passed`. The model-timing strategy should only be considered a true improvement if it lowers average TWD cost without worsening worst-rate risk or cost volatility.

This means the current local model-timing strategy did not beat fixed-day exchange on average cost. On the current production dataset after TAIEX/TSMC fallback backfill, the strategy reduced average cost by about NT$161 per USD 10,000 and improved worst rate, but cost volatility increased. Until average cost, worst-rate risk, and volatility improve together, it should be presented as a risk reminder, not as proof that the system saves money.

The API exposes the same report at:

```text
GET /backtests/strategy?start_year=2023&target_usd=10000&tune=true
```

The Streamlit dashboard includes a Strategy Backtest section with pass/fail status, savings versus fixed date, and per-year tuned policies.

## 12. Phase 5-7 Risk, Timing Advisor, And LINE

Current decision command:

```bash
python scripts/generate_decision.py
```

Configure the optional planner profile if you want deadline awareness. The family-facing LINE report does not show dollar amounts or percentages; it only says whether the timing looks favorable, neutral, or risky.

The family-facing report explains the timing call with a short ranked list of human-readable market inputs instead of model internals. The current reason groups are:

- 美元指數 DXY / broad USD index.
- 美國2年債利率.
- 外資買賣超台股.
- 台股與台積電.
- 人民幣、韓元、日圓亞洲貨幣壓力.
- VIX / 美股全球風險情緒.

```bash
python scripts/configure_profile.py \
  --target-usd-amount 10000 \
  --usd-already-held 3000 \
  --next-payment-date 2026-09-15 \
  --monthly-usd-need 2500 \
  --twd-available 250000
```

Current LINE dry-run command:

```bash
python scripts/send_line_report.py
```

To send through LINE Messaging API after setting `.env`:

```bash
python scripts/send_line_report.py --send
```

Implemented:

- TWD Risk Score from model probabilities, USD/TWD momentum, DXY, rates, Asia FX, global risk-off, and data quality.
- Exchange Opportunity Score from USD/TWD percentile over 20D, 60D, 120D, and 252D.
- Market regimes: `RISK_ON`, `RISK_OFF`, `USD_STRONG`, `USD_WEAK`, `HIGH_VOL`, `LOW_VOL`, `TWD_IDIOSYNCRATIC`.
- Timing advisor internally maps risk to `WAIT`, `EXCHANGE_25_PERCENT`, `EXCHANGE_50_PERCENT`, `EXCHANGE_75_PERCENT`, or `EXCHANGE_100_PERCENT`, but family-facing messages only describe timing, not amount.
- LINE Messaging API client, concise Traditional Chinese timing report, family-friendly reason ranking, alert candidates, and dedupe logic.
- Webhook helper endpoint for discovering `LINE_USER_ID`.
- Exchange plans are persisted to the `exchange_plans` table.
- Daily report includes a data-status line. Core data failures make timing guidance conservative.
- Immediate alert candidates: data quality warning, TWD depreciation warning, sudden FX move, good exchange opportunity, and macro event notice.
- Model explanation block using logistic coefficients and XGBoost gain, labeled as association rather than causation.

Required from you before real LINE sending:

- `LINE_CHANNEL_ACCESS_TOKEN`
- `LINE_USER_ID`
- Later: the exact bank your family uses, if not Bank of Taiwan/Land Bank fallback.

## 13. Phase 8-10 API, Dashboard, Scheduler, Monitoring

Start the API:

```bash
uvicorn app.api.main:app --reload --port 8000
```

Useful endpoints:

- `GET /health`
- `GET /overview`
- `GET /features/latest`
- `GET /data-quality`
- `GET /model-health`
- `GET /ai/latest`
- `GET /charts/usdtwd?limit=365`

Start the dashboard:

```bash
streamlit run dashboard/main.py
```

Run model-health evaluation:

```bash
python scripts/evaluate_model_health.py
```

This compares saved predictions with actual USD/TWD outcomes after each horizon matures. The report includes sample count, accuracy, Brier score, TWD depreciation recall, MAE, and RMSE for 7D, 30D, and 90D windows. If too few real daily predictions have matured, the status stays `INSUFFICIENT_HISTORY` instead of pretending the model is already proven.

Run optional OpenAI AI risk interpretation:

```bash
OPENAI_ENABLED=true python scripts/generate_ai_interpretation.py --save
```

The OpenAI layer summarizes existing model outputs and market context in Traditional Chinese. It does not directly change `prob_up`, `risk_score`, or exchange percentages.

Run operational readiness checks:

```bash
python scripts/check_readiness.py
```

Audit data coverage and critical feature missingness:

```bash
python scripts/data_quality_report.py
```

The data-quality report includes a machine-readable `summary`. `BLOCKING` means core data is stale or missing and LINE timing advice should be treated conservatively. `LIMITED` means today's core data is usable, but some supplemental latest values or longer historical features remain incomplete.

CNH free historical data is unreliable through Yahoo Finance. The system keeps CNH as supplemental and uses `CHINA_FX_PROXY_*` for Asia FX pressure. That proxy uses FRED `DEXCHUS` USD/CNY history first and only falls back to CNH if CNY is unavailable. It is not labeled or treated as CNH.

Market features use `features.max_market_stale_days` from `config/settings.yaml` to prevent stale forward-filled values from being treated as valid observations. The default is 7 calendar days, which tolerates weekends and holidays but flags longer source outages or historical gaps as missing data. This may make data-quality reports look stricter, but it avoids training models on silently stale TAIEX, TSMC, DXY, rates, or Asia FX values.

Import economic events from a manually prepared official-source CSV:

```bash
python scripts/import_economic_events.py path/to/events.csv
```

Required CSV columns:

```text
event_name,release_time,previous,forecast,actual
```

Consensus forecasts are not fabricated. If forecast data is unavailable from a reliable source, leave it blank and the surprise field will remain unavailable.

An example file is available at `sample_data/economic_events.example.csv`.

Run scheduler:

```bash
python scripts/run_scheduler.py
```

Scheduled jobs use UTC internally and run ingestion, feature building, model training, prospective validation, and monitoring in order. LINE sending requires `LINE_REPORT_ENABLED=true`. In Docker/Railway the scheduler uses the container's current Python executable, so it does not require a local `.venv`.

## Docker

Build and run the API container:

```bash
docker build -t twd-fx-monitor .
docker run --env-file .env -p 8000:8000 twd-fx-monitor
```

For production, use PostgreSQL and persistent storage. Model artifacts are local generated files and should be rebuilt or mounted in the production environment.

## 14. Design Guardrails

- No random train/test split for time-series modeling.
- Point-in-time data remains a requirement: current daily source lags are conservative proxies, not verified release timestamps or historical vintages. See the audit limitations.
- No fabricated consensus forecasts or Fed probabilities.
- Poor model performance lowers ensemble weight.
- Missing provider data lowers confidence.
- Exchange recommendations are staged percentages, not all-or-nothing trading calls.
