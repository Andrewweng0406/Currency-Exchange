# TWD FX Monitor

Production-oriented USD/TWD probability forecasting and exchange risk management for a student in the US whose family converts TWD to USD. The project is built around probability, confidence, risk control, auditability, and LINE Messaging API alerts. It does not claim exact future FX prices.

## Current Status

Implemented:

- Data-source audit in `DATA_SOURCES.md`.
- SQLite local development database with PostgreSQL-ready SQLAlchemy models.
- Real data ingestion for CBC USD/TWD close, Bank of Taiwan USD spot selling, FRED market/macro series, TWSE foreign flow, TAIEX, 2330, and Yahoo Finance supplemental market series.
- Provider timeout/retry handling and structured JSON logs.
- Conservative TWSE historical backfill script for TAIEX, 2330, and foreign-flow trading dates.
- Daily feature engineering pipeline with technical, macro-market, Asia FX, TWSE, TSMC, and data-completeness features.
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

For PostgreSQL:

```bash
docker compose up -d postgres
DATABASE_URL=postgresql+psycopg://twdfx:twdfx_dev_password@localhost:5432/twdfx
```

PostgreSQL will require adding a driver such as `psycopg[binary]` to your environment.

## 3. Run Phase 1 Ingestion

```bash
python scripts/ingest_phase1.py
```

The command exits with code `2` if one or more providers fail, but successful providers still write data. This is deliberate: provider failure should be visible without destroying the rest of the update.

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

The script is resumable: existing foreign-flow dates are skipped.

## 6. Build Features

```bash
python scripts/build_features.py
```

The feature set is stored in the `features` table as `daily_v1`. It includes:

- USD/TWD returns, rolling highs/lows, moving averages, volatility, RSI, ATR, Bollinger position, momentum, and rate of change.
- DXY/broad USD index returns.
- US 2Y/10Y changes and 2s10s spread.
- VIX, S&P 500, Nasdaq, USD/CNH, USD/KRW, USD/JPY returns and volatility.
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

## 8. Planned Commands

These commands will be added in later phases:

```bash
python scripts/run_backtest_phase3.py
python scripts/train_models.py --horizon all
python scripts/backtest_strategy.py
python scripts/send_line_report.py
uvicorn app.api.main:app --reload
streamlit run dashboard/main.py
python scripts/run_scheduler.py
```

## 9. Production Deployment Notes

Use PostgreSQL in production, not SQLite. Store all secrets in environment variables. Run ingestion, prediction, alerting, and model monitoring as separate scheduled jobs so a data-provider outage does not stop the dashboard or erase historical data.

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

The prediction-based Strategy C is intentionally not simulated yet because real model predictions and risk scores start in later phases. No synthetic prediction signals are used.

## 11. Phase 4 Models

Current model command:

```bash
python scripts/train_models.py --horizon all
```

For each horizon (`1d`, `5d`, `20d`) it trains:

- Logistic Regression as an interpretable baseline.
- XGBoost for nonlinear relationships.
- A time-series direction baseline that uses USD/TWD momentum and historical up-rate behavior.

The ensemble weights are computed from walk-forward validation metrics rather than fixed by hand. Model artifacts are saved locally under `models/artifacts/` and are intentionally not committed. Latest predictions and model performance are persisted to the database for monitoring.

Run strategy comparison:

```bash
python scripts/backtest_strategy.py
```

The current first-pass Strategy C uses real walk-forward 5D probabilities. It is intentionally conservative and not tuned to make the backtest look pretty. On the current local dataset, the 2023+ smoke comparison for monthly USD 10,000 needs produced:

- Fixed day once: average rate 31.4851, worst rate 33.1858, maximum regret NT$31,617.
- Equal tranches: average rate 31.4944, worst rate 32.9907, maximum regret NT$14,868.
- Risk-based strategy: average rate 31.4820, worst rate 33.0361, maximum regret NT$18,209.

This means the first-pass risk strategy slightly improved average cost versus fixed day and materially reduced maximum regret, but it did not dominate equal tranches on regret. This should be treated as an honest baseline, not a finished optimized policy.

## 12. Phase 5-7 Risk, Exchange Planner, And LINE

Current decision command:

```bash
python scripts/generate_decision.py
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
- Exchange planner outputs `WAIT`, `EXCHANGE_25_PERCENT`, `EXCHANGE_50_PERCENT`, `EXCHANGE_75_PERCENT`, or `EXCHANGE_100_PERCENT`.
- LINE Messaging API client, Traditional Chinese report formatting, alert candidates, and dedupe logic.
- Webhook helper endpoint for discovering `LINE_USER_ID`.
- Exchange plans are persisted to the `exchange_plans` table.

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
- `GET /charts/usdtwd?limit=365`

Start the dashboard:

```bash
streamlit run dashboard/main.py
```

Run model-health evaluation:

```bash
python scripts/evaluate_model_health.py
```

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

Scheduled jobs use UTC internally and run ingestion, feature building, model training, monitoring, and LINE report sending.

## Docker

Build and run the API container:

```bash
docker build -t twd-fx-monitor .
docker run --env-file .env -p 8000:8000 twd-fx-monitor
```

For production, use PostgreSQL and persistent storage. Model artifacts are local generated files and should be rebuilt or mounted in the production environment.

## 14. Design Guardrails

- No random train/test split for time-series modeling.
- No look-ahead bias: macro values must be joined by release timestamp.
- No fabricated consensus forecasts or Fed probabilities.
- Poor model performance lowers ensemble weight.
- Missing provider data lowers confidence.
- Exchange recommendations are staged percentages, not all-or-nothing trading calls.
