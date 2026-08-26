# TWD FX Monitor

Production-oriented USD/TWD probability forecasting and exchange risk management for a student in the US whose family converts TWD to USD. The project is built around probability, confidence, risk control, auditability, and LINE Messaging API alerts. It does not claim exact future FX prices.

## Current Status

Phase 1 is implemented:

- Data-source audit in `DATA_SOURCES.md`.
- SQLite local development database with PostgreSQL-ready SQLAlchemy models.
- Real data ingestion for CBC USD/TWD close, Bank of Taiwan USD spot selling, FRED market/macro series, TWSE foreign flow, TAIEX, 2330, and Yahoo Finance supplemental market series.
- Provider timeout/retry handling and structured JSON logs.
- Initial unit tests for config and TWSE parsing.

Later phases will add feature engineering, walk-forward backtests, models, risk scoring, exchange planner, LINE alerts, dashboard, economic calendar, and monitoring.

## 1. Create Python Environment

```bash
cd twd-fx-monitor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

## 5. LINE Official Account Setup

LINE Notify is discontinued and is not used. Later phases will use LINE Official Account plus LINE Messaging API.

High-level setup:

1. Create or open a LINE Official Account.
2. Enable Messaging API in the LINE Developers console.
3. Create a long-lived channel access token.
4. Get your LINE user ID from webhook events or a small helper endpoint that will be added in the LINE phase.
5. Set these in `.env`:

```bash
LINE_CHANNEL_ACCESS_TOKEN=...
LINE_USER_ID=...
```

Never commit `.env`.

## 6. Planned Commands

These commands will be added in later phases:

```bash
python scripts/build_features.py
python scripts/train_models.py --horizon 1d
python scripts/train_models.py --horizon 5d
python scripts/train_models.py --horizon 20d
python scripts/backtest_strategy.py
python scripts/send_line_test.py
uvicorn app.api.main:app --reload
streamlit run dashboard/main.py
python scripts/run_scheduler.py
```

## 7. Production Deployment Notes

Use PostgreSQL in production, not SQLite. Store all secrets in environment variables. Run ingestion, prediction, alerting, and model monitoring as separate scheduled jobs so a data-provider outage does not stop the dashboard or erase historical data.

The production scheduler should use conservative data frequencies:

- FRED and TWSE: daily.
- Bank quote: during Taiwan bank business hours or before decision reports.
- Yahoo supplemental market data: daily unless intraday alerting is explicitly enabled.
- Macro/event data: aligned to official release timestamps.

## 8. Design Guardrails

- No random train/test split for time-series modeling.
- No look-ahead bias: macro values must be joined by release timestamp.
- No fabricated consensus forecasts or Fed probabilities.
- Poor model performance lowers ensemble weight.
- Missing provider data lowers confidence.
- Exchange recommendations are staged percentages, not all-or-nothing trading calls.
