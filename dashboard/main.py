from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.backtesting.dataset import load_feature_frame
from app.backtesting.strategy_compare import compare_exchange_strategies, summarize_strategy_comparison, walk_forward_tune_strategy
from app.config import settings
from app.database.schema import BankRate, Prediction
from app.database.session import make_session
from app.exchange.planner import ExchangeInputs, recommend_exchange
from app.line.family_reasons import family_reasons
from app.monitoring.model_health import latest_model_health_summary
from app.risk.scoring import latest_risk_snapshot
from sqlalchemy import select

st.set_page_config(page_title="TWD FX Monitor", layout="wide")
st.title("🇹🇼 TWD FX Monitor")

session = make_session(settings()["database"]["url"])
features = load_feature_frame(session)
latest = features.iloc[-1]
risk = latest_risk_snapshot(session)
bank = session.execute(select(BankRate).order_by(BankRate.observed_at_utc.desc()).limit(1)).scalar_one_or_none()
predictions = session.execute(
    select(Prediction).where(Prediction.model_version == "phase4_ensemble_v1").order_by(Prediction.observed_at_utc.desc())
).scalars().all()
pred_by_horizon = {}
for pred in predictions:
    pred_by_horizon.setdefault(pred.horizon, pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("USD/TWD", f"{latest['USDTWD_CLOSE']:.3f}", f"{latest.get('USDTWD_RETURN_1D') or 0:+.2%}")
col2.metric("Bank USD Sell", f"{bank.spot_selling:.3f}" if bank else "N/A", bank.bank_name if bank else "")
col3.metric("TWD Risk", f"{risk.twd_risk_score}/100", ", ".join(risk.regime[:2]))
col4.metric("Opportunity", f"{risk.opportunity_score}/100", f"Confidence {risk.confidence:.0%}")

st.subheader("Predictions")
pred_cols = st.columns(3)
for idx, horizon in enumerate(["1d", "5d", "20d"]):
    pred = pred_by_horizon.get(horizon)
    interval = ""
    if pred and pred.input_snapshot:
        import json

        snapshot = json.loads(pred.input_snapshot)
        bounds = snapshot.get("prediction_interval_80") or {}
        if bounds.get("lower") is not None and bounds.get("upper") is not None:
            interval = f"80% {bounds['lower']:+.2%} to {bounds['upper']:+.2%}"
    pred_cols[idx].metric(
        horizon.upper(),
        f"USD ↑ {pred.prob_up:.0%}" if pred else "N/A",
        interval or (f"Expected {pred.expected_return:+.2%}" if pred and pred.expected_return is not None else ""),
    )

tail_cols = st.columns(3)
tail_cols[0].metric("5D > +1%", f"{risk.tail_risk.probabilities.get('USD_TWD_UP_GT_1PCT', 0):.0%}")
tail_cols[1].metric("5D > +2%", f"{risk.tail_risk.probabilities.get('USD_TWD_UP_GT_2PCT', 0):.0%}")
tail_cols[2].metric("CBC Risk", f"{risk.cbc_intervention_risk.level}", "ESTIMATED")

st.subheader("Timing Advisor")
with st.form("planner"):
    payment = st.date_input("Next USD payment date", value=None)
    submitted = st.form_submit_button("Check timing")
if submitted:
    rec = recommend_exchange(ExchangeInputs(1, 0, next_payment_date=payment), risk.twd_risk_score, risk.opportunity_score)
    timing_labels = {
        "WAIT": "Wait. Current signals do not suggest urgency.",
        "EXCHANGE_25_PERCENT": "Start watching closely. Consider exchanging if you need USD soon.",
        "EXCHANGE_50_PERCENT": "Reasonable timing to exchange. Avoid waiting for a perfect rate.",
        "EXCHANGE_75_PERCENT": "Elevated TWD depreciation risk. Consider exchanging sooner.",
        "EXCHANGE_100_PERCENT": "Timing or risk is tight. Prioritize securing USD soon.",
    }
    st.info(timing_labels.get(rec.action, rec.action.replace("_", " ")))

st.subheader("Charts")
chart_df = features.tail(365)
tab1, tab2, tab3, tab4 = st.tabs(["USD/TWD", "DXY", "US 2Y", "Foreign Flow"])
tab1.plotly_chart(px.line(chart_df, x="date", y="USDTWD_CLOSE"), use_container_width=True)
tab2.plotly_chart(px.line(chart_df, x="date", y="DXY_CLOSE"), use_container_width=True)
tab3.plotly_chart(px.line(chart_df, x="date", y="US2Y_CLOSE"), use_container_width=True)
tab4.plotly_chart(px.bar(chart_df.tail(120), x="date", y="FOREIGN_FLOW_5D"), use_container_width=True)

st.subheader("主要判斷依據")
reason_rows = [reason.__dict__ for reason in family_reasons(latest.to_dict())]
st.dataframe(reason_rows, use_container_width=True)

st.subheader("模型追蹤")
health = latest_model_health_summary(session)
st.caption(health.message_zh)
if health.metrics:
    st.dataframe([item.__dict__ for item in health.metrics], use_container_width=True)
else:
    st.info(health.label_zh)

st.subheader("Strategy Backtest")
bt_col1, bt_col2, bt_col3 = st.columns(3)
backtest_start_year = bt_col1.number_input("Start year", min_value=2021, max_value=2026, value=2023, step=1)
backtest_target = bt_col2.number_input("USD need", min_value=1000, max_value=100000, value=10000, step=1000)
use_tuning = bt_col3.checkbox("Walk-forward tuning", value=True)
if use_tuning:
    tuned = walk_forward_tune_strategy(session, target_usd=float(backtest_target), start_year=int(backtest_start_year))
    summary = tuned.summary
    years = [year.__dict__ | {"policy": year.policy.__dict__} for year in tuned.years]
    if summary:
        st.metric("Tuned Strategy", "PASS" if summary.passed else "NOT YET", f"NT$ {summary.savings_vs_fixed_day_twd:,.0f}")
        st.caption(summary.conclusion_zh)
    st.dataframe(years, use_container_width=True)
else:
    comparisons = compare_exchange_strategies(session, target_usd=float(backtest_target), start_year=int(backtest_start_year))
    summary = summarize_strategy_comparison(comparisons)
    if summary:
        st.metric("Model Timing", "PASS" if summary.passed else "NOT YET", f"NT$ {summary.savings_vs_fixed_day_twd:,.0f}")
        st.caption(summary.conclusion_zh)
    st.dataframe([item.__dict__ for item in comparisons], use_container_width=True)
