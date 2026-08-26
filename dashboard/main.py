from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.backtesting.dataset import load_feature_frame
from app.config import settings
from app.database.schema import BankRate, Prediction
from app.database.session import make_session
from app.exchange.planner import ExchangeInputs, recommend_exchange
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

st.subheader("Exchange Planner")
with st.form("planner"):
    c1, c2, c3, c4 = st.columns(4)
    target = c1.number_input("Target USD", min_value=0.0, value=10000.0, step=500.0)
    held = c2.number_input("USD Held", min_value=0.0, value=0.0, step=500.0)
    twd = c3.number_input("TWD Available", min_value=0.0, value=0.0, step=10000.0)
    payment = c4.date_input("Next Payment Date", value=None)
    submitted = st.form_submit_button("Calculate")
if submitted:
    rec = recommend_exchange(ExchangeInputs(target, held, twd, payment), risk.twd_risk_score, risk.opportunity_score)
    st.info(f"{rec.action}: exchange about ${rec.suggested_usd_to_exchange:,.0f} now. Shortfall ${rec.usd_shortfall:,.0f}.")

st.subheader("Charts")
chart_df = features.tail(365)
tab1, tab2, tab3, tab4 = st.tabs(["USD/TWD", "DXY", "US 2Y", "Foreign Flow"])
tab1.plotly_chart(px.line(chart_df, x="date", y="USDTWD_CLOSE"), use_container_width=True)
tab2.plotly_chart(px.line(chart_df, x="date", y="DXY_CLOSE"), use_container_width=True)
tab3.plotly_chart(px.line(chart_df, x="date", y="US2Y_CLOSE"), use_container_width=True)
tab4.plotly_chart(px.bar(chart_df.tail(120), x="date", y="FOREIGN_FLOW_5D"), use_container_width=True)

st.subheader("Top Contributors")
st.dataframe(risk.contributors, use_container_width=True)
