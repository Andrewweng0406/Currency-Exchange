from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.backtesting.dataset import load_feature_frame
from app.backtesting.strategy_compare import build_strategy_report, compare_exchange_strategies, walk_forward_tune_strategy
from app.config import settings
from app.database.schema import Prediction
from app.database.session import make_session
from app.exchange.planner import ExchangeInputs, recommend_exchange
from app.line.family_reasons import family_reasons
from app.monitoring.model_health import latest_model_health_summary
from app.monitoring.forward_validation import forward_validation_summary
from app.models.training import MODEL_VERSION
from app.risk.scoring import latest_risk_snapshot
from app.ops.advice import guard_model_advice
from app.ops.bank_quotes import latest_bank_quote
from app.ops.data_quality import data_coverage_report, summarize_data_quality
from app.scheduler.state import pipeline_status
from sqlalchemy import select

st.set_page_config(page_title="TWD FX Monitor", layout="wide")
st.title("🇹🇼 TWD FX Monitor")

with make_session(settings()["database"]["url"]) as session:
    features = load_feature_frame(session)
    if features.empty:
        st.info("尚無可用匯率特徵，請先完成資料匯入與特徵建立。")
        st.stop()
    latest = features.iloc[-1]
    quality = guard_model_advice(session, summarize_data_quality(data_coverage_report(session)))
    st.caption(f"市場資料日期：{latest['date']}；此為每日觀察資料。")
    if quality.blocks_model_advice:
        st.error(quality.message_zh + " 選時建議已停用。")
        st.caption("；".join(quality.issues))
    elif quality.status == "LIMITED":
        st.warning(quality.message_zh)
    risk = latest_risk_snapshot(session)
    bank = latest_bank_quote(session)
    predictions = session.execute(
        select(Prediction).where(Prediction.model_version == MODEL_VERSION,
                                 Prediction.source == "model_training_latest",
                                 Prediction.observed_at_utc == latest["date"].to_pydatetime())
        .order_by(Prediction.id.asc())
    ).scalars().all()
    pred_by_horizon = {}
    for pred in predictions:
        pred_by_horizon.setdefault(pred.horizon, pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("USD/TWD", f"{latest['USDTWD_CLOSE']:.3f}", f"{latest.get('USDTWD_RETURN_1D') or 0:+.2%}")
    col2.metric("Bank USD Sell", f"{bank.spot_selling:.3f}" if bank else "N/A", bank.bank_name if bank else "")
    if bank:
        st.caption(f"銀行報價：{bank.bank_name}，{bank.observed_at_utc} UTC（來源：{bank.source}）；銀行美元即期賣出價才是買美元的參考成本。")
    col3.metric("TWD Risk", f"{risk.twd_risk_score}/100", ", ".join(risk.regime[:2]))
    col4.metric("Opportunity", f"{risk.opportunity_score}/100", f"資料與模型一致度 {risk.confidence:.0%}（非準確率）")

    st.subheader("Predictions")
    st.caption("僅顯示上述市場資料日期的預測；N/A 表示該期限尚未建立。機率仍在追蹤驗證中。")
    pred_cols = st.columns(3)
    for idx, horizon in enumerate(["1d", "5d", "20d"]):
        pred = pred_by_horizon.get(horizon)
        interval = ""
        if pred and pred.input_snapshot:
            import json

            snapshot = json.loads(pred.input_snapshot)
            bounds = snapshot.get("prediction_interval_80") or {}
            if bounds.get("lower") is not None and bounds.get("upper") is not None:
                interval = f"歷史報酬 10–90 分位：{bounds['lower']:+.2%} 至 {bounds['upper']:+.2%}"
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
    if submitted and quality.blocks_model_advice:
        st.warning("資料尚未通過檢查，暫不提供選時建議。")
    if submitted and not quality.blocks_model_advice:
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
    for tab, column, bar in [(tab1, "USDTWD_CLOSE", False), (tab2, "DXY_CLOSE", False),
                             (tab3, "US2Y_CLOSE", False), (tab4, "FOREIGN_FLOW_5D", True)]:
        if column not in chart_df or chart_df[column].dropna().empty:
            tab.info("此資料來源暫無可用資料。")
        else:
            figure = px.bar(chart_df.tail(120), x="date", y=column) if bar else px.line(chart_df, x="date", y=column)
            tab.plotly_chart(figure, use_container_width=True)

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

    st.subheader("事前預測追蹤")
    execution = pipeline_status()
    execution_labels = {"NO_RECORD": "尚無流程執行紀錄", "RUNNING": "資料更新中", "SUCCEEDED": "最近一次更新完成",
                        "FAILED": "最近一次更新失敗", "INTERRUPTED": "上次更新已中斷",
                        "STALE": "更新已逾期", "STATE_ERROR": "無法讀取更新紀錄"}
    execution_label = execution_labels.get(execution["status"], execution["status"])
    if execution["status"] in {"FAILED", "INTERRUPTED", "STALE", "STATE_ERROR"}:
        st.warning(execution_label)
    else:
        st.caption(execution_label)
    if execution.get("last_success"):
        st.caption(f"上次完整更新：{execution['last_success']['finished_at_utc']}。執行成功不代表預測已驗證有效。")
    forward = forward_validation_summary(session)
    st.caption(forward["scope"])
    if not forward["cohorts"]:
        st.info("尚未建立事前紀錄；啟動每日資料流程後會開始累積。")
    else:
        for cohort in forward["cohorts"]:
            label = cohort["protocol_hash"][:12]
            current = "目前規則" if cohort["protocol_hash"] == forward["current_protocol_hash"] else "先前規則（分開統計）"
            st.write(f"{current} {label}：已記錄 {cohort['snapshots']} 筆，到期 {cohort['matured']} 筆，待到期 {cohort['pending']} 筆。")
            if cohort["metrics"]:
                st.dataframe(cohort["metrics"], use_container_width=True)
    if forward["integrity_errors"]:
        st.error("事前紀錄的指紋不符，請先排查資料完整性。")

    st.subheader("Strategy Backtest")
    bt_col1, bt_col2, bt_col3 = st.columns(3)
    backtest_start_year = bt_col1.number_input("Start year", min_value=2021, max_value=date.today().year, value=2023, step=1)
    backtest_target = bt_col2.number_input("USD need", min_value=1000, max_value=100000, value=10000, step=1000)
    use_tuning = bt_col3.checkbox("Walk-forward tuning", value=True)
    st.caption("研究回測使用市場收盤價，未計銀行價差與費用；不代表實際可成交成本或已證明能省錢。")
    if not st.button("執行策略回測"):
        st.stop()
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
        report = build_strategy_report(comparisons, target_usd=float(backtest_target))
        summary = report.summary
        if summary:
            st.metric("Model Timing", report.label_zh, f"NT$ {summary.savings_vs_fixed_day_twd:,.0f}")
            st.caption(report.caution_zh)
            for finding in report.key_findings_zh:
                st.write(f"- {finding}")
        st.dataframe([item.__dict__ for item in comparisons], use_container_width=True)
