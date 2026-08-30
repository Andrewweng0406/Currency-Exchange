from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.config import risk_policy
from app.database.schema import Alert
from app.database.upsert import upsert_rows
from app.line.formatter import alert_message
from app.ops.data_quality import DataQualitySummary
from app.risk.scoring import RiskSnapshot


@dataclass(frozen=True)
class AlertCandidate:
    alert_type: str
    severity: str
    title: str
    body: str
    dedupe_key: str
    risk_score: int | None = None


def generate_alerts(
    risk: RiskSnapshot,
    predictions: dict[str, dict[str, float | None]],
    latest_features: dict | None = None,
    upcoming_events: list[dict] | None = None,
    data_quality: DataQualitySummary | None = None,
) -> list[AlertCandidate]:
    policy = risk_policy()["alerts"]
    alerts = []
    if data_quality and data_quality.blocks_model_advice:
        issue = data_quality.issues[0] if data_quality.issues else data_quality.label_zh
        alerts.append(
            AlertCandidate(
                alert_type="DATA_QUALITY_WARNING",
                severity="HIGH",
                title="⚠️ 匯率系統資料提醒",
                body=f"{data_quality.message_zh}\n\n主要問題：\n{issue}",
                dedupe_key=f"DATA_QUALITY_WARNING:{data_quality.status}:{issue}",
                risk_score=risk.twd_risk_score,
            )
        )
    prob_5d = predictions.get("5d", {}).get("prob_up")
    if prob_5d is not None and prob_5d >= policy["twd_depreciation_warning"]["probability_up_min"]:
        alerts.append(
            AlertCandidate(
                alert_type="TWD_DEPRECIATION_WARNING",
                severity="HIGH",
                title="🚨 台幣貶值預警",
                body=f"未來 5 日 USD/TWD 上升機率：\n{prob_5d:.0%}",
                dedupe_key="TWD_DEPRECIATION_WARNING:5d",
                risk_score=risk.twd_risk_score,
            )
        )
    latest_features = latest_features or {}
    fx_return = latest_features.get("USDTWD_RETURN_1D")
    fx_vol = latest_features.get("USDTWD_VOLATILITY_20D")
    if fx_return is not None and fx_vol:
        move_z = abs(float(fx_return)) / max(float(fx_vol), 0.0001)
        if move_z >= policy["sudden_fx_move"]["rolling_zscore_min"]:
            alerts.append(
                AlertCandidate(
                    alert_type="SUDDEN_FX_MOVE",
                    severity="HIGH",
                    title="⚠️ USD/TWD 快速波動",
                    body=f"USD/TWD 1日變動：\n{float(fx_return):+.2%}\n\nRolling move z-score：約 {move_z:.1f}",
                    dedupe_key="SUDDEN_FX_MOVE",
                    risk_score=risk.twd_risk_score,
                )
            )
    if risk.opportunity_score >= policy["good_exchange_opportunity"]["opportunity_score_min"]:
        alerts.append(
            AlertCandidate(
                alert_type="GOOD_EXCHANGE_OPPORTUNITY",
                severity="MEDIUM",
                title="🟢 美元換匯機會",
                body=f"Opportunity Score：\n{risk.opportunity_score}/100\n\n目前匯率處於相對有利區間。",
                dedupe_key="GOOD_EXCHANGE_OPPORTUNITY",
                risk_score=risk.twd_risk_score,
            )
        )
    for event in (upcoming_events or [])[:3]:
        alerts.append(
            AlertCandidate(
                alert_type="MACRO_EVENT",
                severity="MEDIUM",
                title="⚠️ 匯率重大事件",
                body=f"{event.get('event_name', 'Upcoming Event')}\n\n發布：\n{event.get('release_time_utc', 'N/A')}\n\n可能造成 USD/TWD 波動增加。",
                dedupe_key=f"MACRO_EVENT:{event.get('event_name')}:{event.get('release_time_utc')}",
                risk_score=risk.twd_risk_score,
            )
        )
    return alerts


def should_send_alert(session, candidate: AlertCandidate, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    policy = risk_policy()["alerts"]
    since = now - timedelta(minutes=int(policy["dedupe_window_minutes"]))
    existing = session.execute(
        select(Alert)
        .where(Alert.dedupe_key == candidate.dedupe_key)
        .where(Alert.observed_at_utc >= since)
        .order_by(Alert.observed_at_utc.desc())
        .limit(1)
    ).scalar_one_or_none()
    return existing is None


def persist_alert(session, candidate: AlertCandidate, now: datetime | None = None) -> Alert:
    now = now or datetime.now(timezone.utc)
    message = alert_message(candidate.alert_type, candidate.title, candidate.body, candidate.risk_score)
    rows = [
        {
            "observed_at_utc": now,
            "source": "alert_engine",
            "alert_type": candidate.alert_type,
            "message": message,
            "severity": candidate.severity,
            "dedupe_key": candidate.dedupe_key,
        }
    ]
    upsert_rows(session, Alert, rows, ("alert_type", "observed_at_utc", "source"))
    return session.execute(select(Alert).where(Alert.alert_type == candidate.alert_type).order_by(Alert.id.desc()).limit(1)).scalar_one()
