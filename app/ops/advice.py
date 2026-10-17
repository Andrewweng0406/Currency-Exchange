from __future__ import annotations

from app.monitoring.model_health import latest_model_health_summary
from app.ops.data_quality import DataQualitySummary


def guard_model_advice(session, quality: DataQualitySummary) -> DataQualitySummary:
    """Usable data alone does not establish that live probability forecasts work."""
    if quality.blocks_model_advice:
        return quality
    health = latest_model_health_summary(session)
    if health.status != "OK":
        return DataQualitySummary(
            status="UNVERIFIED_MODEL",
            label_zh="模型尚未通過追蹤驗證",
            message_zh="目前真實到期預測尚不足或表現偏弱，暫停模型選時建議；付款需求請另行安排。",
            blocks_model_advice=True,
            issues=[health.message_zh, *quality.issues],
        )
    return quality
