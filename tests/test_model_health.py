from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.schema import Base, ModelPerformance
from app.monitoring.model_health import latest_model_health_summary


def test_model_health_summary_handles_empty_history():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        summary = latest_model_health_summary(session)
    assert summary.status == "INSUFFICIENT_HISTORY"
    assert summary.metrics == []


def test_model_health_summary_groups_latest_metrics():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    now = datetime(2026, 1, 10, tzinfo=timezone.utc)
    rows = [
        ModelPerformance(
            observed_at_utc=now,
            source="model_monitoring",
            model_version="phase4_ensemble_v1",
            horizon="5d",
            metric="sample_count",
            value=30,
            window="30D",
        ),
        ModelPerformance(
            observed_at_utc=now,
            source="model_monitoring",
            model_version="phase4_ensemble_v1",
            horizon="5d",
            metric="accuracy",
            value=0.62,
            window="30D",
        ),
        ModelPerformance(
            observed_at_utc=now,
            source="model_monitoring",
            model_version="phase4_ensemble_v1",
            horizon="5d",
            metric="brier_score",
            value=0.22,
            window="30D",
        ),
        ModelPerformance(
            observed_at_utc=now,
            source="model_monitoring",
            model_version="phase4_ensemble_v1",
            horizon="5d",
            metric="twd_depreciation_recall",
            value=0.67,
            window="30D",
        ),
    ]
    with Session(engine) as session:
        session.add_all(rows)
        session.commit()
        summary = latest_model_health_summary(session)
    assert summary.status == "OK"
    assert summary.metrics[0].horizon == "5d"
    assert summary.metrics[0].window == "30D"
    assert summary.metrics[0].sample_count == 30
    assert summary.metrics[0].accuracy == 0.62
    assert summary.metrics[0].twd_depreciation_recall == 0.67
