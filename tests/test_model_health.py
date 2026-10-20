from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.schema import Base, ModelPerformance
from app.monitoring.model_health import latest_model_health_summary


def test_short_live_history_keeps_scores_without_fabricating_skill_bounds():
    import json
    import pandas as pd
    from app.database.schema import Feature, Prediction
    from app.monitoring.model_health import evaluate_matured_predictions

    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    dates = pd.bdate_range(end=pd.Timestamp.now(tz='UTC').normalize()-pd.Timedelta(days=2), periods=30)
    with Session(engine) as session:
        for idx, date in enumerate(dates):
            session.add(Feature(observed_at_utc=date.to_pydatetime(), source='feature_builder',
                                feature_set='daily_v3', values_json=json.dumps({'USDTWD_CLOSE':30+idx*.01})))
        for date in dates[-11:-1]:
            session.add(Prediction(observed_at_utc=date.to_pydatetime(), source='model_training_latest',
                                   model_version='phase4_ensemble_v3', horizon='1d', prob_up=.6,
                                   expected_return=.001, input_snapshot=json.dumps({
                                       'issued_at_utc':(date+pd.Timedelta(hours=12)).isoformat(),
                                       'training_up_rate':.5})))
        session.commit()
        assert evaluate_matured_predictions(session) > 0
        summary = latest_model_health_summary(session)
    assert summary.status == 'INSUFFICIENT_HISTORY'
    annual = next(m for m in summary.metrics if m.window == '365D')
    assert annual.sample_count == 10 and annual.brier_score is not None
    assert annual.skill_upper_vs_coin is None and annual.skill_upper_vs_prior is None


def test_model_health_summary_handles_empty_history():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        summary = latest_model_health_summary(session)
    assert summary.status == "INSUFFICIENT_HISTORY"
    assert summary.metrics == []


def test_cached_short_history_bounds_cannot_bypass_current_block_requirement():
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        for horizon in ('1d','5d','20d'):
            for name, value in [('sample_count',30),('effective_sample_count',12),
                                ('skill_upper_vs_coin',-.01),('skill_upper_vs_prior',-.01)]:
                session.add(ModelPerformance(observed_at_utc=now, source='model_monitoring',
                                             model_version='phase4_ensemble_v3', horizon=horizon,
                                             metric=name, value=value, window='365D'))
        session.commit()
        assert latest_model_health_summary(session).status == 'INSUFFICIENT_HISTORY'


def test_model_health_summary_groups_latest_metrics():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    now = datetime(2026, 1, 10, tzinfo=timezone.utc)
    rows = [
        ModelPerformance(
            observed_at_utc=now,
            source="model_monitoring",
            model_version="phase4_ensemble_v3",
            horizon="5d",
            metric="sample_count",
            value=30,
            window="30D",
        ),
        ModelPerformance(
            observed_at_utc=now,
            source="model_monitoring",
            model_version="phase4_ensemble_v3",
            horizon="5d",
            metric="accuracy",
            value=0.62,
            window="30D",
        ),
        ModelPerformance(
            observed_at_utc=now,
            source="model_monitoring",
            model_version="phase4_ensemble_v3",
            horizon="5d",
            metric="brier_score",
            value=0.22,
            window="30D",
        ),
        ModelPerformance(
            observed_at_utc=now,
            source="model_monitoring",
            model_version="phase4_ensemble_v3",
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
    assert summary.status == "INSUFFICIENT_HISTORY"
    assert summary.metrics[0].horizon == "5d"
    assert summary.metrics[0].window == "30D"
    assert summary.metrics[0].sample_count == 30
    assert summary.metrics[0].accuracy == 0.62
    assert summary.metrics[0].twd_depreciation_recall == 0.67
