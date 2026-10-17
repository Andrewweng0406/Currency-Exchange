import json
from datetime import datetime, timezone
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.backtesting.dataset import add_forward_targets, modeling_columns
from app.backtesting.splits import yearly_expanding_splits
from app.database.schema import Base, Feature, FxPrice, Prediction
from app.features.builder import _load_fx, _market_series_with_missing_flag, build_feature_frame
from app.features.indicators import pct_return
from app.monitoring.model_health import evaluate_matured_predictions
from app.providers.base import ProviderError
from app.providers.fred import _csv_text
from app.risk.scoring import _scaled_centered, _risk_components, _confidence


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_unrealized_targets_stay_unknown(horizon):
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=50, tz="UTC"), "USDTWD_CLOSE": np.arange(50) + 30.0})
    out = add_forward_targets(frame, horizons=(horizon,)).frame
    targets = [c for c in out if c.startswith("TARGET_")]
    assert out[targets].tail(horizon).isna().all().all()
    assert out[f"TARGET_UP_{horizon}D"].iloc[:-horizon].eq(1).all()
    assert not set(targets) & set(modeling_columns(out))


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_every_training_label_matures_before_test(horizon):
    frame = pd.DataFrame({"date": pd.bdate_range("2015-01-01", "2023-12-31", tz="UTC"), "USDTWD_CLOSE": 31.0})
    frame = add_forward_targets(frame, horizons=(horizon,)).frame
    end_col = f"TARGET_END_{horizon}D"
    splits = yearly_expanding_splits(frame, label_end_col=end_col)
    assert splits
    for split in splits:
        assert frame.loc[split.train_index, end_col].max() < split.test_start


def test_market_alignment_keeps_source_only_dates():
    dates = pd.to_datetime(["2026-01-01", "2026-01-05"], utc=True)
    source = pd.Series([100.0, 102.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True))
    aligned, missing = _market_series_with_missing_flag(source, dates, 7)
    assert aligned.iloc[-1] == 102
    assert missing.iloc[-1] == 0


def test_returns_do_not_hide_missing_observations():
    out = pct_return(pd.Series([100.0, np.nan, 105.0]), 1)
    assert out.isna().all()


def test_missing_values_do_not_become_maximum_risk():
    assert _scaled_centered(np.nan, 0.015) == 50
    row = pd.Series({"DXY_RETURN_5D": np.nan, "BROAD_USD_INDEX_RETURN_5D": 0.0, "DATA_COMPLETENESS": 0.0})
    parts = _risk_components(row, {})
    assert parts["dxy_momentum"] == 50
    assert parts["data_quality_penalty"] == 100
    assert np.isfinite(_confidence(row, {}))


def test_cbc_is_preferred_over_yahoo():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for source, close in [("yahoo_finance", 32.0), ("cbc_taiwan", 31.0)]:
            session.add(FxPrice(observed_at_utc=datetime(2026, 1, 1), source=source, pair="USD/TWD", close=close))
        session.commit()
        assert _load_fx(session).iloc[0]["close"] == 31
        frame = build_feature_frame(session)
        assert frame.iloc[0]["DATA_COMPLETENESS"] == 0


def test_monitoring_handles_no_matured_predictions():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        session.add(Feature(observed_at_utc=now, source="test", feature_set="daily_v3", values_json=json.dumps({"USDTWD_CLOSE": 31.0})))
        session.add(Prediction(observed_at_utc=now, source="test", model_version="phase4_ensemble_v3", horizon="5d", prob_up=0.6, expected_return=0.001))
        session.commit()
        assert evaluate_matured_predictions(session) == 0


def test_invalid_fred_archive_raises_provider_error():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "no data")
    class Response:
        content = buffer.getvalue()
    with pytest.raises(ProviderError, match="CSV"):
        _csv_text(Response())


def test_future_and_incomplete_yahoo_fx_bars_are_excluded():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    today = pd.Timestamp.now(tz="UTC").normalize()
    with Session(engine) as session:
        for offset in (-1, 0, 1):
            session.add(FxPrice(observed_at_utc=(today + pd.Timedelta(days=offset)).to_pydatetime(), source="yahoo_finance", pair="USD/TWD", close=31.0))
        session.commit()
        frame = _load_fx(session)
        assert len(frame) == 1
        assert frame.iloc[0]["date"] == today - pd.Timedelta(days=1)


def test_weak_models_receive_no_weight_when_baseline_is_better():
    from types import SimpleNamespace
    from app.models.training import _weights_from_scores
    weights = _weights_from_scores({
        "time_series": SimpleNamespace(brier_score=0.23),
        "logistic": SimpleNamespace(brier_score=0.25),
        "xgboost": SimpleNamespace(brier_score=0.28),
    })
    assert weights == {"time_series": 1.0, "logistic": 0.0, "xgboost": 0.0}


def test_model_advice_stays_disabled_without_live_validation():
    from app.ops.advice import guard_model_advice
    from app.ops.data_quality import DataQualitySummary
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        result = guard_model_advice(session, DataQualitySummary("OK", "正常", "資料正常", False, []))
    assert result.blocks_model_advice
    assert result.status == "UNVERIFIED_MODEL"
