from datetime import datetime, timezone

from app.database.schema import Base, FxPrice, MarketData
from app.features.builder import build_feature_frame
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def test_build_feature_frame_from_minimal_data():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for day in range(1, 31):
            observed = datetime(2026, 1, day, tzinfo=timezone.utc)
            session.add(
                FxPrice(
                    observed_at_utc=observed,
                    source="test",
                    pair="USD/TWD",
                    open=30 + day,
                    high=31 + day,
                    low=29 + day,
                    close=30 + day,
                    volume=None,
                )
            )
            session.add(
                MarketData(
                    observed_at_utc=observed,
                    source="test",
                    symbol="DXY",
                    open=None,
                    high=None,
                    low=None,
                    close=100 + day,
                    volume=None,
                )
            )
        session.commit()
        features = build_feature_frame(session)

    assert not features.empty
    assert "USDTWD_RETURN_1D" in features.columns
    assert "USDTWD_RSI_14D" in features.columns
    assert "DXY_RETURN_1D" in features.columns


def test_market_features_mark_stale_forward_filled_data_missing():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for day in range(1, 12):
            observed = datetime(2026, 1, day, tzinfo=timezone.utc)
            session.add(
                FxPrice(
                    observed_at_utc=observed,
                    source="test",
                    pair="USD/TWD",
                    open=30 + day,
                    high=31 + day,
                    low=29 + day,
                    close=30 + day,
                    volume=None,
                )
            )
        session.add(
            MarketData(
                observed_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
                source="test",
                symbol="DXY",
                open=None,
                high=None,
                low=None,
                close=100,
                volume=None,
            )
        )
        session.commit()
        features = build_feature_frame(session)

    stale_row = features.loc[features["date"] == datetime(2026, 1, 9, tzinfo=timezone.utc)].iloc[0]
    fresh_row = features.loc[features["date"] == datetime(2026, 1, 8, tzinfo=timezone.utc)].iloc[0]
    assert fresh_row["DXY_DATA_MISSING"] == 0
    assert stale_row["DXY_DATA_MISSING"] == 1
    assert stale_row["DXY_CLOSE"] != stale_row["DXY_CLOSE"]
