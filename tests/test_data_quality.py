from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.schema import Base, Feature
from app.ops.data_quality import data_coverage_report


def test_data_quality_reports_fail_without_rows():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        report = data_coverage_report(session)
    assert report["overall_status"] == "FAIL"


def test_data_quality_checks_feature_missing_ratio():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            Feature(
                observed_at_utc=datetime.now(timezone.utc),
                source="test",
                feature_set="daily_v1",
                values_json='{"USDTWD_CLOSE": 31.0, "DATA_COMPLETENESS": 1.0}',
            )
        )
        session.commit()
        report = data_coverage_report(session)
    assert any(item["feature"] == "USDTWD_CLOSE" for item in report["feature_quality"])
