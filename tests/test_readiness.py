from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.schema import Base, Feature
from app.ops.readiness import run_readiness_checks


def test_readiness_reports_missing_tables():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        checks = run_readiness_checks(session)
    assert any(check.name == "table:features" and not check.ok for check in checks)


def test_readiness_freshness_detail():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Feature(observed_at_utc=datetime.now(timezone.utc), source="test", feature_set="x", values_json="{}"))
        session.commit()
        checks = run_readiness_checks(session)
    assert any(check.name == "freshness:features" and check.detail.startswith("latest") for check in checks)
