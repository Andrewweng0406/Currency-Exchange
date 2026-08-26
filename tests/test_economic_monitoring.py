from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database.schema import Base, EconomicEvent
from app.economic_events.importer import import_events_csv


def test_import_events_csv(tmp_path: Path):
    path = tmp_path / "events.csv"
    path.write_text("event_name,release_time,previous,forecast,actual\nCPI,2026-01-15 08:30:00-05:00,2.9,3.0,3.1\n")
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert import_events_csv(session, str(path)) == 1
        event = session.execute(select(EconomicEvent)).scalar_one()
        assert event.surprise == 0.10000000000000009
