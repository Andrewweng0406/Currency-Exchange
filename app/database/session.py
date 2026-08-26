from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import ROOT


def make_engine(database_url: str):
    if database_url.startswith("sqlite:///"):
        db_path = database_url.removeprefix("sqlite:///")
        path = Path(db_path)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite:///{path}"
    return create_engine(database_url, future=True)


def make_session(database_url: str) -> Session:
    engine = make_engine(database_url)
    from app.database.schema import Base

    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, future=True)()
