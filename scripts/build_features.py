from __future__ import annotations

import sys
from pathlib import Path

import structlog

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.session import make_session
from app.features.builder import persist_features
from app.logging import configure_logging


def main() -> None:
    configure_logging()
    log = structlog.get_logger()
    session = make_session(settings()["database"]["url"])
    result = persist_features(session)
    log.info("features_built", rows=result.rows, start_date=result.start_date, end_date=result.end_date)


if __name__ == "__main__":
    main()
