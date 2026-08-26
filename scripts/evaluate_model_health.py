from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.session import make_session
from app.monitoring.model_health import evaluate_matured_predictions


def main() -> None:
    session = make_session(settings()["database"]["url"])
    print(evaluate_matured_predictions(session))


if __name__ == "__main__":
    main()
