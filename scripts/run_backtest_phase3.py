from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.backtesting.report import phase3_summary
from app.config import settings
from app.database.session import make_session


def main() -> None:
    session = make_session(settings()["database"]["url"])
    print(json.dumps(phase3_summary(session), indent=2, ensure_ascii=False, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
