from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.backtesting.strategy_compare import compare_exchange_strategies
from app.config import settings
from app.database.session import make_session


def main() -> None:
    session = make_session(settings()["database"]["url"])
    result = compare_exchange_strategies(session)
    print(json.dumps([asdict(item) for item in result], indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
