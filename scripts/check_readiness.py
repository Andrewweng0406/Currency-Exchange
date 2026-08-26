from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.session import make_session
from app.ops.readiness import run_readiness_checks


def main() -> None:
    session = make_session(settings()["database"]["url"])
    checks = run_readiness_checks(session)
    print(json.dumps([asdict(check) for check in checks], indent=2, ensure_ascii=False))
    if not all(check.ok for check in checks if not check.name.startswith("env:LINE")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
