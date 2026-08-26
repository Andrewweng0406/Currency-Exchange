from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.session import make_session
from app.ops.data_quality import data_coverage_report


def main() -> None:
    session = make_session(settings()["database"]["url"])
    print(json.dumps(data_coverage_report(session), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
