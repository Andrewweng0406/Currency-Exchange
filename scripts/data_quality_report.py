from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.session import make_session
from app.ops.data_quality import data_coverage_report, summarize_data_quality


def main() -> None:
    session = make_session(settings()["database"]["url"])
    report = data_coverage_report(session)
    report["summary"] = summarize_data_quality(report).__dict__
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
