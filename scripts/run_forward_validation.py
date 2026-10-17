"""Record prospective forecasts and score matured outcomes; never sends messages."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.session import make_session
from app.monitoring.forward_validation import (
    capture_forward_snapshots, evaluate_forward_outcomes, forward_validation_summary,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    with make_session(settings()["database"]["url"]) as session:
        result = {}
        if not args.status_only:
            result["capture"] = capture_forward_snapshots(session)
            result["evaluation"] = evaluate_forward_outcomes(session)
        result["summary"] = forward_validation_summary(session)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if not args.status_only and (result.get("capture", {}).get("missing_horizons") or result.get("capture", {}).get("reason")):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
