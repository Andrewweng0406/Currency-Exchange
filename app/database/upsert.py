from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Type

from sqlalchemy import select
from sqlalchemy.orm import Session


def _as_py_datetime(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value


def upsert_rows(
    session: Session,
    model: Type,
    rows: Iterable[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> int:
    count = 0
    for row in rows:
        clean = {k: (_as_py_datetime(v) if k.endswith("_utc") or k in {"observed_at_utc", "release_time_utc"} else v) for k, v in row.items()}
        filters = [getattr(model, field) == clean[field] for field in key_fields]
        existing = session.execute(select(model).where(*filters)).scalar_one_or_none()
        if existing:
            for key, value in clean.items():
                setattr(existing, key, value)
        else:
            session.add(model(**clean))
        count += 1
    session.commit()
    return count
