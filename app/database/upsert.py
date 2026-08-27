from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Type

from sqlalchemy import UniqueConstraint, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

POSTGRES_BATCH_SIZE = 5_000


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
    clean_rows = [_clean_row(row) for row in rows]
    if not clean_rows:
        return 0
    if session.bind and session.bind.dialect.name == "postgresql" and _has_unique_key(model, key_fields):
        return _upsert_postgresql(session, model, clean_rows, key_fields)
    return _upsert_row_by_row(session, model, clean_rows, key_fields)


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (_as_py_datetime(value) if key.endswith("_utc") or key in {"observed_at_utc", "release_time_utc"} else value)
        for key, value in row.items()
    }


def _has_unique_key(model: Type, key_fields: tuple[str, ...]) -> bool:
    expected = set(key_fields)
    for constraint in model.__table__.constraints:
        if isinstance(constraint, UniqueConstraint) and {column.name for column in constraint.columns} == expected:
            return True
    return False


def _upsert_postgresql(session: Session, model: Type, rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> int:
    try:
        for start in range(0, len(rows), POSTGRES_BATCH_SIZE):
            batch = rows[start : start + POSTGRES_BATCH_SIZE]
            table = model.__table__
            statement = pg_insert(table).values(batch)
            update_fields = {
                column.name: getattr(statement.excluded, column.name)
                for column in table.columns
                if column.name not in {"id", "created_at_utc", *key_fields} and column.name in batch[0]
            }
            statement = statement.on_conflict_do_update(index_elements=list(key_fields), set_=update_fields)
            session.execute(statement)
        session.commit()
    except Exception:
        session.rollback()
        raise
    return len(rows)


def _upsert_row_by_row(
    session: Session,
    model: Type,
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> int:
    count = 0
    try:
        for row in rows:
            filters = [getattr(model, field) == row[field] for field in key_fields]
            existing = session.execute(select(model).where(*filters)).scalar_one_or_none()
            if existing:
                for key, value in row.items():
                    setattr(existing, key, value)
            else:
                session.add(model(**row))
            count += 1
        session.commit()
    except Exception:
        session.rollback()
        raise
    return count
