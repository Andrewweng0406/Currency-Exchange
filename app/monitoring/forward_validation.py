"""Append-only prospective evidence, separate from historical model diagnostics."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from numbers import Real

import joblib
import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.backtesting.dataset import load_feature_frame
from app.backtesting.metrics import classification_metrics
from app.config import ROOT, risk_policy, settings
from app.database.schema import ForwardOutcome, ForwardSnapshot, FxPrice, Prediction
from app.models.training import HORIZON_DAYS, MODEL_VERSION
from app.ops.bank_quotes import latest_bank_quote


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value: str | bytes):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _utc(value):
    return pd.to_datetime(value, utc=True)


def _runtime_packages():
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")}
    try:
        distribution = "xgboost-cpu"
        packages["xgboost"] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        distribution = "xgboost"
        packages["xgboost"] = importlib.metadata.version(distribution)
    packages["xgboost_distribution"] = distribution
    return packages


def protocol_spec():
    files = ["features/builder.py", "features/availability.py", "features/indicators.py",
             "models/training.py", "models/baselines.py", "backtesting/dataset.py",
             "backtesting/splits.py", "backtesting/metrics.py", "backtesting/uncertainty.py",
             "providers/cbc.py", "providers/fred.py", "providers/yahoo.py", "providers/twse.py",
             "providers/treasury.py", "providers/hncb.py", "monitoring/forward_validation.py"]
    config = settings()
    return {
        "version": "prospective_v1", "feature_set": "daily_v3", "model_version": MODEL_VERSION,
        "horizons": HORIZON_DAYS, "target": "official CBC close; up vs flat_or_down",
        "available_hour_utc": 9, "issue_deadline": "next weekday 09:00 UTC; conservative on holidays",
        "scoring": "Brier and AUC; same rows vs coin 0.5 and training-period prior",
        "outcome_policy": "first recorded official outcome; later revisions never overwrite",
        "promotion": "monitor only; no automatic live strategy activation",
        "recipe_sha256": {name: digest((ROOT / "app" / name).read_bytes()) for name in files},
        "packages": _runtime_packages(),
        "strategy": risk_policy()["strategy_backtest"]["model_timing_once"],
        "opportunity_weights": risk_policy()["opportunity_score"]["weights"],
        "feature_settings": config.get("features", {}),
        "data_sources": {
            "fred_series": config["providers"]["fred"]["series"],
            "yahoo_symbols": config["providers"]["yahoo"]["symbols"],
            "bank": {name: config["providers"]["bank"].get(name) for name in ("name", "provider", "fallback_provider", "currency")},
        },
    }


def _insert_once(session, row, existing_query):
    try:
        with session.begin_nested():
            session.add(row)
            session.flush()
        return True
    except IntegrityError:
        if session.execute(existing_query).first() is None:
            raise
        return False


def _archive_model(archive, model_hash, content, suffix=".joblib"):
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive / f"{model_hash}{suffix}"
    with tempfile.NamedTemporaryFile(dir=archive, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        try:
            os.link(temporary, destination)  # Publish a complete file without overwriting evidence.
        except FileExistsError:
            pass
        if digest(destination.read_bytes()) != model_hash:
            raise ValueError("Archived file fingerprint mismatch")
    finally:
        temporary.unlink(missing_ok=True)


def capture_forward_snapshots(session, now=None, archive_dir: Path | None = None):
    now = _utc(now if now is not None else datetime.now(timezone.utc))
    features = load_feature_frame(session, "daily_v3")
    if features.empty:
        return {"captured": 0, "reason": "no_features"}
    row = features.iloc[-1]
    origin = _utc(row["date"]).normalize()
    deadline = origin + pd.offsets.BDay(1) + pd.Timedelta(hours=9)
    if row.get("source") != "cbc_taiwan" or not origin + pd.Timedelta(hours=9) <= now < deadline:
        return {"captured": 0, "reason": "outside_prospective_window"}
    newer_close = session.execute(select(FxPrice.id).where(
        FxPrice.pair == "USD/TWD", FxPrice.source == "cbc_taiwan",
        FxPrice.observed_at_utc > origin.to_pydatetime(),
        FxPrice.observed_at_utc <= now.to_pydatetime(),
    ).limit(1)).first()
    if newer_close:
        return {"captured": 0, "reason": "features_behind_official_close"}
    inputs = {key: (None if pd.isna(value) else float(value)) for key, value in row.items()
              if key not in {"date", "source"} and isinstance(value, Real)}
    if not math.isfinite(float(row["USDTWD_CLOSE"])) or float(row["USDTWD_CLOSE"]) <= 0:
        raise ValueError("Official origin price must be finite and positive")
    spec = protocol_spec()
    protocol_json = canonical(spec)
    protocol_hash = digest(protocol_json)
    bank = latest_bank_quote(session, now=now)
    bank_payload = None if bank is None else {
        "bank_name": bank.bank_name, "source": bank.source, "spot_selling": bank.spot_selling,
        "observed_at_utc": _utc(bank.observed_at_utc).isoformat(), "row_id": bank.id,
    }
    archive = archive_dir or ROOT / "models" / "artifacts" / "forward"
    for name, expected_hash in spec.get("recipe_sha256", {}).items():
        content = (ROOT / "app" / name).read_bytes()
        if digest(content) != expected_hash:
            raise ValueError("Recipe changed while registering the prospective protocol")
        _archive_model(archive / "sources", expected_hash, content, suffix=".py")
    _archive_model(archive / "protocols", protocol_hash, protocol_json.encode(), suffix=".json")
    captured, duplicates, missing = 0, 0, []
    for horizon in HORIZON_DAYS:
        query = select(ForwardSnapshot.id).where(
            ForwardSnapshot.protocol_hash == protocol_hash,
            ForwardSnapshot.origin_utc == origin.to_pydatetime(), ForwardSnapshot.horizon == horizon,
        )
        if session.execute(query).first():
            duplicates += 1
            continue
        prediction = session.execute(select(Prediction).where(
            Prediction.model_version == MODEL_VERSION, Prediction.horizon == horizon,
            Prediction.observed_at_utc == origin.to_pydatetime(),
            Prediction.source == "model_training_latest",
        ).limit(1)).scalar_one_or_none()
        path = ROOT / "models" / "artifacts" / f"{MODEL_VERSION}_{horizon}.joblib"
        if prediction is None or not path.exists():
            missing.append(horizon)
            continue
        raw_artifact = path.read_bytes()
        model_hash = digest(raw_artifact)
        # Load the exact bytes being fingerprinted, even if another process
        # atomically replaces the latest artifact while this job is reading.
        from io import BytesIO
        artifact = joblib.load(BytesIO(raw_artifact))
        if artifact.get("feature_set") != "daily_v3" or artifact.get("horizon") != horizon or artifact.get("model_version") != MODEL_VERSION:
            raise ValueError("Artifact does not match the registered data or horizon")
        if any(column not in inputs for column in artifact["feature_columns"]):
            raise ValueError("Forward evidence is missing model input columns")
        if _utc(artifact["trained_at_utc"]) > now:
            raise ValueError("Cannot issue a forecast from a future-trained artifact")
        x = features.tail(1).reindex(columns=artifact["feature_columns"])
        probabilities = {name: float(model.predict_proba(x)[0, 1]) for name, model in artifact["classifiers"].items()}
        p = sum(probabilities[name] * weight for name, weight in artifact["classification_weights"].items())
        prior = float(artifact["training_up_rate"])
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in [p, prior, *probabilities.values()]):
            raise ValueError("Forward probabilities must be finite and between zero and one")
        payload = {
            "origin_close": float(row["USDTWD_CLOSE"]), "feature_set": "daily_v3", "inputs": inputs,
            "model_version": MODEL_VERSION, "model_sha256": model_hash,
            "model_trained_at_utc": artifact["trained_at_utc"], "feature_columns": artifact["feature_columns"],
            "prob_up": p, "coin_flip": 0.5, "historical_rate": prior,
            "component_probabilities": probabilities, "weights": artifact["classification_weights"],
            "bank_quote": bank_payload, "purpose": "prospective market direction; no bank-cost claim",
        }
        serialized = canonical(payload)
        _archive_model(archive, model_hash, raw_artifact)
        snapshot = ForwardSnapshot(
            protocol_hash=protocol_hash, protocol_json=protocol_json, origin_utc=origin.to_pydatetime(),
            horizon=horizon, prediction_id=prediction.id, issued_at_utc=now.to_pydatetime(),
            recorded_at_utc=now.to_pydatetime(), payload_json=serialized, payload_hash=digest(serialized),
        )
        # This is a newly issued paper forecast at NOW, not a retroactive upgrade
        # of the legacy prediction referenced above.
        if _insert_once(session, snapshot, query):
            captured += 1
        else:
            duplicates += 1
    session.commit()
    return {"captured": captured, "duplicates": duplicates, "missing_horizons": missing,
            "protocol_hash": protocol_hash, "origin_utc": origin.isoformat()}


def evaluate_forward_outcomes(session, now=None):
    now = _utc(now if now is not None else datetime.now(timezone.utc))
    existing = set(session.execute(select(ForwardOutcome.snapshot_id)).scalars())
    snapshots = session.execute(select(ForwardSnapshot).order_by(ForwardSnapshot.origin_utc)).scalars().all()
    written = 0
    for snapshot in snapshots:
        if snapshot.id in existing:
            continue
        if digest(snapshot.payload_json) != snapshot.payload_hash or digest(snapshot.protocol_json) != snapshot.protocol_hash:
            raise ValueError("Forward snapshot fingerprint mismatch")
        origin = _utc(snapshot.origin_utc)
        issued = max(_utc(snapshot.issued_at_utc), _utc(snapshot.recorded_at_utc))
        future = session.execute(select(FxPrice).where(
            FxPrice.pair == "USD/TWD", FxPrice.source == "cbc_taiwan",
            FxPrice.observed_at_utc > origin.to_pydatetime(),
            FxPrice.observed_at_utc <= now.normalize().to_pydatetime(),
        ).order_by(FxPrice.observed_at_utc)).scalars().all()
        days = HORIZON_DAYS[snapshot.horizon]
        if len(future) < days:
            continue
        target = future[days - 1]
        first_available = _utc(future[0].observed_at_utc) + pd.Timedelta(hours=9)
        target_available = _utc(target.observed_at_utc) + pd.Timedelta(hours=9)
        if issued >= first_available:
            raise ValueError("Snapshot was recorded after the first future close")
        if target_available > now:
            continue
        close = float(target.close)
        if not math.isfinite(close) or close <= 0:
            raise ValueError("Invalid official outcome price")
        payload = json.loads(snapshot.payload_json)
        realized = close / payload["origin_close"] - 1
        outcome = ForwardOutcome(snapshot_id=snapshot.id, target_utc=target.observed_at_utc,
            evaluated_at_utc=now.to_pydatetime(), close=close, realized_return=realized, target_up=int(realized > 0),
            evidence_json=canonical({"source": "cbc_taiwan", "target_row_id": target.id,
                                     "path": [{"date": _utc(item.observed_at_utc).isoformat(), "close": item.close}
                                              for item in future[:days]]}))
        if _insert_once(session, outcome, select(ForwardOutcome.id).where(ForwardOutcome.snapshot_id == snapshot.id)):
            written += 1
    session.commit()
    return {"outcomes_recorded": written}


def forward_validation_summary(session):
    rows = session.execute(select(ForwardSnapshot).order_by(ForwardSnapshot.recorded_at_utc.desc())).scalars().all()
    outcomes = {row.snapshot_id: row for row in session.execute(select(ForwardOutcome)).scalars()}
    groups = {}
    integrity_errors = 0
    for row in rows:
        if digest(row.payload_json) != row.payload_hash or digest(row.protocol_json) != row.protocol_hash:
            integrity_errors += 1
            continue
        group = groups.setdefault(row.protocol_hash, {"protocol_hash": row.protocol_hash,
            "snapshots": 0, "matured": 0, "pending": 0, "bank_quotes": 0, "metrics": [],
            "first_recorded_at_utc": _utc(row.recorded_at_utc).isoformat(), "_records": {}})
        group["snapshots"] += 1
        group["first_recorded_at_utc"] = min(group["first_recorded_at_utc"], _utc(row.recorded_at_utc).isoformat())
        payload = json.loads(row.payload_json)
        group["bank_quotes"] += int(payload.get("bank_quote") is not None)
        outcome = outcomes.get(row.id)
        group["pending" if outcome is None else "matured"] += 1
        if outcome:
            group["_records"].setdefault(row.horizon, []).append({"y": outcome.target_up, **{
                name: payload[name] for name in ("prob_up", "coin_flip", "historical_rate")}})
    for group in groups.values():
        for horizon, records in group.pop("_records").items():
            frame = pd.DataFrame(records)
            for name in ("prob_up", "coin_flip", "historical_rate"):
                metrics = classification_metrics(frame["y"], frame[name])
                group["metrics"].append({"horizon": horizon, "model": name, "n": len(frame),
                                         "brier": metrics.brier_score,
                                         "auc": metrics.roc_auc if math.isfinite(metrics.roc_auc) else None})
    current_hash = digest(canonical(protocol_spec()))
    return {"status": "INTEGRITY_ERROR" if integrity_errors else ("COLLECTING" if current_hash in groups else "NOT_STARTED"),
            "current_protocol_hash": current_hash,
            "scope": "事前市場方向驗證；銀行報價是證據紀錄，尚非實際換匯成本績效。",
            "automatic_promotion": False, "integrity_errors": integrity_errors, "cohorts": list(groups.values())}
