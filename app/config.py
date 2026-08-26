from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        body = value[2:-1]
        if ":-" in body:
            key, default = body.split(":-", 1)
            return os.getenv(key, default)
        return os.getenv(body, "")
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    with Path(path).open("r", encoding="utf-8") as f:
        return _expand_env(yaml.safe_load(f) or {})


def settings() -> dict[str, Any]:
    return load_yaml(ROOT / "config" / "settings.yaml")


def risk_policy() -> dict[str, Any]:
    return load_yaml(ROOT / "config" / "risk_policy.yaml")
