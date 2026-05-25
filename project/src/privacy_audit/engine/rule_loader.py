"""Load decision rules from YAML-like config."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_inline_list(value: str) -> list[str]:
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return []
    body = value[1:-1].strip()
    if not body:
        return []
    return [token.strip().strip("\"'") for token in body.split(",") if token.strip()]


def load_decision_rules(path: Path) -> list[dict[str, Any]]:
    """Load simple rules list from YAML file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rules: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or stripped == "rules:":
            continue
        if stripped.startswith("- "):
            if current:
                rules.append(current)
            current = {}
            stripped = stripped[2:].strip()
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                current[key.strip()] = value.strip().strip("\"'")
            continue
        if current is None or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("["):
            current[key] = _parse_inline_list(value)
        else:
            current[key] = value.strip("\"'")

    if current:
        rules.append(current)
    return rules

