"""Policy parser for declared personal information claims."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_taxonomy_minimal(text: str) -> dict[str, dict[str, Any]]:
    """Fallback parser for the simple ``field: [a, b, c]`` schema (no PyYAML).

    Only understands single-line string lists; nested block structures such as
    ``api_targets`` are ignored. Kept so the loader still works without PyYAML.
    """
    taxonomy: dict[str, dict[str, Any]] = {}
    current_key: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not raw.startswith(" ") and line.endswith(":"):
            current_key = line[:-1].strip()
            taxonomy[current_key] = {"aliases": [], "permissions": [], "apis": []}
            continue
        if current_key is None or ":" not in line:
            continue
        field, value = line.split(":", 1)
        field = field.strip()
        value = value.strip()
        if not value.startswith("[") or not value.endswith("]"):
            continue
        body = value[1:-1].strip()
        items: list[str] = []
        if body:
            parts = [item.strip() for item in body.split(",")]
            items = [part.strip("\"'") for part in parts if part]
        taxonomy[current_key][field] = items
    return taxonomy


def load_personal_info_taxonomy(config_path: Path) -> dict[str, dict[str, Any]]:
    """Load the personal-info taxonomy from YAML.

    Prefers PyYAML (so nested fields like ``api_targets`` — lists of
    ``{clazz, method}`` / ``{enumerate}`` dicts — are preserved for the
    data-driven Frida monitor). Falls back to a minimal string-list parser when
    PyYAML is unavailable or the file fails to parse.
    """
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return {
                str(key): (value if isinstance(value, dict) else {})
                for key, value in data.items()
            }
    except Exception:
        pass
    return _load_taxonomy_minimal(text)


def parse_declared_pii(policy_text: str, taxonomy: dict[str, dict[str, list[str]]]) -> list[str]:
    """Extract declared personal information types by alias matching."""
    text = policy_text.lower()
    declared: list[str] = []
    for pii_type, meta in taxonomy.items():
        aliases = meta.get("aliases", [])
        candidates = [pii_type, *aliases]
        if any(alias.lower() in text for alias in candidates if alias):
            declared.append(pii_type)
    return sorted(set(declared))
