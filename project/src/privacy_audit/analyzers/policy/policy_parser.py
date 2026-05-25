"""Policy parser for declared personal information claims."""

from __future__ import annotations

from pathlib import Path


def load_personal_info_taxonomy(config_path: Path) -> dict[str, dict[str, list[str]]]:
    """Load a minimal taxonomy from a simple YAML file.

    This loader intentionally supports the current project schema only:
    top-level key -> aliases/permissions/apis list.
    """
    lines = config_path.read_text(encoding="utf-8").splitlines()
    taxonomy: dict[str, dict[str, list[str]]] = {}
    current_key: str | None = None

    for raw in lines:
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
