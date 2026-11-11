"""Load tracker/SDK blocklists (Exodus-style) for static DEX and network matching."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_DEFAULT_BLOCKLIST_DIR = (
    Path(__file__).resolve().parents[3] / "input" / "tracker_blocklist"
)


def default_blocklist_dir() -> Path:
    return _DEFAULT_BLOCKLIST_DIR


def _split_exodus_signatures(raw: str) -> list[str]:
    """Exodus uses regex-like signatures: com.foo|com\\.bar -> package prefixes."""
    if not raw or not str(raw).strip():
        return []
    parts: list[str] = []
    for chunk in str(raw).split("|"):
        sig = chunk.replace("\\.", ".").strip()
        if not sig or len(sig) < 4:
            continue
        if sig not in parts:
            parts.append(sig)
    return parts


def _parse_network_hosts(raw: str) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    hosts: list[str] = []
    for token in re.split(r"[\s|,;]+", str(raw).strip()):
        host = token.strip().lower().lstrip(".")
        if host and "." in host and host not in hosts:
            hosts.append(host)
    return hosts


@lru_cache(maxsize=4)
def load_sdk_signatures(blocklist_dir: str | None = None) -> dict[str, Any]:
    """Load ``sdk_signatures.json`` (curated or synced from Exodus)."""
    root = Path(blocklist_dir) if blocklist_dir else default_blocklist_dir()
    path = root / "sdk_signatures.json"
    if not path.exists():
        return {"schema_version": 1, "source": "missing", "trackers": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"schema_version": 1, "source": "invalid", "trackers": []}
    trackers = data.get("trackers")
    if not isinstance(trackers, list):
        data["trackers"] = []
    return data


@lru_cache(maxsize=4)
def load_tracker_domains(blocklist_dir: str | None = None) -> dict[str, Any]:
    """Load ``trackers_domains.json`` for network host matching (R3)."""
    root = Path(blocklist_dir) if blocklist_dir else default_blocklist_dir()
    path = root / "trackers_domains.json"
    if not path.exists():
        return {"schema_version": 1, "domains": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"schema_version": 1, "domains": []}
    if not isinstance(data.get("domains"), list):
        data["domains"] = []
    return data


def normalize_tracker_entries(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize Exodus API JSON or internal bundle into tracker dicts."""
    trackers: list[dict[str, Any]] = []
    if "trackers" in raw and isinstance(raw["trackers"], dict):
        for key, entry in raw["trackers"].items():
            if not isinstance(entry, dict):
                continue
            trackers.append(_normalize_one_tracker(entry, fallback_id=str(key)))
    elif isinstance(raw.get("trackers"), list):
        for entry in raw["trackers"]:
            if isinstance(entry, dict):
                trackers.append(_normalize_one_tracker(entry))
    return trackers


def _normalize_one_tracker(entry: dict[str, Any], fallback_id: str = "") -> dict[str, Any]:
    tid = str(entry.get("id") or entry.get("tracker_id") or fallback_id or "")
    name = str(entry.get("name") or "").strip()
    categories = entry.get("categories") or []
    if isinstance(categories, str):
        categories = [categories]
    code_raw = entry.get("code_signatures")
    if code_raw is None:
        code_raw = entry.get("code_signature") or entry.get("codeSignature") or ""
    if isinstance(code_raw, list):
        code_sigs = [str(x).strip() for x in code_raw if str(x).strip()]
    else:
        code_sigs = _split_exodus_signatures(str(code_raw))

    net_raw = entry.get("network_hosts")
    if net_raw is None:
        net_raw = entry.get("network_signature") or entry.get("networkSignature") or ""
    if isinstance(net_raw, list):
        net_hosts = [str(x).strip().lower() for x in net_raw if str(x).strip()]
    else:
        net_hosts = _parse_network_hosts(str(net_raw))

    return {
        "id": tid,
        "name": name,
        "categories": [str(c) for c in categories if c],
        "code_signatures": code_sigs,
        "network_hosts": net_hosts,
        "website": str(entry.get("website") or "").strip(),
    }


def build_domain_index(trackers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten tracker network hosts for mitm matching."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tr in trackers:
        for host in tr.get("network_hosts") or []:
            key = host.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "host": host,
                    "tracker_id": tr.get("id", ""),
                    "name": tr.get("name", ""),
                    "categories": list(tr.get("categories") or []),
                }
            )
    return rows
