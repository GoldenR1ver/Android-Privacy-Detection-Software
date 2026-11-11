"""Sync frida_tier.json from probe_hook output (R4)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tier_from_java_bridge(java_bridge: str) -> tuple[str, bool, list[str]]:
    bridge = (java_bridge or "").strip().lower()
    if bridge == "never_ready":
        return (
            "network_static_only",
            False,
            ["static_tracker", "mitm", "ui"],
        )
    if bridge in {"ok", "partial"}:
        return ("tier2", True, ["frida_java", "frida_network", "ui"])
    if bridge == "timeout":
        return ("tier2", False, ["mitm", "ui", "static_tracker"])
    return ("unknown", False, ["ui", "static_tracker"])


def apply_probe_to_tier_entry(probe: dict[str, Any]) -> dict[str, Any]:
    """Build tier fields from a single probe JSON record."""
    java_bridge = str(probe.get("java_bridge") or "unknown")
    tier, enable_consent, tracks = _tier_from_java_bridge(java_bridge)
    if probe.get("runtime_event_count", 0) and java_bridge == "ok":
        tier = "tier1"
        enable_consent = True
    entry: dict[str, Any] = {
        "tier": tier,
        "java_bridge": java_bridge,
        "enable_frida_consent": enable_consent,
        "preferred_dynamic_tracks": list(probe.get("preferred_dynamic_tracks") or tracks),
        "probe_updated_at": probe.get("probe_at_utc") or _utc_now(),
        "probe_success": bool(probe.get("success")),
        "probe_mode": probe.get("mode", ""),
    }
    if tier == "network_static_only":
        entry["preferred_launch_mode"] = "spawn"
        entry["spawn_fallback_attach"] = True
    return entry


def sync_frida_tier_from_probe_dir(
    tier_file: Path,
    probe_dir: Path,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Merge all ``output/frida_probe/*.json`` into ``input/frida_tier.json``."""
    if not tier_file.exists():
        raise FileNotFoundError(tier_file)
    data = json.loads(tier_file.read_text(encoding="utf-8"))
    apps: dict[str, Any] = dict(data.get("apps") or {})
    updated: list[str] = []
    missing_probe: list[str] = []

    for app_name, entry in list(apps.items()):
        slug_path = probe_dir / f"{_slug_from_app_name(app_name, probe_dir)}.json"
        if not slug_path.exists():
            # try direct slug files matching app
            from privacy_audit.ingestion.evidence_loader import _slug_for_app

            slug_path = probe_dir / f"{_slug_for_app(app_name)}.json"
        if not slug_path.exists():
            missing_probe.append(app_name)
            continue
        probe = json.loads(slug_path.read_text(encoding="utf-8"))
        if not isinstance(probe, dict):
            continue
        merged = dict(entry) if isinstance(entry, dict) else {}
        merged.update(apply_probe_to_tier_entry(probe))
        apps[app_name] = merged
        updated.append(app_name)

    data["apps"] = apps
    data["probe_sync_at_utc"] = _utc_now()
    if write:
        tier_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "tier_file": str(tier_file),
        "probe_dir": str(probe_dir),
        "updated_apps": updated,
        "missing_probe": missing_probe,
        "updated_count": len(updated),
    }


def _slug_from_app_name(app_name: str, probe_dir: Path) -> str:
    from privacy_audit.ingestion.evidence_loader import _slug_for_app

    return _slug_for_app(app_name)

