"""Resolve the dynamic capture track: Frida vs mitm vs UI-only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from privacy_audit.analyzers.dynamic.frida_runner import get_app_frida_tier_entry
from privacy_audit.ingestion.evidence_loader import _slug_for_app


def _read_probe(probe_dir: Path, app_name: str) -> dict[str, Any]:
    path = probe_dir / f"{_slug_for_app(app_name)}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def resolve_capture_tracks(
    app_name: str,
    *,
    dynamic_track: str = "auto",
    enable_frida: bool | None = None,
    enable_mitm: bool | None = None,
    probe_dir: Path | None = None,
    frida_tier_file: Path | None = None,
) -> dict[str, Any]:
    """
    Return resolved flags for dynamic capture.

    dynamic_track: auto | frida | mitm | consent | ui

    Policy (mitm-primary): ``auto`` and ``consent`` enable mitm by default.
    Frida is opt-in via ``enable_frida=True`` or ``dynamic_track=frida``.
    """
    track = (dynamic_track or "auto").strip().lower()
    probe = _read_probe(probe_dir or Path("output/frida_probe"), app_name)
    tier: dict[str, Any] = {}
    if frida_tier_file is not None and frida_tier_file.exists():
        tier = get_app_frida_tier_entry(app_name, frida_tier_file)

    java_bridge = str(probe.get("java_bridge") or tier.get("java_bridge") or "")
    preferred = list(tier.get("preferred_dynamic_tracks") or probe.get("preferred_dynamic_tracks") or [])
    tier_name = str(tier.get("tier") or "")

    use_frida = bool(enable_frida) if enable_frida is not None else False
    use_mitm = bool(enable_mitm) if enable_mitm is not None else False
    resolved = track

    if track in {"auto", "consent"}:
        use_mitm = True
        use_frida = False
        resolved = "mitm"
        if track == "auto" and enable_frida is True:
            use_frida = True
            resolved = "mitm+frida"
    elif track == "frida":
        use_frida = True
        use_mitm = bool(enable_mitm) if enable_mitm is not None else True
        resolved = "mitm+frida" if use_mitm else "frida"
    elif track == "mitm":
        use_frida = False
        use_mitm = True
        resolved = "mitm"
    elif track == "ui":
        use_frida = False
        use_mitm = False
        resolved = "ui"

    if enable_frida is True:
        use_frida = True
        if use_mitm:
            resolved = "mitm+frida"
        else:
            resolved = "frida"
    if enable_mitm is True:
        use_mitm = True
        if use_frida and resolved == "frida":
            resolved = "mitm+frida"
        elif not use_frida:
            resolved = "mitm"

    return {
        "dynamic_track_requested": track,
        "dynamic_track_resolved": resolved,
        "enable_frida": use_frida,
        "enable_mitm": use_mitm,
        "java_bridge": java_bridge,
        "tier": tier_name,
        "preferred_dynamic_tracks": preferred,
        "mitm_primary": use_mitm and not (use_frida and resolved == "frida"),
        "probe_path": str((probe_dir or Path("output/frida_probe")) / f"{_slug_for_app(app_name)}.json"),
    }

