"""
Generate per-app track matrix: java_bridge | dynamic_tracks | lab outputs (R4).

Usage (from project/src):
  python scripts/generate_app_track_matrix.py
  python scripts/generate_app_track_matrix.py --output output/app_track_matrix.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from privacy_audit.ingestion.evidence_loader import _slug_for_app  # noqa: E402
from privacy_audit.workflows.capture_track import resolve_capture_tracks  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _has_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def build_app_track_matrix(
    *,
    tier_file: Path,
    probe_dir: Path,
    static_dir: Path,
    dynamic_dir: Path,
    disclosure_dir: Path,
    consent_dir: Path,
    access_dir: Path,
) -> dict[str, Any]:
    tier_data = _read_json(tier_file)
    apps: dict[str, Any] = dict(tier_data.get("apps") or {})
    rows: list[dict[str, Any]] = []

    for app_name in sorted(apps.keys()):
        slug = _slug_for_app(app_name)
        probe = _read_json(probe_dir / f"{slug}.json")
        plan = resolve_capture_tracks(
            app_name,
            dynamic_track="auto",
            probe_dir=probe_dir,
            frida_tier_file=tier_file,
        )
        static_path = static_dir / slug / "static_evidence.json"
        if not static_path.exists():
            static_path = static_dir / app_name / "static_evidence.json"
        dynamic_path = dynamic_dir / app_name / "dynamic_evidence.json"
        static_data = _read_json(static_path)
        dynamic_data = _read_json(dynamic_path)
        network = dynamic_data.get("network_result") or {}

        rows.append(
            {
                "app_name": app_name,
                "tier": plan.get("tier") or apps[app_name].get("tier", ""),
                "java_bridge": plan.get("java_bridge") or probe.get("java_bridge", ""),
                "dynamic_track_auto": plan.get("dynamic_track_resolved", ""),
                "preferred_dynamic_tracks": plan.get("preferred_dynamic_tracks", []),
                "embedded_tracker_count": len(static_data.get("embedded_trackers") or []),
                "mitm_flow_count": network.get("flow_count", dynamic_data.get("summary", {}).get("mitm_flow_count", 0)),
                "tracker_hit_count": (network.get("summary") or {}).get("tracker_hit_count", 0),
                "frida_runtime_events": dynamic_data.get("summary", {}).get("frida_runtime_events", 0),
                "mitm_blocked": network.get("mitm_blocked"),
                "has_static_evidence": _has_file(static_path),
                "has_dynamic_evidence": _has_file(dynamic_path),
                "has_disclosure": _has_file(disclosure_dir / f"{app_name}_disclosure_report.json"),
                "has_consent": _has_file(consent_dir / f"{app_name}_consent_result.json"),
                "has_access": _has_file(access_dir / f"{app_name}_access_result.json"),
            }
        )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "app_count": len(rows),
        "apps": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate app track coverage matrix (R4)")
    parser.add_argument("--output", type=Path, default=_SRC / "output" / "app_track_matrix.json")
    parser.add_argument("--tier-file", type=Path, default=_SRC / "input" / "frida_tier.json")
    parser.add_argument("--probe-dir", type=Path, default=_SRC / "output" / "frida_probe")
    args = parser.parse_args()

    matrix = build_app_track_matrix(
        tier_file=args.tier_file,
        probe_dir=args.probe_dir,
        static_dir=_SRC / "output" / "static",
        dynamic_dir=_SRC / "output" / "dynamic",
        disclosure_dir=_SRC / "output" / "disclosure",
        consent_dir=_SRC / "output" / "consent",
        access_dir=_SRC / "output" / "access",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "app_count": matrix["app_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

