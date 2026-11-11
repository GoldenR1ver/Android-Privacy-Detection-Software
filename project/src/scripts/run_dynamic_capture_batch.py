"""
Batch dynamic with --dynamic-track auto (mitm-primary; Frida opt-in via --enable-frida).

Usage (from project/src):
  python scripts/run_dynamic_capture_batch.py --dry-run
  python scripts/run_dynamic_capture_batch.py --real-run --apps 钓鱼人,小荷健康
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from privacy_audit.workflows.capture_track import resolve_capture_tracks  # noqa: E402
from privacy_audit.workflows.dynamic_capture import run_dynamic_capture  # noqa: E402


def _load_apps(packages_file: Path) -> dict[str, str]:
    if not packages_file.exists():
        return {}
    raw = json.loads(packages_file.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for name, entry in raw.items():
        if isinstance(entry, dict):
            pkg = str(entry.get("package") or "")
        else:
            pkg = str(entry)
        if pkg:
            out[str(name)] = pkg
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch dynamic dynamic-track auto (R4)")
    parser.add_argument("--apps", default="", help="Comma-separated app names (default: all in app_packages.json)")
    parser.add_argument("--output-dir", type=Path, default=_SRC / "output" / "dynamic")
    parser.add_argument("--real-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--probe-dir", type=Path, default=_SRC / "output" / "frida_probe")
    parser.add_argument("--mitm-adb-proxy-host", default="172.27.154.110")
    parser.add_argument(
        "--adb-serial",
        default=os.environ.get("ANDROID_SERIAL", "127.0.0.1:16384"),
        help="adb -s serial when multiple devices",
    )
    args = parser.parse_args()

    packages = _load_apps(_SRC / "input" / "app_packages.json")
    app_names = [a.strip() for a in args.apps.split(",") if a.strip()] or list(packages.keys())
    tier_file = _SRC / "input" / "frida_tier.json"
    reports: list[dict] = []

    for app_name in app_names:
        package = packages.get(app_name, "")
        if not package:
            print(f"skip {app_name}: no package")
            continue
        plan = resolve_capture_tracks(
            app_name,
            dynamic_track="auto",
            probe_dir=args.probe_dir,
            frida_tier_file=tier_file,
        )
        pii_file = _SRC / "input" / "pii_text" / f"{app_name}.txt"
        result = run_dynamic_capture(
            app_name=app_name,
            app_package=package,
            taxonomy_file=_SRC / "privacy_audit" / "config" / "personal_info.yaml",
            pii_text_file=pii_file if pii_file.exists() else None,
            output_dir=args.output_dir,
            dry_run=not args.real_run,
            dynamic_track="auto",
            enable_frida=plan["enable_frida"],
            enable_mitm=plan["enable_mitm"],
            skip_preflight=args.skip_preflight,
            probe_dir=args.probe_dir,
            frida_tier_file=tier_file,
            tracker_blocklist_dir=_SRC / "input" / "tracker_blocklist",
            mitm_adb_proxy_host=args.mitm_adb_proxy_host,
            adb_serial=args.adb_serial.strip() or None,
        )
        row = {
            "app_name": app_name,
            "package": package,
            "track_plan": plan,
            "report_path": result.get("report_path"),
            "mitm_flow_count": (result.get("mitm_result") or {}).get("flow_count", 0),
            "tracker_hits": (result.get("network_result") or {}).get("summary", {}).get(
                "tracker_hit_count", 0
            ),
            "frida_runtime_events": (result.get("frida_result") or {}).get("runtime_event_count", 0),
        }
        reports.append(row)
        print(json.dumps(row, ensure_ascii=False))

    summary_path = args.output_dir / "dynamic_capture_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

