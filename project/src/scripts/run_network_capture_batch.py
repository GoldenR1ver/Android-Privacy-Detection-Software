"""
Batch dynamic capture with mitm network track for java_never_ready / network_static_only apps.

Usage (from project/src):
  python scripts/run_network_capture_batch.py --dry-run
  python scripts/run_network_capture_batch.py --real-run --apps 钓鱼人,韭研公社
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from privacy_audit.workflows.dynamic_capture import run_dynamic_capture  # noqa: E402


def _default_apps() -> list[str]:
    tier_file = _SRC / "input" / "frida_tier.json"
    if not tier_file.exists():
        return ["钓鱼人", "韭研公社"]
    data = json.loads(tier_file.read_text(encoding="utf-8"))
    apps = data.get("apps") or {}
    selected: list[str] = []
    for name, entry in apps.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("java_bridge") == "never_ready":
            selected.append(name)
        elif entry.get("tier") == "network_static_only":
            selected.append(name)
        elif "mitm" in (entry.get("preferred_dynamic_tracks") or []):
            selected.append(name)
    return selected or ["钓鱼人", "韭研公社"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch dynamic capture mitm track (R3)")
    parser.add_argument("--apps", default="", help="Comma-separated app names (default: tier mitm candidates)")
    parser.add_argument("--output-dir", default="output/network", type=Path)
    parser.add_argument("--real-run", action="store_true")
    parser.add_argument("--mitm-flows-file", type=Path, help="Replay flows JSON for dry-run validation")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    app_names = [a.strip() for a in args.apps.split(",") if a.strip()] or _default_apps()
    packages_file = _SRC / "input" / "app_packages.json"
    packages = json.loads(packages_file.read_text(encoding="utf-8")) if packages_file.exists() else {}

    reports: list[dict] = []
    for app_name in app_names:
        pkg_entry = packages.get(app_name) or {}
        if isinstance(pkg_entry, dict):
            package = str(pkg_entry.get("package") or "")
        else:
            package = str(pkg_entry)
        if not package:
            print(f"skip {app_name}: no package in app_packages.json")
            continue
        pii_file = _SRC / "input" / "pii_text" / f"{app_name}.txt"
        result = run_dynamic_capture(
            app_name=app_name,
            app_package=package,
            taxonomy_file=_SRC / "privacy_audit" / "config" / "personal_info.yaml",
            pii_text_file=pii_file if pii_file.exists() else None,
            output_dir=args.output_dir,
            dry_run=not args.real_run,
            enable_frida=False,
            enable_mitm=True,
            dynamic_track="mitm",
            skip_preflight=args.skip_preflight,
            probe_dir=_SRC / "output" / "frida_probe",
            tracker_blocklist_dir=_SRC / "input" / "tracker_blocklist",
            mitm_flows_file=args.mitm_flows_file,
        )
        reports.append(
            {
                "app_name": app_name,
                "report_path": result.get("report_path"),
                "tracker_hits": (result.get("network_result") or {}).get("summary", {}).get(
                    "tracker_hit_count", 0
                ),
            }
        )
        print(json.dumps(reports[-1], ensure_ascii=False))

    summary_path = args.output_dir / "mitm_batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
