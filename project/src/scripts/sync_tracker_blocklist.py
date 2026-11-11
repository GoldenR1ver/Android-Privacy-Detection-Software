"""
Sync tracker blocklist from Exodus Privacy API into input/tracker_blocklist/.

Usage (from project/src):
  python scripts/sync_tracker_blocklist.py
  python scripts/sync_tracker_blocklist.py --offline   # validate bundled files only
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from privacy_audit.analyzers.static.tracker_blocklist_loader import (  # noqa: E402
    build_domain_index,
    default_blocklist_dir,
    normalize_tracker_entries,
)

EXODUS_API = "https://reports.exodus-privacy.eu.org/api/trackers"


def _fetch_exodus(timeout_sec: float = 60.0) -> dict:
    req = urllib.request.Request(
        EXODUS_API,
        headers={"User-Agent": "Android-Privacy-Detection-Software/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sync_blocklist(output_dir: Path, *, offline: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    sdk_path = output_dir / "sdk_signatures.json"
    domains_path = output_dir / "trackers_domains.json"

    if offline:
        if not sdk_path.exists():
            raise FileNotFoundError(f"Missing {sdk_path}")
        data = json.loads(sdk_path.read_text(encoding="utf-8"))
        trackers = normalize_tracker_entries(data)
        if not trackers and isinstance(data.get("trackers"), list):
            trackers = [t for t in data["trackers"] if isinstance(t, dict)]
        domains = {
            "schema_version": 1,
            "source": str(data.get("source") or "curated_bootstrap"),
            "updated_at": data.get("updated_at") or datetime.now(timezone.utc).isoformat(),
            "domain_count": 0,
            "domains": build_domain_index(trackers),
        }
        domains["domain_count"] = len(domains["domains"])
        domains_path.write_text(
            json.dumps(domains, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "mode": "offline",
            "tracker_count": len(trackers),
            "domain_count": domains["domain_count"],
            "sdk_path": str(sdk_path),
            "domains_path": str(domains_path),
        }

    raw = _fetch_exodus()
    trackers = normalize_tracker_entries(raw)
    bundle = {
        "schema_version": 1,
        "source": "exodus",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "api_url": EXODUS_API,
        "tracker_count": len(trackers),
        "trackers": trackers,
    }
    sdk_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    domains = {
        "schema_version": 1,
        "source": "exodus",
        "updated_at": bundle["updated_at"],
        "domain_count": 0,
        "domains": build_domain_index(trackers),
    }
    domains["domain_count"] = len(domains["domains"])
    domains_path.write_text(json.dumps(domains, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "mode": "sync",
        "tracker_count": len(trackers),
        "domain_count": domains["domain_count"],
        "sdk_path": str(sdk_path),
        "domains_path": str(domains_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync Exodus tracker blocklist")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_blocklist_dir(),
        help="Output directory (default: input/tracker_blocklist)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only validate existing bundled blocklist",
    )
    args = parser.parse_args()

    result = sync_blocklist(args.output_dir, offline=args.offline)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
