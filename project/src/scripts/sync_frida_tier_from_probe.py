"""
Sync input/frida_tier.json from output/frida_probe/*.json (R4).

Usage (from project/src):
  python scripts/sync_frida_tier_from_probe.py
  python scripts/sync_frida_tier_from_probe.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from privacy_audit.workflows.tier_sync import sync_frida_tier_from_probe_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync frida_tier.json from probe outputs")
    parser.add_argument(
        "--tier-file",
        type=Path,
        default=_SRC / "input" / "frida_tier.json",
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=_SRC / "output" / "frida_probe",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only, do not write tier file")
    args = parser.parse_args()

    result = sync_frida_tier_from_probe_dir(
        args.tier_file,
        args.probe_dir,
        write=not args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
