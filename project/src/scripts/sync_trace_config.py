"""
Sync PMA-style privacy_trace_methods.txt with personal_info.yaml api_targets.

Usage (from project/src):
  python scripts/sync_trace_config.py --check
  python scripts/sync_trace_config.py --write-txt
  python scripts/sync_trace_config.py --write-yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from privacy_audit.config.trace_config import (  # noqa: E402
    build_txt_from_yaml,
    check_trace_yaml_consistency,
    default_taxonomy_file,
    default_trace_methods_file,
    merge_txt_into_yaml,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync PMA trace list with personal_info.yaml")
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=default_trace_methods_file(),
        help="privacy_trace_methods.txt path",
    )
    parser.add_argument(
        "--taxonomy-file",
        type=Path,
        default=default_taxonomy_file(),
        help="personal_info.yaml path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate txt vs yaml consistency (default if no write flag)",
    )
    parser.add_argument(
        "--write-txt",
        action="store_true",
        help="Regenerate privacy_trace_methods.txt from yaml api_targets",
    )
    parser.add_argument(
        "--write-yaml",
        action="store_true",
        help="Merge txt-only entries into personal_info.yaml api_targets",
    )
    args = parser.parse_args()

    if args.write_txt:
        content = build_txt_from_yaml(args.taxonomy_file)
        args.trace_file.parent.mkdir(parents=True, exist_ok=True)
        args.trace_file.write_text(content, encoding="utf-8")
        result = {"action": "write_txt", "path": str(args.trace_file), "bytes": len(content)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.write_yaml:
        result = merge_txt_into_yaml(args.trace_file, args.taxonomy_file, write=True)
        result["action"] = "write_yaml"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    report = check_trace_yaml_consistency(args.trace_file, args.taxonomy_file)
    report["action"] = "check"
    report["trace_file"] = str(args.trace_file)
    report["taxonomy_file"] = str(args.taxonomy_file)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("consistent") else 1


if __name__ == "__main__":
    raise SystemExit(main())
