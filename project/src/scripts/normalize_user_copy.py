"""
Normalize Access audit user export copies under input/user_copy/ to flat JSON files.

The access CLI expects --copy-file to be a single JSON object per app.
Raw exports may be nested folders, .txt key-value dumps, or .doc (manual step).

Usage (from project/src):
  python scripts/normalize_user_copy.py
  python scripts/normalize_user_copy.py --input-dir input/user_copy --output-dir input/user_copy_normalized
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]

_KV_RE = re.compile(r"^([^:：]+)[:：]\s*(.*)$")


def _parse_kv_text(text: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _KV_RE.match(line)
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        if key:
            data[key] = value
    return data


def _find_copy_sources(app_dir: Path) -> list[Path]:
    sources: list[Path] = []
    for pattern in ("*.json", "*.txt"):
        sources.extend(sorted(app_dir.rglob(pattern)))
    return sources


def normalize_app_copy(app_name: str, app_dir: Path) -> tuple[dict[str, str], list[str]]:
    """Return (payload, warnings)."""
    warnings: list[str] = []
    for doc in sorted(app_dir.rglob("*.doc")) + sorted(app_dir.rglob("*.docx")):
        warnings.append(f"{app_name}: skip binary {doc.relative_to(app_dir)} — export to .txt or .json manually")

    sources = _find_copy_sources(app_dir)
    if not sources:
        warnings.append(f"{app_name}: no .json/.txt found under {app_dir}")
        return {}, warnings

    merged: dict[str, str] = {}
    for path in sources:
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                warnings.append(f"{app_name}: invalid JSON {path.name}: {exc}")
                continue
            if isinstance(payload, dict):
                for k, v in payload.items():
                    merged[str(k)] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            else:
                warnings.append(f"{app_name}: JSON root is not object in {path.name}")
        elif path.suffix.lower() == ".txt":
            parsed = _parse_kv_text(path.read_text(encoding="utf-8", errors="replace"))
            merged.update(parsed)
        if merged:
            merged["_source_file"] = str(path.relative_to(app_dir)).replace("\\", "/")

    if not merged:
        warnings.append(f"{app_name}: no parseable fields from {len(sources)} file(s)")
    return merged, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize user_copy exports to access JSON")
    parser.add_argument("--input-dir", default="input/user_copy", help="Raw copy tree")
    parser.add_argument(
        "--output-dir",
        default="input/user_copy_normalized",
        help="Flat JSON output: <App名>.json",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = _SRC / input_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _SRC / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"input dir not found: {input_dir}")
        return 1

    all_warnings: list[str] = []
    written = 0
    for app_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        app_name = app_dir.name
        payload, warnings = normalize_app_copy(app_name, app_dir)
        all_warnings.extend(warnings)
        if not payload:
            continue
        out_path = output_dir / f"{app_name}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote fields={len(payload)}")
        written += 1

    # Also accept flat files directly under input_dir (e.g. 小荷健康.json)
    for path in sorted(input_dir.glob("*.json")):
        out_path = output_dir / path.name
        out_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print("Copied flat json")
        written += 1

    if all_warnings:
        print(f"WARN: {len(all_warnings)} message(s) — see stderr")
        import sys

        for msg in all_warnings:
            sys.stderr.buffer.write((msg + "\n").encode("utf-8", errors="backslashreplace"))

    print(f"Done: {written} app JSON file(s)")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
