"""DEX-based embedded tracker/SDK detection (TrackerControl / Exodus style).

Scans APK DEX class names against ``input/tracker_blocklist/sdk_signatures.json``.
Does not require Frida or Java bridge — suitable for apps where runtime hooks fail.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

from privacy_audit.analyzers.static.tracker_blocklist_loader import (
    load_sdk_signatures,
    normalize_tracker_entries,
)

# DEX type descriptors: Lcom/example/Foo;
_DEX_CLASS_RE = re.compile(rb"L([a-zA-Z0-9_$/\x80-\xff]{4,200});")
_MIN_CLASS_LEN = 9  # e.g. Laa/bb/C;


class _SignatureTrie:
    """Prefix trie for package-level code signature lookup."""

    def __init__(self) -> None:
        self._root: dict[str, Any] = {}

    def insert(self, signature: str, tracker: dict[str, Any]) -> None:
        node = self._root
        parts = signature.split(".")
        for part in parts:
            node = node.setdefault(part, {})
        node["$tracker"] = tracker
        node["$signature"] = signature

    def find_longest(self, class_name: str) -> tuple[dict[str, Any], str] | None:
        """Return (tracker, matched_signature) for longest package prefix match."""
        if not class_name or "." not in class_name:
            return None
        node = self._root
        best: tuple[dict[str, Any], str] | None = None
        for part in class_name.split("."):
            if part not in node:
                break
            node = node[part]
            if "$tracker" in node:
                best = (node["$tracker"], str(node.get("$signature", "")))
        return best


def _dex_class_names(apk_path: Path) -> set[str]:
    names: set[str] = set()
    try:
        with zipfile.ZipFile(apk_path, "r") as zf:
            dex_files = [n for n in zf.namelist() if n.endswith(".dex")]
            for dex_name in dex_files:
                data = zf.read(dex_name)
                for match in _DEX_CLASS_RE.finditer(data):
                    internal = match.group(1).decode("utf-8", errors="ignore")
                    if "/" not in internal:
                        continue
                    class_name = internal.replace("/", ".")
                    if len(class_name) < _MIN_CLASS_LEN:
                        continue
                    if not class_name[0].isalpha():
                        continue
                    names.add(class_name)
    except Exception:
        return set()
    return names


def _build_trie(trackers: list[dict[str, Any]]) -> _SignatureTrie:
    trie = _SignatureTrie()
    for tracker in trackers:
        for sig in tracker.get("code_signatures") or []:
            sig = str(sig).strip()
            if len(sig) >= 4 and "." in sig:
                trie.insert(sig, tracker)
    return trie


def scan_embedded_trackers(
    apk_path: Path,
    *,
    blocklist_dir: Path | None = None,
) -> dict[str, Any]:
    """Scan APK for embedded tracker/SDK libraries by DEX class prefixes."""
    blocklist = load_sdk_signatures(
        str(blocklist_dir) if blocklist_dir else None
    )
    trackers = normalize_tracker_entries(blocklist)
    if not trackers and isinstance(blocklist.get("trackers"), list):
        trackers = [t for t in blocklist["trackers"] if isinstance(t, dict)]

    classes = _dex_class_names(apk_path)
    trie = _build_trie(trackers)

    # tracker_id -> aggregated hit
    hits_by_id: dict[str, dict[str, Any]] = {}
    for class_name in classes:
        match = trie.find_longest(class_name)
        if match is None:
            continue
        tracker, signature = match
        tid = str(tracker.get("id") or tracker.get("name") or signature)
        row = hits_by_id.get(tid)
        if row is None:
            row = {
                "id": tracker.get("id", ""),
                "name": tracker.get("name", ""),
                "categories": list(tracker.get("categories") or []),
                "website": tracker.get("website", ""),
                "matched_signatures": [signature],
                "sample_classes": [class_name],
                "class_hit_count": 1,
            }
            hits_by_id[tid] = row
        else:
            row["class_hit_count"] = int(row.get("class_hit_count", 0)) + 1
            sigs = row.setdefault("matched_signatures", [])
            if signature not in sigs:
                sigs.append(signature)
            samples = row.setdefault("sample_classes", [])
            if len(samples) < 5 and class_name not in samples:
                samples.append(class_name)

    embedded = sorted(hits_by_id.values(), key=lambda r: (-int(r.get("class_hit_count", 0)), r.get("name", "")))

    return {
        "adapter": "tracker_dex_scanner",
        "apk_path": str(apk_path),
        "blocklist_source": str(blocklist.get("source", "")),
        "blocklist_tracker_count": len(trackers),
        "dex_class_count": len(classes),
        "embedded_trackers": embedded,
        "embedded_tracker_count": len(embedded),
        "embedded_tracker_names": [str(r.get("name", "")) for r in embedded if r.get("name")],
    }
