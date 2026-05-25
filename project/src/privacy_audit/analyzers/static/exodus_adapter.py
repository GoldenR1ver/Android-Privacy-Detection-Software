"""Exodus-like static adapter.

This module provides a lightweight, local static pass that mimics the
high-level output shape needed from Exodus:
- permissions discovered in APK
- tracker/package hints
- pii types linked by permission mapping
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any


TRACKER_HINTS = (
    "facebook",
    "firebase",
    "adjust",
    "appsflyer",
    "umeng",
    "sensorsdata",
    "bugly",
    "flurry",
    "branch",
    "kochava",
    "talkingdata",
)


def _extract_permissions_with_androguard(apk_path: Path) -> list[str]:
    try:
        from androguard.core.apk import APK  # type: ignore
    except Exception:
        return []
    try:
        apk = APK(str(apk_path))
        perms = sorted(set(apk.get_permissions()))
        return perms
    except Exception:
        return []


def _extract_permissions_with_aapt(apk_path: Path) -> list[str]:
    import subprocess

    try:
        proc = subprocess.run(
            ["aapt", "dump", "permissions", str(apk_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    perms: set[str] = set()
    for line in proc.stdout.splitlines():
        match = re.search(r"(android\.permission\.[A-Z0-9_]+)", line)
        if match:
            perms.add(match.group(1))
    return sorted(perms)


def _extract_trackers_from_dex(apk_path: Path) -> list[str]:
    """Heuristic tracker extraction by string hints in classes.dex."""
    trackers: set[str] = set()
    try:
        with zipfile.ZipFile(apk_path, "r") as zf:
            dex_names = [name for name in zf.namelist() if name.endswith(".dex")]
            for name in dex_names:
                data = zf.read(name)
                text = data.decode("latin-1", errors="ignore").lower()
                for hint in TRACKER_HINTS:
                    if hint in text:
                        trackers.add(hint)
    except Exception:
        return []
    return sorted(trackers)


def _map_permissions_to_pii(
    permissions: list[str],
    taxonomy: dict[str, dict[str, list[str]]] | None,
) -> dict[str, list[str]]:
    if not taxonomy:
        return {}
    perms_set = set(permissions)
    pii_hits: dict[str, list[str]] = {}
    for pii_type, meta in taxonomy.items():
        configured = set(meta.get("permissions", []))
        if not configured:
            continue
        direct_hits = sorted(configured & perms_set)
        if not direct_hits:
            continue
        pii_hits[pii_type] = direct_hits
    return pii_hits


def run_exodus(
    apk_path: Path,
    taxonomy: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, Any]:
    """Run an Exodus-like static scan and return normalized evidence."""
    permissions = _extract_permissions_with_androguard(apk_path)
    permission_source = "androguard"
    if not permissions:
        permissions = _extract_permissions_with_aapt(apk_path)
        permission_source = "aapt" if permissions else "none"

    trackers = _extract_trackers_from_dex(apk_path)
    pii_permission_hits = _map_permissions_to_pii(permissions, taxonomy)

    return {
        "adapter": "exodus_like",
        "apk_path": str(apk_path),
        "permission_source": permission_source,
        "permissions": permissions,
        "trackers": trackers,
        "pii_permission_hits": pii_permission_hits,
    }
