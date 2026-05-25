"""Androguard static scanner for API evidence."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any


def _static_fast_mode() -> bool:
    return os.environ.get("PRIVACY_AUDIT_STATIC_FAST", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _scan_with_androguard(
    apk_path: Path,
    taxonomy: dict[str, dict[str, list[str]]],
) -> tuple[dict[str, list[dict[str, str]]], bool]:
    if _static_fast_mode():
        return {}, False
    try:
        from androguard.misc import AnalyzeAPK  # type: ignore
    except Exception:
        return {}, False
    try:
        _a, _d, dx = AnalyzeAPK(str(apk_path))
    except Exception:
        return {}, False

    api_hits: dict[str, list[dict[str, str]]] = {}
    for pii_type, meta in taxonomy.items():
        apis = meta.get("apis", [])
        if not apis:
            continue
        hits: list[dict[str, str]] = []
        for method in dx.get_methods():
            method_full = (
                f"{method.method.get_class_name()}->"
                f"{method.method.get_name()}{method.method.get_descriptor()}"
            )
            low = method_full.lower()
            matched = [api for api in apis if api.lower() in low]
            if not matched:
                continue
            hits.append(
                {
                    "matched_api": matched[0],
                    "method": method_full,
                    "context": f"class={method.method.get_class_name()}",
                }
            )
        if hits:
            api_hits[pii_type] = hits
    return api_hits, True


def _scan_by_dex_strings(
    apk_path: Path,
    taxonomy: dict[str, dict[str, list[str]]],
) -> dict[str, list[dict[str, str]]]:
    """Fallback API scanning by raw dex string search."""
    joined = ""
    try:
        with zipfile.ZipFile(apk_path, "r") as zf:
            dex_names = [name for name in zf.namelist() if name.endswith(".dex")]
            chunks: list[str] = []
            for name in dex_names:
                data = zf.read(name)
                chunks.append(data.decode("latin-1", errors="ignore"))
            joined = "\n".join(chunks).lower()
    except Exception:
        return {}

    api_hits: dict[str, list[dict[str, str]]] = {}
    for pii_type, meta in taxonomy.items():
        apis = meta.get("apis", [])
        if not apis:
            continue
        hits: list[dict[str, str]] = []
        for api in apis:
            if api.lower() in joined:
                hits.append(
                    {
                        "matched_api": api,
                        "method": "unknown_method",
                        "context": "matched_in_dex_strings",
                    }
                )
        if hits:
            api_hits[pii_type] = hits
    return api_hits


def scan_static_apis(
    apk_path: Path,
    taxonomy: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, Any]:
    """Scan static API hits and return normalized evidence."""
    taxonomy = taxonomy or {}
    api_hits, used_androguard = _scan_with_androguard(apk_path, taxonomy)
    source = "androguard" if used_androguard else "dex_string_fallback"
    if not api_hits:
        api_hits = _scan_by_dex_strings(apk_path, taxonomy)

    hit_counts = {pii_type: len(items) for pii_type, items in api_hits.items()}
    return {
        "adapter": "androguard_scanner",
        "apk_path": str(apk_path),
        "scan_source": source,
        "api_hits": api_hits,
        "hit_counts": hit_counts,
    }
