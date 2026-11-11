"""Resolve per-app disclosure report paths from batch job directories."""

from __future__ import annotations

from pathlib import Path

# New canonical name first, then the legacy Lab4-1 name for existing artifacts.
REPORT_SUFFIXES = ("disclosure_report.json", "lab4_1_report.json")


def resolve_batch_report_path(reports_root: Path, app_name: str) -> Path | None:
    """
    Accept flat <app>_disclosure_report.json or nested
    reports/<app>/<app>_disclosure_report.json. Legacy *_lab4_1_report.json
    names are still resolved so older experiment outputs remain readable.
    """
    for suffix in REPORT_SUFFIXES:
        flat = reports_root / f"{app_name}_{suffix}"
        if flat.is_file():
            return flat
        nested = reports_root / app_name / f"{app_name}_{suffix}"
        if nested.is_file():
            return nested
    return None
