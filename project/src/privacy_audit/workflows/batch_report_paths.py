"""Resolve per-app Lab4-1 report paths from batch job directories."""

from __future__ import annotations

from pathlib import Path


def resolve_batch_report_path(reports_root: Path, app_name: str) -> Path | None:
    """
    Accept flat <app>_lab4_1_report.json or nested reports/<app>/<app>_lab4_1_report.json.
    """
    flat = reports_root / f"{app_name}_lab4_1_report.json"
    if flat.is_file():
        return flat
    nested = reports_root / app_name / f"{app_name}_lab4_1_report.json"
    if nested.is_file():
        return nested
    return None
