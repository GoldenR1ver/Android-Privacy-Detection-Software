"""PMA-style Excel evidence export (API | stack | policy declaration)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

PMA_COLUMNS = [
    "app_name",
    "pii_type",
    "api_name",
    "stack_summary",
    "declared_in_policy",
    "evidence_level",
    "args_summary",
    "hook_file",
    "source",
]


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {col: str(row.get(col, "") or "") for col in PMA_COLUMNS}


def write_excel(rows: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    """Write PMA evidence rows to .xlsx (openpyxl) or .csv fallback."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [_normalize_row(row) for row in rows]

    try:
        from openpyxl import Workbook
    except ImportError:
        csv_path = output_path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=PMA_COLUMNS)
            writer.writeheader()
            writer.writerows(normalized)
        return {
            "format": "csv",
            "path": str(csv_path),
            "row_count": len(normalized),
            "note": "openpyxl not installed; wrote CSV with same columns",
        }

    wb = Workbook()
    ws = wb.active
    ws.title = "privacy_evidence"
    ws.append(PMA_COLUMNS)
    for row in normalized:
        ws.append([row[col] for col in PMA_COLUMNS])
    wb.save(output_path)
    return {
        "format": "xlsx",
        "path": str(output_path),
        "row_count": len(normalized),
    }


def write_excel_bundle(bundle: dict[str, Any], output_path: Path) -> dict[str, Any]:
    """Accept ``{rows: [...], meta: {...}}`` from export pipeline."""
    rows = bundle.get("rows") or []
    result = write_excel(rows, output_path)
    result["meta"] = dict(bundle.get("meta") or {})
    return result
