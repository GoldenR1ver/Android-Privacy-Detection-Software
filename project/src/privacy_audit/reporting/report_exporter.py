"""Report exporter utilities."""

import json

from pathlib import Path


def export_json_report(data: dict, output_path: Path) -> None:
    """Export report as UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
