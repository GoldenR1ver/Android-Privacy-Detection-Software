"""Lab4-2 consent authenticity pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from privacy_audit.reporting.report_exporter import export_json_report


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_runtime_hits(evidence: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    frida = evidence.get("frida_result", {}) if isinstance(evidence, dict) else {}
    events = frida.get("events", []) if isinstance(frida, dict) else []
    for event in events:
        if not isinstance(event, dict):
            continue
        pii_type = str(event.get("pii_type", "unknown"))
        api_name = str(event.get("api_name", ""))
        if pii_type in {"", "unknown"}:
            continue
        if api_name in {"", "hook_loaded", "hook_error"}:
            continue
        counts[pii_type] = counts.get(pii_type, 0) + 1
    return counts


def run_lab4_2(
    app_name: str,
    before_off_file: Path | None = None,
    after_off_file: Path | None = None,
    output_dir: Path | None = None,
    toggle_name: str = "privacy_switch",
) -> dict[str, Any]:
    """Compare before/after switch-off dynamic evidence for consent authenticity."""
    before_data = _load_json(before_off_file)
    after_data = _load_json(after_off_file)
    before_counts = _extract_runtime_hits(before_data)
    after_counts = _extract_runtime_hits(after_data)

    pii_types = sorted(set(before_counts) | set(after_counts))
    comparisons: list[dict[str, Any]] = []
    not_effective = 0
    effective = 0
    for pii in pii_types:
        before = before_counts.get(pii, 0)
        after = after_counts.get(pii, 0)
        if before > 0 and after == 0:
            verdict = "effective_off"
            effective += 1
        elif before > 0 and after > 0:
            verdict = "still_collecting_after_off"
            not_effective += 1
        elif before == 0 and after > 0:
            verdict = "risk_increased_after_off"
            not_effective += 1
        else:
            verdict = "insufficient_observation"
        comparisons.append(
            {
                "pii_type": pii,
                "before_off_hits": before,
                "after_off_hits": after,
                "verdict": verdict,
                "evidence_fields": [
                    "before_off.frida_result.events",
                    "after_off.frida_result.events",
                ],
            }
        )

    status = "pass" if not_effective == 0 and effective > 0 else "review_needed"
    result: dict[str, Any] = {
        "app_name": app_name,
        "lab": "4-2",
        "toggle_name": toggle_name,
        "status": status,
        "before_off_file": str(before_off_file) if before_off_file else "",
        "after_off_file": str(after_off_file) if after_off_file else "",
        "comparisons": comparisons,
        "summary": {
            "pii_compared": len(pii_types),
            "effective_off_count": effective,
            "not_effective_count": not_effective,
            "before_runtime_hits": sum(before_counts.values()),
            "after_runtime_hits": sum(after_counts.values()),
        },
    }

    if output_dir is not None:
        out_path = output_dir / f"{app_name}_lab4_2_result.json"
        export_json_report(result, out_path)
        result["report_path"] = str(out_path)
    return result
