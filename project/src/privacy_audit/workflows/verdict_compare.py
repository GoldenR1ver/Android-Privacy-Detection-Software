"""Compare verdicts between two Lab4-1 batch jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from privacy_audit.workflows.batch_report_paths import resolve_batch_report_path


def _verdict_map(report: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in report.get("verdicts", []):
        pii_type = row.get("pii_type")
        if not pii_type:
            continue
        out[pii_type] = str(row.get("status") or row.get("verdict") or "")
    return out


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_batch_verdicts(
    *,
    before_dir: Path,
    after_dir: Path,
    apps: list[str],
    before_label: str = "before",
    after_label: str = "after",
) -> dict[str, Any]:
    per_app: dict[str, Any] = {}
    summary = {
        "apps_compared": 0,
        "apps_unchanged": 0,
        "apps_with_changes": 0,
        "total_verdict_changes": 0,
        "observed_pii_changes": 0,
        "feasible_evidence_delta": 0,
    }

    for app in apps:
        before_path = resolve_batch_report_path(before_dir, app)
        after_path = resolve_batch_report_path(after_dir, app)
        if before_path is None or after_path is None:
            per_app[app] = {
                "status": "missing_report",
                "before": str(before_dir / f"{app}_lab4_1_report.json"),
                "after": str(after_dir / f"{app}_lab4_1_report.json"),
            }
            continue

        before = _load_report(before_path)
        after = _load_report(after_path)
        before_map = _verdict_map(before)
        after_map = _verdict_map(after)
        all_types = sorted(set(before_map) | set(after_map))

        changes: list[dict[str, str]] = []
        kb_influenced_after = 0
        for pii_type in all_types:
            b = before_map.get(pii_type)
            a = after_map.get(pii_type)
            if b != a:
                changes.append({"pii_type": pii_type, before_label: b or None, after_label: a or None})
        for row in after.get("verdicts", []):
            if row.get("kb_influenced"):
                kb_influenced_after += 1

        before_obs = set(before.get("evidence", {}).get("observed_pii", []))
        after_obs = set(after.get("evidence", {}).get("observed_pii", []))
        obs_added = sorted(after_obs - before_obs)
        obs_removed = sorted(before_obs - after_obs)

        before_feasible = int((before.get("feasible_evidence") or {}).get("feasible_count", 0))
        after_feasible = int((after.get("feasible_evidence") or {}).get("feasible_count", 0))
        feasible_delta = after_feasible - before_feasible

        entry = {
            "status": "ok",
            "verdict_changes": changes,
            "observed_added": obs_added,
            "observed_removed": obs_removed,
            "feasible_count_before": before_feasible,
            "feasible_count_after": after_feasible,
            "feasible_count_delta": feasible_delta,
            "kb_influenced_verdicts_after": kb_influenced_after,
            "verdict_basis_after": after.get("verdict_basis"),
            "evidence_source_before": before.get("evidence", {}).get("source"),
            "evidence_source_after": after.get("evidence", {}).get("source"),
            "observed_technical_count_after": len(
                after.get("evidence", {}).get("observed_pii_technical") or []
            ),
            "observed_kb_only_count_after": len(after.get("evidence", {}).get("observed_pii_kb") or []),
        }
        per_app[app] = entry

        summary["apps_compared"] += 1
        if changes or obs_added or obs_removed:
            summary["apps_with_changes"] += 1
        else:
            summary["apps_unchanged"] += 1
        summary["total_verdict_changes"] += len(changes)
        if obs_added or obs_removed:
            summary["observed_pii_changes"] += 1
        summary["feasible_evidence_delta"] += feasible_delta

    return {
        "schema_version": 1,
        "before_label": before_label,
        "after_label": after_label,
        "before_dir": str(before_dir),
        "after_dir": str(after_dir),
        "summary": summary,
        "per_app": per_app,
    }


def export_verdict_compare(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
