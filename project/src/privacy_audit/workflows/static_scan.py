"""Static scan pipeline: APK analysis with Exodus + Androguard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privacy_audit.analyzers.policy.policy_parser import load_personal_info_taxonomy
from privacy_audit.analyzers.static.androguard_scanner import scan_static_apis
from privacy_audit.analyzers.static.exodus_adapter import run_exodus
from privacy_audit.analyzers.static.tracker_dex_scanner import scan_embedded_trackers
from privacy_audit.ingestion.apk_registry import load_apk_list
from privacy_audit.reporting.report_exporter import export_json_report


def _build_static_evidence(
    app_name: str,
    apk_path: Path,
    exodus_result: dict[str, Any],
    androguard_result: dict[str, Any],
    tracker_result: dict[str, Any],
) -> dict[str, Any]:
    observed_pii = sorted(
        set(exodus_result.get("pii_permission_hits", {}).keys())
        | set(androguard_result.get("api_hits", {}).keys())
    )
    evidence_items: list[dict[str, Any]] = []

    for pii_type, perms in exodus_result.get("pii_permission_hits", {}).items():
        evidence_items.append(
            {
                "source": "static.permission",
                "pii_type": pii_type,
                "signal": "permission_hit",
                "value": ",".join(perms),
                "evidence_fields": ["exodus.permissions"],
            }
        )
    for pii_type, api_rows in androguard_result.get("api_hits", {}).items():
        for row in api_rows:
            evidence_items.append(
                {
                    "source": "static.api",
                    "pii_type": pii_type,
                    "signal": "api_hit",
                    "value": row.get("matched_api", ""),
                    "context": row.get("context", ""),
                    "evidence_fields": ["androguard.api_hits"],
                }
            )

    embedded = tracker_result.get("embedded_trackers") or []
    for tr in embedded:
        if not isinstance(tr, dict):
            continue
        evidence_items.append(
            {
                "source": "static.embedded_tracker",
                "pii_type": "third_party_sdk",
                "signal": "dex_signature_hit",
                "value": str(tr.get("name", "")),
                "context": ",".join(tr.get("matched_signatures") or [])[:240],
                "evidence_fields": ["embedded_trackers"],
            }
        )

    return {
        "schema_version": 2,
        "app_name": app_name,
        "apk_path": str(apk_path),
        "observed_pii": observed_pii,
        "exodus": exodus_result,
        "androguard": androguard_result,
        "embedded_trackers": embedded,
        "tracker_dex_scan": tracker_result,
        "evidence_items": evidence_items,
        "summary": {
            "observed_pii_count": len(observed_pii),
            "permission_hit_types": len(exodus_result.get("pii_permission_hits", {})),
            "api_hit_types": len(androguard_result.get("api_hits", {})),
            "tracker_hint_count": len(exodus_result.get("trackers", [])),
            "embedded_tracker_count": len(embedded),
            "embedded_tracker_names": tracker_result.get("embedded_tracker_names", []),
        },
    }


def run_static_scan(
    apk_path: Path,
    taxonomy_file: Path,
    output_dir: Path,
    *,
    tracker_blocklist_dir: Path | None = None,
) -> dict[str, Any]:
    """Run a static scan for a single APK."""
    taxonomy = load_personal_info_taxonomy(taxonomy_file)
    app_name = apk_path.stem
    exodus_result = run_exodus(apk_path, taxonomy=taxonomy)
    androguard_result = scan_static_apis(apk_path, taxonomy=taxonomy)
    tracker_result = scan_embedded_trackers(
        apk_path,
        blocklist_dir=tracker_blocklist_dir,
    )
    result = _build_static_evidence(
        app_name=app_name,
        apk_path=apk_path,
        exodus_result=exodus_result,
        androguard_result=androguard_result,
        tracker_result=tracker_result,
    )
    target = output_dir / app_name / "static_evidence.json"
    export_json_report(result, target)
    result["report_path"] = str(target)
    return result


def run_static_scan_directory(
    apk_dir: Path,
    taxonomy_file: Path,
    output_dir: Path,
    limit: int = 0,
    *,
    tracker_blocklist_dir: Path | None = None,
) -> dict[str, Any]:
    """Run static scans for all APKs in a directory."""
    apk_paths = load_apk_list(apk_dir)
    if limit > 0:
        apk_paths = apk_paths[:limit]

    reports: list[dict[str, Any]] = []
    for apk_path in apk_paths:
        reports.append(
            run_static_scan(
                apk_path=apk_path,
                taxonomy_file=taxonomy_file,
                output_dir=output_dir,
                tracker_blocklist_dir=tracker_blocklist_dir,
            )
        )

    summary = {
        "pipeline": "static_scan",
        "apk_dir": str(apk_dir),
        "total_apks": len(apk_paths),
        "reports": reports,
    }
    export_json_report(summary, output_dir / "static_scan_summary.json")
    return summary


