"""Disclosure (informed-right) audit pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privacy_audit.analyzers.policy.policy_parser import (
    load_personal_info_taxonomy,
    parse_declared_pii,
)
from privacy_audit.engine.compare_engine import compare_claims_with_evidence
from privacy_audit.engine.tracker_disclosure import (
    compare_embedded_trackers_with_policy,
    merge_sdk_verdicts_into_compare_result,
)
from privacy_audit.ingestion.evidence_loader import resolve_static_path
from privacy_audit.ingestion.evidence_loader import build_merged_evidence
from privacy_audit.reporting.disclosure_report import finalize_disclosure_report
from privacy_audit.reporting.report_exporter import export_json_report


def run_disclosure_audit(
    app_name: str,
    policy_file: Path,
    output_dir: Path,
    taxonomy_file: Path,
    rules_file: Path | None = None,
    evidence_file: Path | None = None,
    evidence_dir: Path | None = None,
    static_evidence_dir: Path | None = None,
    legacy_run_dir: Path | None = None,
    legacy_md_run_dir: Path | None = None,
    include_dynamic_candidates: bool = True,
    use_knowledge_base: bool = True,
    kb_min_confidence: float = 0.72,
) -> dict[str, Any]:
    """Parse policy claims, merge evidence, compare, and export a JSON report."""
    policy_text = policy_file.read_text(encoding="utf-8")
    taxonomy = load_personal_info_taxonomy(taxonomy_file)
    declared = parse_declared_pii(policy_text, taxonomy)

    evidence = build_merged_evidence(
        app_name=app_name,
        taxonomy=taxonomy,
        evidence_dir=evidence_dir,
        static_dir=static_evidence_dir,
        legacy_run_dir=legacy_run_dir,
        legacy_md_run_dir=legacy_md_run_dir,
        evidence_file=evidence_file,
        include_dynamic_candidates=include_dynamic_candidates,
        use_knowledge_base=use_knowledge_base,
        kb_min_confidence=kb_min_confidence,
    )

    claims = {"app_name": app_name, "declared_pii": declared}
    result = compare_claims_with_evidence(
        claims=claims,
        evidence=evidence,
        rules_file=rules_file,
        use_knowledge_base=use_knowledge_base,
    )
    result["audit"] = "disclosure"

    static_path = resolve_static_path(app_name, static_evidence_dir)
    embedded_trackers: list[dict[str, Any]] = []
    if static_path is not None and static_path.exists():
        import json

        static_payload = json.loads(static_path.read_text(encoding="utf-8"))
        raw = static_payload.get("embedded_trackers") or []
        embedded_trackers = [row for row in raw if isinstance(row, dict)]
    sdk_disclosure = compare_embedded_trackers_with_policy(
        policy_text,
        embedded_trackers,
    )
    result["sdk_disclosure"] = sdk_disclosure
    result = merge_sdk_verdicts_into_compare_result(result, sdk_disclosure)

    result["evidence"] = evidence
    result = finalize_disclosure_report(result)

    output_path = output_dir / f"{app_name}_disclosure_report.json"
    export_json_report(result, output_path)
    result["report_path"] = str(output_path)
    return result


