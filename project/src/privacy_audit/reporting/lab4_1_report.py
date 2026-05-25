"""Lab4-1 report shaping: top-level feasible_evidence section."""

from __future__ import annotations

from typing import Any


def build_feasible_evidence_section(evidence: dict[str, Any]) -> dict[str, Any]:
    """Extract KB feasible evidence into a dedicated report chapter."""
    kb = evidence.get("knowledge_base") or {}
    items = list(kb.get("feasible_evidence") or [])
    return {
        "schema_version": 1,
        "kb_root": kb.get("kb_root", ""),
        "min_confidence": kb.get("min_confidence"),
        "observed_audit_types": list(kb.get("observed_audit_types") or []),
        "legal_item_ids": list(kb.get("legal_item_ids") or []),
        "legal_matrix": dict(kb.get("legal_matrix") or {}),
        "feasible_count": kb.get("feasible_count", len(items)),
        "high_confidence_count": kb.get(
            "high_confidence_count",
            sum(1 for item in items if item.get("feasible_for_verdict")),
        ),
        "items": items,
    }


def slim_evidence_for_report(evidence: dict[str, Any]) -> dict[str, Any]:
    """Keep evidence in reports without duplicating full feasible_evidence list."""
    slim = dict(evidence)
    kb = dict(slim.get("knowledge_base") or {})
    if kb:
        slim["knowledge_base"] = {
            "schema_version": kb.get("schema_version", 1),
            "kb_root": kb.get("kb_root", ""),
            "min_confidence": kb.get("min_confidence"),
            "observed_audit_types": kb.get("observed_audit_types", []),
            "legal_item_ids": kb.get("legal_item_ids", []),
            "feasible_count": kb.get("feasible_count", 0),
            "high_confidence_count": kb.get("high_confidence_count", 0),
            "feasible_evidence_in_section": "feasible_evidence",
        }
    return slim


def finalize_lab4_1_report(result: dict[str, Any]) -> dict[str, Any]:
    """Attach feasible_evidence chapter and avoid duplicating large KB payloads under evidence."""
    evidence = dict(result.get("evidence") or {})
    result["feasible_evidence"] = build_feasible_evidence_section(evidence)
    result["evidence"] = slim_evidence_for_report(evidence)
    return result
