"""Knowledge-base influenced verdict adjustments for Lab4-1."""

from __future__ import annotations

from typing import Any

_NON_COLLECTION_AUDIT_TYPES = frozenset({"rights_info", "security_info"})


def technical_observed_from_sources(sources: dict[str, list[str]]) -> set[str]:
    """Union of all source buckets except knowledge_base."""
    observed: set[str] = set()
    for key, vals in sources.items():
        if key == "knowledge_base":
            continue
        observed.update(vals or [])
    return observed


def _types_from_feasible_items(
    evidence: dict[str, Any],
    *,
    min_confidence: float,
    require_feasible_flag: bool,
) -> set[str]:
    supported: set[str] = set()
    kb = evidence.get("knowledge_base") or {}
    legal_matrix = kb.get("legal_matrix") or {}
    for item in kb.get("feasible_evidence") or []:
        conf = float(item.get("confidence", 0))
        if conf < min_confidence:
            continue
        if require_feasible_flag and not item.get("feasible_for_verdict"):
            continue
        for audit_type in item.get("audit_pii_types") or []:
            if audit_type not in _NON_COLLECTION_AUDIT_TYPES:
                supported.add(str(audit_type))
        for item_id in item.get("legal_item_ids") or []:
            bridged = (legal_matrix.get(item_id) or {}).get("audit_pii_types") or []
            for audit_type in bridged:
                if audit_type not in _NON_COLLECTION_AUDIT_TYPES:
                    supported.add(str(audit_type))
    return supported


def kb_supported_types(evidence: dict[str, Any]) -> set[str]:
    """Audit types backed by feasible_for_verdict KB hits (incl. legal_matrix bridge)."""
    kb = evidence.get("knowledge_base") or {}
    min_conf = float(kb.get("min_confidence", 0.72))
    return _types_from_feasible_items(
        evidence,
        min_confidence=min_conf,
        require_feasible_flag=True,
    )


def kb_declared_support_types(evidence: dict[str, Any], *, min_confidence: float = 0.6) -> set[str]:
    """Broader KB support for upgrading declared_not_observed (medium-confidence hits)."""
    return _types_from_feasible_items(
        evidence,
        min_confidence=min_confidence,
        require_feasible_flag=False,
    )


def finalize_observed_layers(
    evidence: dict[str, Any],
    *,
    use_knowledge_base: bool,
) -> None:
    """Populate technical / kb / verdict observed sets on evidence dict."""
    sources: dict[str, list[str]] = evidence.get("sources") or {}
    technical = technical_observed_from_sources(sources)
    kb_from_sources = set(sources.get("knowledge_base") or [])
    kb_from_feasible = kb_supported_types(evidence) if use_knowledge_base else set()
    kb_types = (kb_from_sources | kb_from_feasible) if use_knowledge_base else set()
    kb_only = kb_types - technical

    evidence["observed_pii_technical"] = sorted(technical)
    evidence["observed_pii_kb"] = sorted(kb_only)
    if use_knowledge_base:
        for_verdict = technical | kb_types
    else:
        for_verdict = technical
    evidence["observed_pii_for_verdict"] = sorted(for_verdict)
    evidence["observed_pii"] = sorted(technical | kb_types)


def apply_kb_verdict_adjustments(
    verdicts: list[dict[str, Any]],
    *,
    declared: set[str],
    technical: set[str],
    kb_types: set[str],
    kb_supported: set[str],
    kb_declared_support: set[str] | None = None,
    use_knowledge_base: bool,
    rules: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Upgrade/downgrade verdict rows when KB provides legal backing."""
    if not use_knowledge_base:
        return verdicts

    declared_support = kb_declared_support or set()

    kb_confirm = rules.get(
        "declared and kb_supported",
        {
            "verdict": "kb_supported_consistent",
            "risk_level": "low",
            "message": "Declared; KB feasible evidence supports collection (technical trace weak).",
            "evidence_fields": ["claims.declared_pii", "feasible_evidence"],
        },
    )
    kb_inferred = rules.get(
        "not_declared and kb_only",
        {
            "verdict": "kb_inferred_undeclared",
            "risk_level": "high",
            "message": "KB high-confidence inference without policy declaration or technical trace.",
            "evidence_fields": ["feasible_evidence", "evidence.observed_pii_kb"],
        },
    )
    adjusted: list[dict[str, Any]] = []
    for row in verdicts:
        pii_type = row.get("pii_type", "")
        in_declared = pii_type in declared
        in_technical = pii_type in technical
        in_kb = pii_type in kb_types or pii_type in kb_supported
        kb_only = in_kb and not in_technical
        new_row = dict(row)

        if (
            in_declared
            and not in_technical
            and (
                in_kb
                or pii_type in kb_supported
                or pii_type in declared_support
            )
            and row.get("status") == "declared_not_observed"
        ):
            new_row.update(
                {
                    "status": kb_confirm["verdict"],
                    "risk_level": kb_confirm["risk_level"],
                    "message": kb_confirm["message"],
                    "evidence_fields": list(kb_confirm["evidence_fields"]),
                    "kb_influenced": True,
                    "verdict_basis": "knowledge_base",
                }
            )
        elif not in_declared and kb_only:
            new_row.update(
                {
                    "status": kb_inferred["verdict"],
                    "risk_level": kb_inferred["risk_level"],
                    "message": kb_inferred["message"],
                    "evidence_fields": list(kb_inferred["evidence_fields"]),
                    "kb_influenced": True,
                    "verdict_basis": "knowledge_base_only",
                }
            )
        elif not in_declared and in_technical and row.get("status") == "inconsistent_high_risk":
            new_row["verdict_basis"] = "technical"
        elif in_declared and in_technical:
            new_row["verdict_basis"] = "technical"
        elif in_declared and in_kb:
            new_row["verdict_basis"] = "technical_and_kb"
        else:
            new_row.setdefault("verdict_basis", "rule_engine")

        adjusted.append(new_row)
    return adjusted
