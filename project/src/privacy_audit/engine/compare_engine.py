"""Rule-based compare engine for Lab4-1 minimal loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privacy_audit.engine.kb_verdict import (
    apply_kb_verdict_adjustments,
    finalize_observed_layers,
    kb_declared_support_types,
    kb_supported_types,
)
from privacy_audit.engine.rule_loader import load_decision_rules


def _default_rule_map() -> dict[str, dict[str, Any]]:
    return {
        "declared and observed": {
            "verdict": "consistent",
            "risk_level": "low",
            "message": "Policy declared and runtime/static evidence observed.",
            "evidence_fields": ["claims.declared_pii", "evidence.observed_pii"],
        },
        "not_declared and observed": {
            "verdict": "inconsistent_high_risk",
            "risk_level": "high",
            "message": "Observed without declaration in policy.",
            "evidence_fields": ["evidence.observed_pii"],
        },
        "declared and not_observed": {
            "verdict": "declared_not_observed",
            "risk_level": "medium",
            "message": "Declared in policy but not observed in current evidence.",
            "evidence_fields": ["claims.declared_pii"],
        },
    }


def _build_rule_map(rules_file: Path | None) -> dict[str, dict[str, Any]]:
    if rules_file is None or not rules_file.exists():
        return _default_rule_map()
    loaded = load_decision_rules(rules_file)
    mapping: dict[str, dict[str, Any]] = {}
    for item in loaded:
        when = str(item.get("when", "")).strip()
        if not when:
            continue
        mapping[when] = {
            "verdict": str(item.get("verdict", "insufficient_evidence")),
            "risk_level": str(item.get("risk_level", "unknown")),
            "message": str(item.get("message", "")),
            "evidence_fields": list(item.get("evidence_fields", [])),
        }
    return mapping or _default_rule_map()


def compare_claims_with_evidence(
    claims: dict,
    evidence: dict,
    rules_file: Path | None = None,
    *,
    use_knowledge_base: bool = True,
) -> dict:
    """Evaluate consistency between policy claims and observed evidence."""
    defaults = _default_rule_map()
    rule_map = _build_rule_map(rules_file)
    declared = set(claims.get("declared_pii", []))

    if "observed_pii_for_verdict" not in evidence:
        finalize_observed_layers(evidence, use_knowledge_base=use_knowledge_base)

    technical = set(evidence.get("observed_pii_technical") or [])
    kb_types = set(evidence.get("observed_pii_kb") or []) | set(
        (evidence.get("sources") or {}).get("knowledge_base") or []
    )
    kb_supported = kb_supported_types(evidence) if use_knowledge_base else set()
    kb_declared_support = kb_declared_support_types(evidence) if use_knowledge_base else set()
    observed = set(evidence.get("observed_pii_for_verdict") or evidence.get("observed_pii", []))
    all_types = sorted(declared | observed | technical | kb_supported)

    verdicts: list[dict[str, str]] = []
    for pii_type in all_types:
        if pii_type in declared and pii_type in observed:
            rule = rule_map.get("declared and observed", defaults["declared and observed"])
        elif pii_type not in declared and pii_type in observed:
            rule = rule_map.get(
                "not_declared and observed",
                defaults["not_declared and observed"],
            )
        else:
            rule = rule_map.get(
                "declared and not_observed",
                defaults["declared and not_observed"],
            )
        verdicts.append(
            {
                "pii_type": pii_type,
                "status": rule["verdict"],
                "risk_level": rule["risk_level"],
                "message": rule["message"],
                "evidence_fields": rule["evidence_fields"],
                "verdict_basis": "rule_engine",
            }
        )

    verdicts = apply_kb_verdict_adjustments(
        verdicts,
        declared=declared,
        technical=technical,
        kb_types=kb_types,
        kb_supported=kb_supported,
        kb_declared_support=kb_declared_support,
        use_knowledge_base=use_knowledge_base,
        rules=rule_map,
    )

    status_counts: dict[str, int] = {}
    for row in verdicts:
        key = row["status"]
        status_counts[key] = status_counts.get(key, 0) + 1

    kb_influenced_count = sum(1 for row in verdicts if row.get("kb_influenced"))
    return {
        "claims": claims,
        "evidence": evidence,
        "verdicts": verdicts,
        "verdict_basis": {
            "use_knowledge_base": use_knowledge_base,
            "observed_for_verdict": sorted(observed),
            "observed_technical": sorted(technical),
            "observed_kb_only": sorted(kb_types - technical),
            "kb_supported_types": sorted(kb_supported),
            "kb_declared_support_types": sorted(kb_declared_support),
            "kb_influenced_verdict_count": kb_influenced_count,
        },
        "summary": {
            "declared_count": len(declared),
            "observed_count": len(observed),
            "observed_technical_count": len(technical),
            "observed_kb_only_count": len(kb_types - technical),
            "total_compared": len(all_types),
            "status_counts": status_counts,
            "kb_influenced_count": kb_influenced_count,
        },
        "rule_engine": {
            "rules_file": str(rules_file) if rules_file else "built_in_default",
            "rule_count": len(rule_map),
        },
    }
