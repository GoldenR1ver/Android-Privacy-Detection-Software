"""Enrich merged evidence with knowledge-base feasible evidence for AI audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privacy_audit.utils.path_utils import output_provenance
from privacy_knowledgebase.kb_store import KnowledgeBaseStore, default_kb_root, get_kb_store
from privacy_knowledgebase.signal_inferencer import (
    dedupe_hits,
    infer_from_dynamic,
    infer_from_static,
    infer_from_text,
)

# Types that are policy/process metadata, not collection targets for Lab4-1.
_NON_COLLECTION_AUDIT_TYPES = frozenset({"rights_info", "security_info"})


def enrich_merged_evidence(
    evidence: dict[str, Any],
    *,
    kb: KnowledgeBaseStore | None = None,
    static_data: dict[str, Any] | None = None,
    dynamic_data: dict[str, Any] | None = None,
    extra_text: str = "",
    min_confidence: float = 0.72,
    include_low_confidence_in_feasible: bool = True,
    fallback_knowledge_base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Add knowledge_base observed types, feasible_evidence[], and legal_matrix summary.

    High-confidence KB hits are merged into observed_pii; full hit list kept for AI/explainability.
    """
    store = kb or get_kb_store()
    fallback = dict(fallback_knowledge_base or {})
    hits: list[dict[str, Any]] = []

    if static_data:
        hits.extend(infer_from_static(store, static_data))
    if dynamic_data:
        hits.extend(
            infer_from_dynamic(
                store,
                dynamic_data,
                include_candidates=bool(dynamic_data.get("candidate_pii")),
            )
        )
    if extra_text.strip():
        hits.extend(infer_from_text(store, extra_text, "merged.extra_text"))

    hits = dedupe_hits(hits, min_confidence=0.0)

    kb_observed: set[str] = set()
    feasible: list[dict[str, Any]] = []
    legal_items: set[str] = set()

    for hit in hits:
        conf = float(hit.get("confidence", 0))
        audit_types = [
            t
            for t in hit.get("audit_pii_types", [])
            if t not in _NON_COLLECTION_AUDIT_TYPES
        ]
        item_ids = list(hit.get("legal_item_ids", []))
        legal_items.update(item_ids)

        enriched = {
            **hit,
            "legal_labels": [store.item_label(i) for i in item_ids],
            "risk_buckets": [store.risk_bucket_for_item(i) for i in item_ids],
            "feasible_for_verdict": conf >= min_confidence and bool(audit_types),
        }
        if include_low_confidence_in_feasible or conf >= min_confidence:
            feasible.append(enriched)

        if conf >= min_confidence:
            kb_observed.update(audit_types)

    sources = evidence.setdefault("sources", {})
    if kb_observed:
        sources["knowledge_base"] = sorted(kb_observed)

    merged_observed = set(evidence.get("observed_pii", [])) | kb_observed
    evidence["observed_pii"] = sorted(merged_observed)

    labels = {
        item_id: {
            "label": store.item_label(item_id),
            "sensitivity": store.item_sensitivity(item_id),
            "risk_bucket": store.risk_bucket_for_item(item_id),
            "audit_pii_types": store.legal_items_to_audit_types([item_id]),
        }
        for item_id in sorted(legal_items)
    }

    if not feasible and fallback.get("feasible_evidence"):
        feasible = list(fallback.get("feasible_evidence") or [])
        kb_observed = set(fallback.get("observed_audit_types") or []) | kb_observed
        legal_items = set(fallback.get("legal_item_ids") or []) | legal_items
        labels = dict(fallback.get("legal_matrix") or {}) | labels
        merged_observed = set(evidence.get("observed_pii", [])) | kb_observed
        evidence["observed_pii"] = sorted(merged_observed)
        if kb_observed:
            sources["knowledge_base"] = sorted(kb_observed)

    evidence["knowledge_base"] = {
        "schema_version": 1,
        "kb_root": output_provenance("knowledge_base"),
        "min_confidence": min_confidence,
        "observed_audit_types": sorted(kb_observed),
        "legal_item_ids": sorted(legal_items),
        "legal_matrix": labels,
        "feasible_evidence": feasible,
        "feasible_count": len(feasible),
        "high_confidence_count": sum(1 for f in feasible if f.get("feasible_for_verdict")),
    }

    parts = []
    for key, vals in sources.items():
        if vals:
            parts.append(key)
    if "knowledge_base" in sources and sources["knowledge_base"]:
        parts.append("knowledge_base")
    evidence["source"] = f"merged:{'+'.join(parts)}" if parts else evidence.get("source", "empty")

    provenance = evidence.setdefault("provenance", {})
    provenance["knowledge_base"] = output_provenance("knowledge_base")

    return evidence


def enrich_evidence_bundle(
    bundle: dict[str, Any],
    *,
    kb: KnowledgeBaseStore | None = None,
    static_data: dict[str, Any] | None = None,
    dynamic_data: dict[str, Any] | None = None,
    min_confidence: float = 0.72,
) -> dict[str, Any]:
    """Enrich a flat lab4_1_evidence/{app}.json bundle."""
    return enrich_merged_evidence(
        bundle,
        kb=kb,
        static_data=static_data,
        dynamic_data=dynamic_data,
        min_confidence=min_confidence,
    )
