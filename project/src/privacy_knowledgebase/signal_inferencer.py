"""Infer legal/audit PII signals from static, dynamic, and text artifacts."""

from __future__ import annotations

import re
from typing import Any

from privacy_knowledgebase.kb_store import KnowledgeBaseStore

HOOK_TO_AUDIT: dict[str, list[str]] = {
    "hook_imei.js": ["imei"],
    "hook_android_id.js": ["android_id"],
    "hook_location.js": ["location"],
}

API_KEY_TO_AUDIT: dict[str, list[str]] = {
    "imei": ["imei"],
    "android_id": ["android_id"],
    "location": ["location"],
    "phone_number": ["phone_number"],
    "contact_list": ["contact_list"],
    "ip_address": ["ip_address"],
    "mac_address": ["mac_address"],
    "clipboard": ["clipboard"],
    "biometric_info": ["biometric_info"],
    "account_info": ["account_info"],
    "profile_info": ["profile_info"],
}


def _match_rule(rule: dict[str, Any], haystack: str) -> bool:
    for token in rule.get("match", []):
        if str(token).lower() in haystack:
            return True
    return False


def _append_hit(
    hits: list[dict[str, Any]],
    *,
    signal_type: str,
    signal_value: str,
    legal_item_ids: list[str],
    audit_pii_types: list[str],
    confidence: float,
    rationale: str,
    source: str,
) -> None:
    if not legal_item_ids and not audit_pii_types:
        return
    hits.append(
        {
            "signal_type": signal_type,
            "signal_value": signal_value,
            "legal_item_ids": sorted(set(legal_item_ids)),
            "audit_pii_types": sorted(set(audit_pii_types)),
            "confidence": round(confidence, 3),
            "rationale": rationale,
            "source": source,
        }
    )


def infer_from_android_rules(
    kb: KnowledgeBaseStore,
    haystack: str,
    source_prefix: str,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    lowered = haystack.lower()
    for rule in kb.android_rules():
        if not _match_rule(rule, lowered):
            continue
        legal_ids = [str(x) for x in rule.get("collects", [])]
        audit_types = kb.legal_items_to_audit_types(legal_ids)
        _append_hit(
            hits,
            signal_type=str(rule.get("type", "unknown")),
            signal_value="|".join(str(m) for m in rule.get("match", [])),
            legal_item_ids=legal_ids,
            audit_pii_types=audit_types,
            confidence=float(rule.get("confidence", 0.7)),
            rationale=str(rule.get("rationale", "")),
            source=source_prefix,
        )
    return hits


def infer_from_static(
    kb: KnowledgeBaseStore,
    static_data: dict[str, Any],
) -> list[dict[str, Any]]:
    if not static_data:
        return []
    hits: list[dict[str, Any]] = []
    chunks: list[str] = []

    exodus = static_data.get("exodus", {})
    if isinstance(exodus, dict):
        perms = exodus.get("permissions", [])
        if isinstance(perms, list):
            chunks.extend(str(p) for p in perms)
        trackers = exodus.get("trackers", [])
        if isinstance(trackers, list):
            chunks.extend(str(t) for t in trackers)
        pii_hits = exodus.get("pii_permission_hits", {})
        if isinstance(pii_hits, dict):
            for pii_type, perm_list in pii_hits.items():
                audit_types = API_KEY_TO_AUDIT.get(str(pii_type), [str(pii_type)])
                legal_ids = kb.audit_types_to_legal_items(audit_types)
                _append_hit(
                    hits,
                    signal_type="permission",
                    signal_value=",".join(str(p) for p in perm_list),
                    legal_item_ids=legal_ids,
                    audit_pii_types=audit_types,
                    confidence=0.92,
                    rationale="Exodus permission-to-PII hit.",
                    source="static.exodus.pii_permission_hits",
                )

    andro = static_data.get("androguard", {})
    if isinstance(andro, dict):
        api_hits = andro.get("api_hits", {})
        if isinstance(api_hits, dict):
            for pii_type, rows in api_hits.items():
                context_parts = []
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            context_parts.append(str(row.get("matched_api", "")))
                chunks.extend(context_parts)
                audit_types = API_KEY_TO_AUDIT.get(str(pii_type), [str(pii_type)])
                legal_ids = kb.audit_types_to_legal_items(audit_types)
                _append_hit(
                    hits,
                    signal_type="api",
                    signal_value=str(pii_type),
                    legal_item_ids=legal_ids,
                    audit_pii_types=audit_types,
                    confidence=0.88,
                    rationale="Androguard API hit bucket.",
                    source="static.androguard.api_hits",
                )

    haystack = "\n".join(chunks)
    hits.extend(infer_from_android_rules(kb, haystack, "static.android_rules"))
    return hits


def infer_from_dynamic(
    kb: KnowledgeBaseStore,
    dynamic_data: dict[str, Any],
    include_candidates: bool = True,
) -> list[dict[str, Any]]:
    if not dynamic_data:
        return []
    hits: list[dict[str, Any]] = []
    chunks: list[str] = []

    if include_candidates:
        for item in dynamic_data.get("candidate_pii", []):
            pii_type = str(item)
            audit_types = API_KEY_TO_AUDIT.get(pii_type, [pii_type])
            legal_ids = kb.audit_types_to_legal_items(audit_types)
            _append_hit(
                hits,
                signal_type="dynamic_candidate",
                signal_value=pii_type,
                legal_item_ids=legal_ids,
                audit_pii_types=audit_types,
                confidence=0.75,
                rationale="Stage3 candidate PII from static/policy hints.",
                source="dynamic.candidate_pii",
            )

    frida = dynamic_data.get("frida_result", {})
    if isinstance(frida, dict):
        for event in frida.get("events", []):
            if not isinstance(event, dict):
                continue
            pii_type = str(event.get("pii_type", ""))
            api_name = str(event.get("api_name", ""))
            hook_script = str(event.get("hook_script", ""))
            if not pii_type or pii_type == "unknown":
                continue
            confidence = 0.55 if api_name in {"", "hook_loaded"} else 0.9
            audit_types = API_KEY_TO_AUDIT.get(pii_type, [pii_type])
            legal_ids = kb.audit_types_to_legal_items(audit_types)
            chunks.append(f"{api_name} {hook_script}")
            _append_hit(
                hits,
                signal_type="frida",
                signal_value=api_name or hook_script or pii_type,
                legal_item_ids=legal_ids,
                audit_pii_types=audit_types,
                confidence=confidence,
                rationale="Frida runtime event.",
                source="dynamic.frida_result.events",
            )
        hooks = frida.get("hooks", [])
        if isinstance(hooks, list):
            for hook in hooks:
                hook_name = str(hook)
                for script, audit_types in HOOK_TO_AUDIT.items():
                    if script in hook_name or hook_name in script:
                        legal_ids = kb.audit_types_to_legal_items(audit_types)
                        _append_hit(
                            hits,
                            signal_type="frida_hook",
                            signal_value=hook_name,
                            legal_item_ids=legal_ids,
                            audit_pii_types=audit_types,
                            confidence=0.8,
                            rationale="Frida hook script loaded.",
                            source="dynamic.frida_result.hooks",
                        )

    ui = dynamic_data.get("ui_result", {})
    if isinstance(ui, dict):
        steps = ui.get("steps", [])
        if isinstance(steps, list):
            chunks.extend(str(s) for s in steps)
        checks = ui.get("keyword_checks", [])
        if isinstance(checks, list):
            for check in checks:
                if isinstance(check, dict):
                    chunks.append(str(check))

    text_blob = "\n".join(chunks)
    for item_id in kb.match_text_to_legal_items(text_blob):
        audit_types = kb.legal_items_to_audit_types([item_id])
        _append_hit(
            hits,
            signal_type="ui_text",
            signal_value=kb.item_label(item_id),
            legal_item_ids=[item_id],
            audit_pii_types=audit_types,
            confidence=0.7,
            rationale="UI steps/keyword text matched legal alias.",
            source="dynamic.ui_result",
        )

    hits.extend(infer_from_android_rules(kb, text_blob, "dynamic.android_rules"))
    return hits


def infer_from_text(kb: KnowledgeBaseStore, text: str, source: str) -> list[dict[str, Any]]:
    if not text.strip():
        return []
    hits: list[dict[str, Any]] = []
    for item_id in kb.match_text_to_legal_items(text):
        audit_types = kb.legal_items_to_audit_types([item_id])
        _append_hit(
            hits,
            signal_type="text",
            signal_value=kb.item_label(item_id),
            legal_item_ids=[item_id],
            audit_pii_types=audit_types,
            confidence=0.65,
            rationale="Free-text alias match against legal matrix.",
            source=source,
        )
    hits.extend(infer_from_android_rules(kb, text, f"{source}.android_rules"))
    return hits


def dedupe_hits(
    hits: list[dict[str, Any]],
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """Merge hits by (signal_type, signal_value, source); keep max confidence."""
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for hit in hits:
        if float(hit.get("confidence", 0)) < min_confidence:
            continue
        key = (
            str(hit.get("signal_type", "")),
            str(hit.get("signal_value", "")),
            str(hit.get("source", "")),
        )
        existing = merged.get(key)
        if existing is None or float(hit.get("confidence", 0)) > float(existing.get("confidence", 0)):
            merged[key] = hit
        else:
            existing["legal_item_ids"] = sorted(
                set(existing.get("legal_item_ids", [])) | set(hit.get("legal_item_ids", []))
            )
            existing["audit_pii_types"] = sorted(
                set(existing.get("audit_pii_types", [])) | set(hit.get("audit_pii_types", []))
            )
    return sorted(merged.values(), key=lambda h: (-float(h.get("confidence", 0)), h.get("source", "")))
