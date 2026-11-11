"""Access authenticity audit pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from privacy_audit.analyzers.dynamic.hook_loader import is_frida_runtime_api
from privacy_audit.analyzers.policy.policy_parser import (
    load_personal_info_taxonomy,
    parse_declared_pii,
)
from privacy_audit.reporting.report_exporter import export_json_report


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_frida_observed_pii(dynamic_evidence: dict[str, Any]) -> list[str]:
    frida = dynamic_evidence.get("frida_result", {}) if isinstance(dynamic_evidence, dict) else {}
    events = frida.get("events", []) if isinstance(frida, dict) else []
    observed: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        pii_type = str(event.get("pii_type", "unknown"))
        api_name = str(event.get("api_name", ""))
        if pii_type in {"", "unknown"}:
            continue
        if not is_frida_runtime_api(api_name):
            continue
        observed.add(pii_type)
    return sorted(observed)


def _extract_mitm_observed_pii(dynamic_evidence: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """PII types seen in mitm network_result (pii_key_hits / summary.pii_types_observed)."""
    network = dynamic_evidence.get("network_result") if isinstance(dynamic_evidence, dict) else {}
    if not isinstance(network, dict) or not network:
        return [], {}

    observed: set[str] = set()
    for hit in network.get("pii_key_hits") or []:
        if not isinstance(hit, dict):
            continue
        pii_type = str(hit.get("pii_type") or "").strip()
        if pii_type and pii_type != "unknown":
            observed.add(pii_type)

    summary = network.get("summary") if isinstance(network.get("summary"), dict) else {}
    if not observed:
        for pii_type in summary.get("pii_types_observed") or []:
            token = str(pii_type).strip()
            if token and token != "unknown":
                observed.add(token)

    meta = {
        "flow_count": int(network.get("flow_count") or summary.get("flow_count") or 0),
        "pii_key_hit_count": int(
            summary.get("pii_key_hit_count") or len(network.get("pii_key_hits") or [])
        ),
        "tracker_hit_count": int(
            summary.get("tracker_hit_count") or len(network.get("tracker_hits") or [])
        ),
        "mitm_blocked": network.get("mitm_blocked"),
    }
    return sorted(observed), meta


def _resolve_runtime_observation(dynamic_evidence: dict[str, Any]) -> dict[str, Any]:
    """Merge Frida API hits and mitm network PII keys into one runtime observation set."""
    frida_observed = _extract_frida_observed_pii(dynamic_evidence)
    mitm_observed, mitm_summary = _extract_mitm_observed_pii(dynamic_evidence)
    merged = sorted(set(frida_observed) | set(mitm_observed))

    if frida_observed and mitm_observed:
        primary_source = "mitm+frida"
    elif mitm_observed:
        primary_source = "mitm"
    elif frida_observed:
        primary_source = "frida"
    else:
        primary_source = "none"

    return {
        "runtime_observed_pii": merged,
        "frida_observed_pii": frida_observed,
        "mitm_observed_pii": mitm_observed,
        "primary_source": primary_source,
        "mitm_summary": mitm_summary,
    }


def _coverage_interpretation(
    *,
    runtime_observed: list[str],
    primary_source: str,
) -> str:
    if not runtime_observed:
        return (
            "runtime_observed_pii 为空（Frida 与 mitm 均未产出可用运行时类型），"
            "自动覆盖率不适用（N/A），不能解读为「副本未覆盖」或「访问权未实现」。"
            "请依据 manual_copy_audit 对副本字段进行人工审核。"
        )
    base = (
        "自动指标：运行时观测到的 PII 类型中，副本字段名（含 taxonomy 别名）可匹配的比例；"
        "仍需人工审核语义与完整性。"
    )
    if primary_source == "mitm":
        return (
            "观测来源为 mitm 网络抓包（network_result.pii_key_hits），仅覆盖经网络上报的字段键；"
            + base
        )
    if primary_source == "frida":
        return "观测来源为 Frida API Hook；" + base
    return "观测来源为 Frida API Hook 与 mitm 网络抓包合并；" + base


def _flatten_keys(obj: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            curr = f"{prefix}.{k}" if prefix else str(k)
            keys.add(str(k))
            keys.add(curr)
            keys.update(_flatten_keys(v, curr))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_flatten_keys(item, prefix))
    return keys


def _alias_tokens(pii: str, taxonomy: dict[str, Any]) -> list[str]:
    aliases = taxonomy.get(pii, {}).get("aliases", [])
    return [pii, *[str(a) for a in aliases if a]]


def _field_matches_pii(field: str, pii: str, taxonomy: dict[str, Any]) -> bool:
    field_l = field.lower().strip()
    for token in _alias_tokens(pii, taxonomy):
        token_l = token.lower().strip()
        if not token_l:
            continue
        if field_l == token_l or field_l in token_l or token_l in field_l:
            return True
    return False


def _build_manual_copy_audit(
    declared: list[str],
    copy_keys: list[str],
    taxonomy: dict[str, Any],
) -> list[dict[str, Any]]:
    """Suggest copy-field → declared PII links for human review (not an auto-verdict)."""
    skip_prefixes = ("_",)
    rows: list[dict[str, Any]] = []
    for key in copy_keys:
        if not key or key.startswith(skip_prefixes):
            continue
        suggested = [pii for pii in declared if _field_matches_pii(key, pii, taxonomy)]
        if suggested:
            note = "taxonomy 别名命中，需人工确认字段语义与完整性"
        else:
            note = "无自动别名命中，需人工判断是否属于已声明 PII 或其他个人信息"
        rows.append(
            {
                "copy_field": key,
                "suggested_declared_pii": suggested,
                "auto_match": bool(suggested),
                "review_note": note,
            }
        )
    return rows


def run_access_audit(
    app_name: str,
    policy_file: Path,
    taxonomy_file: Path,
    dynamic_evidence_file: Path | None = None,
    copy_file: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare declared/runtime/copy coverage for access-right authenticity."""
    taxonomy = load_personal_info_taxonomy(taxonomy_file)
    policy_text = _read_text(policy_file)
    declared = parse_declared_pii(policy_text, taxonomy)
    dynamic = _read_json(dynamic_evidence_file)
    runtime_observation = _resolve_runtime_observation(dynamic)
    runtime_observed = list(runtime_observation["runtime_observed_pii"])

    copy_payload = _read_json(copy_file)
    copy_keys = sorted(_flatten_keys(copy_payload))
    copy_key_lower = {k.lower() for k in copy_keys}

    copy_covered: list[str] = []
    missing_from_copy: list[str] = []
    for pii in runtime_observed:
        aliases = taxonomy.get(pii, {}).get("aliases", [])
        candidates = [pii, *aliases]
        covered = any(token.lower() in copy_key_lower for token in candidates if token)
        if covered:
            copy_covered.append(pii)
        else:
            missing_from_copy.append(pii)

    manual_copy_audit = _build_manual_copy_audit(declared, copy_keys, taxonomy)
    manual_matched_fields = sum(1 for row in manual_copy_audit if row.get("auto_match"))
    user_copy_fields = len(manual_copy_audit)

    runtime_coverage_applicable = bool(runtime_observed)
    primary_source = str(runtime_observation.get("primary_source") or "none")
    if runtime_coverage_applicable:
        coverage_rate = round(len(copy_covered) / len(runtime_observed), 4)
    else:
        coverage_rate = None
    coverage_interpretation = _coverage_interpretation(
        runtime_observed=runtime_observed,
        primary_source=primary_source,
    )

    result: dict[str, Any] = {
        "app_name": app_name,
        "audit": "access",
        "status": "completed",
        "verdict": "review_needed",
        "policy_file": str(policy_file),
        "dynamic_evidence_file": str(dynamic_evidence_file) if dynamic_evidence_file else "",
        "copy_file": str(copy_file) if copy_file else "",
        "declared_pii": declared,
        "runtime_observed_pii": runtime_observed,
        "runtime_observation": runtime_observation,
        "copy_key_count": len(copy_keys),
        "copy_keys_preview": copy_keys[:80],
        "copy_covered_pii": copy_covered,
        "missing_from_copy": missing_from_copy,
        "manual_copy_audit": manual_copy_audit,
        "summary": {
            "declared_count": len(declared),
            "runtime_observed_count": len(runtime_observed),
            "frida_observed_count": len(runtime_observation.get("frida_observed_pii") or []),
            "mitm_observed_count": len(runtime_observation.get("mitm_observed_pii") or []),
            "runtime_primary_source": primary_source,
            "copy_covered_count": len(copy_covered),
            "missing_count": len(missing_from_copy),
            "coverage_rate": coverage_rate,
            "coverage_rate_applicable": runtime_coverage_applicable,
            "coverage_interpretation": coverage_interpretation,
            "mitm_flow_count": (runtime_observation.get("mitm_summary") or {}).get("flow_count"),
            "mitm_pii_key_hit_count": (runtime_observation.get("mitm_summary") or {}).get(
                "pii_key_hit_count"
            ),
            "user_copy_field_count": user_copy_fields,
            "manual_audit_auto_matched_fields": manual_matched_fields,
            "manual_audit_unmatched_fields": user_copy_fields - manual_matched_fields,
        },
    }

    if output_dir is not None:
        out_path = output_dir / f"{app_name}_access_result.json"
        export_json_report(result, out_path)
        result["report_path"] = str(out_path)
    return result


