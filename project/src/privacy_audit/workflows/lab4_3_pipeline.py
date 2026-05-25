"""Lab4-3 access authenticity pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def _extract_runtime_observed_pii(dynamic_evidence: dict[str, Any]) -> list[str]:
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
        if api_name in {"", "hook_loaded", "hook_error"}:
            continue
        observed.add(pii_type)
    return sorted(observed)


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


def run_lab4_3(
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
    runtime_observed = _extract_runtime_observed_pii(dynamic)

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

    result: dict[str, Any] = {
        "app_name": app_name,
        "lab": "4-3",
        "policy_file": str(policy_file),
        "dynamic_evidence_file": str(dynamic_evidence_file) if dynamic_evidence_file else "",
        "copy_file": str(copy_file) if copy_file else "",
        "declared_pii": declared,
        "runtime_observed_pii": runtime_observed,
        "copy_key_count": len(copy_keys),
        "copy_keys_preview": copy_keys[:80],
        "copy_covered_pii": copy_covered,
        "missing_from_copy": missing_from_copy,
        "summary": {
            "declared_count": len(declared),
            "runtime_observed_count": len(runtime_observed),
            "copy_covered_count": len(copy_covered),
            "missing_count": len(missing_from_copy),
            "coverage_rate": round(
                len(copy_covered) / len(runtime_observed), 4
            )
            if runtime_observed
            else 0.0,
        },
    }

    if output_dir is not None:
        out_path = output_dir / f"{app_name}_lab4_3_result.json"
        export_json_report(result, out_path)
        result["report_path"] = str(out_path)
    return result
