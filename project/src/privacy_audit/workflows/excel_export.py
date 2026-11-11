"""Build PMA-style Excel rows from static + dynamic + policy evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from privacy_audit.analyzers.dynamic.hook_loader import is_frida_runtime_api
from privacy_audit.analyzers.policy.policy_parser import (
    load_personal_info_taxonomy,
    parse_declared_pii,
)
from privacy_audit.ingestion.evidence_loader import (
    _slug_for_app,
    resolve_dynamic_path,
    resolve_static_path,
)
from privacy_audit.reporting.excel_writer import write_excel_bundle


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _policy_declared(pii_type: str, declared: set[str]) -> str:
    return "yes" if pii_type in declared else "no"


def _rows_from_frida_events(
    app_name: str,
    frida_result: dict[str, Any],
    declared: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in frida_result.get("events") or []:
        if not isinstance(event, dict):
            continue
        api_name = str(event.get("api_name") or "")
        if not is_frida_runtime_api(api_name):
            continue
        pii_type = str(event.get("pii_type") or "unknown")
        stack = str(event.get("stack_summary") or "")
        level = "S3_runtime_frida" if stack and stack != "dry_run_stack" else "S2_runtime_hook"
        if stack == "dry_run_stack":
            level = "S2_dry_run"
        rows.append(
            {
                "app_name": app_name,
                "pii_type": pii_type,
                "api_name": api_name,
                "stack_summary": stack,
                "declared_in_policy": _policy_declared(pii_type, declared),
                "evidence_level": level,
                "args_summary": str(event.get("args_summary") or "")[:500],
                "hook_file": str(event.get("hook_file") or ""),
                "source": "dynamic.frida",
            }
        )
    return rows


def _rows_from_static(
    app_name: str,
    static_data: dict[str, Any],
    declared: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    andro = static_data.get("androguard") or {}
    api_hits = andro.get("api_hits") or {}
    if isinstance(api_hits, dict):
        for pii_type, hits in api_hits.items():
            for hit in hits or []:
                if not isinstance(hit, dict):
                    continue
                rows.append(
                    {
                        "app_name": app_name,
                        "pii_type": str(pii_type),
                        "api_name": str(hit.get("matched_api") or ""),
                        "stack_summary": str(hit.get("context") or "")[:500],
                        "declared_in_policy": _policy_declared(str(pii_type), declared),
                        "evidence_level": "S1_static_api",
                        "args_summary": "",
                        "hook_file": "",
                        "source": "static.androguard",
                    }
                )
    for tr in static_data.get("embedded_trackers") or []:
        if not isinstance(tr, dict):
            continue
        name = str(tr.get("name") or "")
        sigs = ", ".join(tr.get("matched_signatures") or [])
        rows.append(
            {
                "app_name": app_name,
                "pii_type": "sdk_info",
                "api_name": f"embedded_sdk:{name}",
                "stack_summary": (tr.get("sample_classes") or [""])[0]
                if isinstance(tr.get("sample_classes"), list)
                else "",
                "declared_in_policy": "unknown",
                "evidence_level": "S1_static_embedded",
                "args_summary": sigs[:500],
                "hook_file": "",
                "source": "static.embedded_tracker",
            }
        )
    return rows


def _rows_from_probe(app_name: str, probe_data: dict[str, Any]) -> list[dict[str, Any]]:
    bridge = str(probe_data.get("java_bridge") or "")
    if not bridge:
        return []
    return [
        {
            "app_name": app_name,
            "pii_type": "frida",
            "api_name": f"probe:{bridge}",
            "stack_summary": "",
            "declared_in_policy": "n/a",
            "evidence_level": "S0_probe",
            "args_summary": json.dumps(
                {
                    "mode": probe_data.get("mode"),
                    "runtime_events": probe_data.get("runtime_event_count", 0),
                    "success": probe_data.get("success"),
                },
                ensure_ascii=False,
            ),
            "hook_file": "probe_hook.py",
            "source": "frida_probe",
        }
    ]


def build_pma_excel_rows(
    app_name: str,
    *,
    policy_text: str = "",
    taxonomy_file: Path | None = None,
    static_evidence_dir: Path | None = None,
    dynamic_evidence_dir: Path | None = None,
    probe_dir: Path | None = None,
    static_evidence_file: Path | None = None,
    dynamic_evidence_file: Path | None = None,
) -> dict[str, Any]:
    taxonomy = load_personal_info_taxonomy(
        taxonomy_file or Path(__file__).resolve().parents[1] / "config" / "personal_info.yaml"
    )
    declared = set(parse_declared_pii(policy_text, taxonomy)) if policy_text else set()

    static_path = static_evidence_file
    if static_path is None:
        static_path = resolve_static_path(app_name, static_evidence_dir)
    dynamic_path = dynamic_evidence_file
    if dynamic_path is None:
        dynamic_path = resolve_dynamic_path(app_name, dynamic_evidence_dir)

    static_data = _read_json(static_path)
    dynamic_data = _read_json(dynamic_path)
    frida_result = dynamic_data.get("frida_result") or {}

    rows: list[dict[str, Any]] = []
    rows.extend(_rows_from_frida_events(app_name, frida_result, declared))
    rows.extend(_rows_from_static(app_name, static_data, declared))

    probe_path = (probe_dir or Path("output/frida_probe")) / f"{_slug_for_app(app_name)}.json"
    probe_data = _read_json(probe_path)
    rows.extend(_rows_from_probe(app_name, probe_data))

    if not rows and static_data:
        rows.extend(_rows_from_static(app_name, static_data, declared))

    return {
        "rows": rows,
        "meta": {
            "schema_version": 1,
            "app_name": app_name,
            "generated_at_utc": _utc_now(),
            "row_count": len(rows),
            "declared_pii_count": len(declared),
            "static_path": str(static_path) if static_path else "",
            "dynamic_path": str(dynamic_path) if dynamic_path else "",
            "probe_path": str(probe_path),
            "java_bridge": probe_data.get("java_bridge", ""),
        },
    }


def export_excel_evidence(
    app_name: str,
    output_path: Path,
    *,
    policy_file: Path | None = None,
    taxonomy_file: Path | None = None,
    static_evidence_dir: Path | None = None,
    dynamic_evidence_dir: Path | None = None,
    probe_dir: Path | None = None,
) -> dict[str, Any]:
    policy_text = policy_file.read_text(encoding="utf-8") if policy_file and policy_file.exists() else ""
    bundle = build_pma_excel_rows(
        app_name,
        policy_text=policy_text,
        taxonomy_file=taxonomy_file,
        static_evidence_dir=static_evidence_dir,
        dynamic_evidence_dir=dynamic_evidence_dir,
        probe_dir=probe_dir,
    )
    write_result = write_excel_bundle(bundle, output_path)
    return {**write_result, "app_name": app_name, "output_path": str(output_path)}

