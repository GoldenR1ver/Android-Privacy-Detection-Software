"""PII network keyword scan + network_evidence bundle (R3.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privacy_audit.analyzers.network.mitm_runner import request_query_text
from privacy_audit.analyzers.network.tracker_matcher import (
    match_flows_to_trackers,
    summarize_tracker_hits,
)
from privacy_audit.analyzers.policy.policy_parser import load_personal_info_taxonomy


def _collect_network_keys(taxonomy: dict[str, Any]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for pii_type, meta in taxonomy.items():
        if not isinstance(meta, dict):
            continue
        keys = meta.get("network_keys") or []
        if isinstance(keys, list) and keys:
            mapping[str(pii_type)] = [str(k).lower() for k in keys if str(k).strip()]
    return mapping


def scan_flows_for_pii_keys(
    flows: list[dict[str, Any]],
    taxonomy: dict[str, Any],
) -> list[dict[str, Any]]:
    """Match request query/body against personal_info.yaml network_keys."""
    key_map = _collect_network_keys(taxonomy)
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for flow in flows:
        if not isinstance(flow, dict):
            continue
        blob = request_query_text(flow)
        if not blob.strip():
            continue
        host = str(flow.get("host") or "")
        path = str(flow.get("path") or "")
        for pii_type, keys in key_map.items():
            for key in keys:
                if len(key) < 3:
                    continue
                if key not in blob:
                    continue
                dedupe = (host, path, f"{pii_type}:{key}")
                if dedupe in seen:
                    continue
                seen.add(dedupe)
                hits.append(
                    {
                        "pii_type": pii_type,
                        "network_key": key,
                        "host": host,
                        "path": path,
                        "timestamp_utc": str(flow.get("timestamp_utc") or ""),
                        "evidence_level": "S3_network_pii_key",
                    }
                )
    return hits


def build_network_evidence(
    mitm_result: dict[str, Any],
    *,
    taxonomy_file: Path | None = None,
    blocklist_dir: Path | None = None,
) -> dict[str, Any]:
    """Assemble network_evidence.json payload from mitm capture."""
    taxonomy_path = taxonomy_file or (
        Path(__file__).resolve().parents[2] / "config" / "personal_info.yaml"
    )
    taxonomy = load_personal_info_taxonomy(taxonomy_path)
    flows = [f for f in mitm_result.get("flows") or [] if isinstance(f, dict)]
    tracker_hits = match_flows_to_trackers(flows, blocklist_dir=blocklist_dir)
    pii_hits = scan_flows_for_pii_keys(flows, taxonomy)
    tracker_summary = summarize_tracker_hits(tracker_hits)

    return {
        "schema_version": 1,
        "adapter": "traffic_checker",
        "flow_count": len(flows),
        "unique_hosts": mitm_result.get("unique_hosts") or sorted(
            {str(f.get("host", "")) for f in flows if f.get("host")}
        ),
        "mitm_blocked": mitm_result.get("mitm_blocked"),
        "mitm_duration_sec": mitm_result.get("duration_sec"),
        "flows_source": mitm_result.get("flows_source", ""),
        "tracker_hits": tracker_hits,
        "pii_key_hits": pii_hits,
        "summary": {
            **tracker_summary,
            "pii_key_hit_count": len(pii_hits),
            "pii_types_observed": sorted({h["pii_type"] for h in pii_hits}),
        },
    }


def check_upload_indicators(
    app_package: str,
    *,
    flows: list[dict[str, Any]] | None = None,
    taxonomy_file: Path | None = None,
) -> dict[str, Any]:
    """Legacy entry: summarize upload indicators from flows."""
    taxonomy_path = taxonomy_file or (
        Path(__file__).resolve().parents[2] / "config" / "personal_info.yaml"
    )
    taxonomy = load_personal_info_taxonomy(taxonomy_path)
    flow_rows = flows or []
    pii_hits = scan_flows_for_pii_keys(flow_rows, taxonomy)
    return {
        "package": app_package,
        "indicators": pii_hits,
        "indicator_count": len(pii_hits),
    }
