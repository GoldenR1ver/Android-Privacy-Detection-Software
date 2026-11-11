"""Consent authenticity audit pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from privacy_audit.analyzers.dynamic.hook_loader import is_frida_runtime_api
from privacy_audit.reporting.report_exporter import export_json_report


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_runtime_hits(evidence: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    frida = evidence.get("frida_result", {}) if isinstance(evidence, dict) else {}
    events = frida.get("events", []) if isinstance(frida, dict) else []
    for event in events:
        if not isinstance(event, dict):
            continue
        pii_type = str(event.get("pii_type", "unknown"))
        api_name = str(event.get("api_name", ""))
        if pii_type in {"", "unknown"}:
            continue
        if not is_frida_runtime_api(api_name):
            continue
        counts[pii_type] = counts.get(pii_type, 0) + 1
    return counts


def _network_payload(evidence: dict[str, Any]) -> dict[str, Any]:
    network = evidence.get("network_result")
    if isinstance(network, dict) and network:
        return network
    return {}


def _extract_network_counts(evidence: dict[str, Any]) -> dict[str, Any]:
    """Track-3 metrics: tracker domain hits + pii network_key hits."""
    network = _network_payload(evidence)
    tracker_hits = [h for h in network.get("tracker_hits") or [] if isinstance(h, dict)]
    pii_hits = [h for h in network.get("pii_key_hits") or [] if isinstance(h, dict)]

    tracker_by_company: dict[str, int] = {}
    for hit in tracker_hits:
        company = str(hit.get("tracker_company") or hit.get("host") or "unknown")
        tracker_by_company[company] = tracker_by_company.get(company, 0) + 1

    pii_by_type: dict[str, int] = {}
    for hit in pii_hits:
        pii_type = str(hit.get("pii_type") or "unknown")
        pii_by_type[pii_type] = pii_by_type.get(pii_type, 0) + 1

    return {
        "tracker_hit_count": len(tracker_hits),
        "tracker_companies": sorted(tracker_by_company.keys()),
        "tracker_by_company": tracker_by_company,
        "pii_key_hit_count": len(pii_hits),
        "pii_key_by_type": pii_by_type,
        "mitm_blocked": network.get("mitm_blocked"),
        "flow_count": int(network.get("flow_count") or 0),
    }


def _compare_network_track(
    before_counts: dict[str, Any],
    after_counts: dict[str, Any],
) -> dict[str, Any]:
    """Consent audit network track: tracker / network_key before vs after switch-off."""
    before_trackers = dict(before_counts.get("tracker_by_company") or {})
    after_trackers = dict(after_counts.get("tracker_by_company") or {})
    companies = sorted(set(before_trackers) | set(after_trackers))

    tracker_rows: list[dict[str, Any]] = []
    still_leaking = 0
    for company in companies:
        before_n = int(before_trackers.get(company, 0))
        after_n = int(after_trackers.get(company, 0))
        if before_n > 0 and after_n > 0:
            verdict = "still_transmitting_after_off"
            still_leaking += 1
        elif before_n > 0 and after_n == 0:
            verdict = "tracker_stopped_after_off"
        elif before_n == 0 and after_n > 0:
            verdict = "new_tracker_after_off"
            still_leaking += 1
        else:
            verdict = "insufficient_observation"
        tracker_rows.append(
            {
                "tracker_company": company,
                "before_hits": before_n,
                "after_hits": after_n,
                "verdict": verdict,
            }
        )

    before_pii = dict(before_counts.get("pii_key_by_type") or {})
    after_pii = dict(after_counts.get("pii_key_by_type") or {})
    pii_types = sorted(set(before_pii) | set(after_pii))
    pii_rows: list[dict[str, Any]] = []
    for pii_type in pii_types:
        before_n = int(before_pii.get(pii_type, 0))
        after_n = int(after_pii.get(pii_type, 0))
        if before_n > 0 and after_n > 0:
            verdict = "still_leaking_pii_keys_after_off"
            still_leaking += 1
        elif before_n > 0 and after_n == 0:
            verdict = "pii_keys_stopped_after_off"
        elif before_n == 0 and after_n > 0:
            verdict = "new_pii_keys_after_off"
            still_leaking += 1
        else:
            verdict = "insufficient_observation"
        pii_rows.append(
            {
                "pii_type": pii_type,
                "before_hits": before_n,
                "after_hits": after_n,
                "verdict": verdict,
            }
        )

    before_fc = int(before_counts.get("flow_count") or 0)
    after_fc = int(after_counts.get("flow_count") or 0)
    if before_fc == 0 and after_fc == 0:
        status = "insufficient_network_observation"
    elif still_leaking > 0:
        status = "review_needed"
    elif before_fc > 0:
        status = "pass"
    else:
        status = "review_needed"

    return {
        "track": "network_mitm",
        "status": status,
        "still_leaking_count": still_leaking,
        "tracker_comparisons": tracker_rows,
        "pii_key_comparisons": pii_rows,
        "before_summary": before_counts,
        "after_summary": after_counts,
    }


def _resolve_overall_status(
    track3: dict[str, Any],
    *,
    frida_effective: int,
    frida_not_effective: int,
) -> tuple[str, str]:
    """mitm track3 is primary when any network flows were captured."""
    before_fc = int((track3.get("before_summary") or {}).get("flow_count") or 0)
    after_fc = int((track3.get("after_summary") or {}).get("flow_count") or 0)
    has_network = before_fc > 0 or after_fc > 0
    track3_status = str(track3.get("status") or "review_needed")
    still_leaking = int(track3.get("still_leaking_count") or 0)

    if has_network:
        if still_leaking > 0:
            return "review_needed", "track3_network"
        return track3_status, "track3_network"

    if frida_not_effective > 0:
        return "review_needed", "track1_frida"
    if frida_effective > 0:
        return "pass", "track1_frida"
    return "review_needed", "insufficient_observation"


def run_consent_audit(
    app_name: str,
    before_off_file: Path | None = None,
    after_off_file: Path | None = None,
    output_dir: Path | None = None,
    toggle_name: str = "privacy_switch",
) -> dict[str, Any]:
    """Compare before/after switch-off dynamic evidence for consent authenticity."""
    before_data = _load_json(before_off_file)
    after_data = _load_json(after_off_file)
    before_counts = _extract_runtime_hits(before_data)
    after_counts = _extract_runtime_hits(after_data)

    pii_types = sorted(set(before_counts) | set(after_counts))
    comparisons: list[dict[str, Any]] = []
    not_effective = 0
    effective = 0
    for pii in pii_types:
        before = before_counts.get(pii, 0)
        after = after_counts.get(pii, 0)
        if before > 0 and after == 0:
            verdict = "effective_off"
            effective += 1
        elif before > 0 and after > 0:
            verdict = "still_collecting_after_off"
            not_effective += 1
        elif before == 0 and after > 0:
            verdict = "risk_increased_after_off"
            not_effective += 1
        else:
            verdict = "insufficient_observation"
        comparisons.append(
            {
                "pii_type": pii,
                "before_off_hits": before,
                "after_off_hits": after,
                "verdict": verdict,
                "evidence_fields": [
                    "before_off.frida_result.events",
                    "after_off.frida_result.events",
                ],
            }
        )

    before_network = _extract_network_counts(before_data)
    after_network = _extract_network_counts(after_data)
    track3 = _compare_network_track(before_network, after_network)
    status, primary_source = _resolve_overall_status(
        track3,
        frida_effective=effective,
        frida_not_effective=not_effective,
    )

    result: dict[str, Any] = {
        "app_name": app_name,
        "audit": "consent",
        "toggle_name": toggle_name,
        "status": status,
        "primary_verdict_source": primary_source,
        "before_off_file": str(before_off_file) if before_off_file else "",
        "after_off_file": str(after_off_file) if after_off_file else "",
        "comparisons": comparisons,
        "track3_network": track3,
        "summary": {
            "pii_compared": len(pii_types),
            "effective_off_count": effective,
            "not_effective_count": not_effective,
            "before_runtime_hits": sum(before_counts.values()),
            "after_runtime_hits": sum(after_counts.values()),
            "before_tracker_hits": before_network.get("tracker_hit_count", 0),
            "after_tracker_hits": after_network.get("tracker_hit_count", 0),
            "before_pii_key_hits": before_network.get("pii_key_hit_count", 0),
            "after_pii_key_hits": after_network.get("pii_key_hit_count", 0),
            "before_flow_count": before_network.get("flow_count", 0),
            "after_flow_count": after_network.get("flow_count", 0),
            "track3_still_leaking": track3.get("still_leaking_count", 0),
        },
    }

    if output_dir is not None:
        out_path = output_dir / f"{app_name}_consent_result.json"
        export_json_report(result, out_path)
        result["report_path"] = str(out_path)
    return result



