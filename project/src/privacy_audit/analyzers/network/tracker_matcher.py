"""Match captured HTTP hosts against tracker domain blocklist (R3.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privacy_audit.analyzers.static.tracker_blocklist_loader import load_tracker_domains


def _host_matches_domain(host: str, domain: str) -> bool:
    host = (host or "").lower().strip().lstrip(".")
    domain = (domain or "").lower().strip().lstrip(".")
    if not host or not domain:
        return False
    return host == domain or host.endswith("." + domain)


def build_domain_index(blocklist_dir: Path | None = None) -> list[dict[str, Any]]:
    data = load_tracker_domains(str(blocklist_dir) if blocklist_dir else None)
    return [row for row in data.get("domains") or [] if isinstance(row, dict)]


def match_host_to_tracker(
    host: str,
    domains: list[dict[str, Any]],
) -> dict[str, Any] | None:
    host_low = (host or "").lower().strip()
    if not host_low:
        return None
    best: dict[str, Any] | None = None
    best_len = -1
    for row in domains:
        domain = str(row.get("host") or "")
        if _host_matches_domain(host_low, domain) and len(domain) > best_len:
            best = row
            best_len = len(domain)
    return best


def match_flows_to_trackers(
    flows: list[dict[str, Any]],
    *,
    blocklist_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return tracker hits: tracker_company, purpose, host, path, timestamp."""
    domains = build_domain_index(blocklist_dir)
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for flow in flows:
        if not isinstance(flow, dict):
            continue
        host = str(flow.get("host") or "")
        match = match_host_to_tracker(host, domains)
        if match is None:
            continue
        path = str(flow.get("path") or "")
        ts = str(flow.get("timestamp_utc") or "")
        key = (host, path, str(match.get("tracker_id", "")))
        if key in seen:
            continue
        seen.add(key)
        categories = match.get("categories") or []
        purpose = ", ".join(str(c) for c in categories) if categories else "tracker"
        hits.append(
            {
                "tracker_id": str(match.get("tracker_id") or ""),
                "tracker_company": str(match.get("name") or ""),
                "purpose": purpose,
                "host": host,
                "path": path,
                "timestamp_utc": ts,
                "method": str(flow.get("method") or ""),
                "evidence_level": "S3_network_mitm",
            }
        )
    return sorted(hits, key=lambda r: (r.get("host", ""), r.get("path", "")))


def summarize_tracker_hits(hits: list[dict[str, Any]]) -> dict[str, Any]:
    companies = sorted({str(h.get("tracker_company") or "") for h in hits if h.get("tracker_company")})
    hosts = sorted({str(h.get("host") or "") for h in hits if h.get("host")})
    return {
        "tracker_hit_count": len(hits),
        "tracker_company_count": len(companies),
        "tracker_companies": companies,
        "tracker_hosts": hosts,
    }
