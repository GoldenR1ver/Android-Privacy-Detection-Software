"""Unit tests for R3 mitm / tracker / network evidence."""

from __future__ import annotations

import json
from pathlib import Path

from privacy_audit.analyzers.network.mitm_addon import write_flows_payload
from privacy_audit.analyzers.network.mitm_runner import load_flows_from_file
from privacy_audit.analyzers.network.tracker_matcher import match_flows_to_trackers
from privacy_audit.analyzers.network.traffic_checker import build_network_evidence
from privacy_audit.workflows.capture_track import resolve_capture_tracks
from privacy_audit.workflows.consent_audit import run_consent_audit


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mitm_diaoyuren_flows.json"
BLOCKLIST = Path(__file__).resolve().parents[2] / "input" / "tracker_blocklist"


def test_write_flows_payload_incremental(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "mitm_flows.json"
    rows = [
        {
            "timestamp_utc": "2026-06-11T18:00:00+00:00",
            "host": "cnlogs.umeng.com",
            "path": "/unify_logs",
            "method": "POST",
            "scheme": "https",
            "request_body": "android_id=abc",
            "response_status": 200,
        }
    ]
    write_flows_payload(out, rows)
    assert out.exists()
    loaded = load_flows_from_file(out)
    assert len(loaded) == 1
    assert loaded[0]["host"] == "cnlogs.umeng.com"


def test_load_flows_and_match_umeng_tracker() -> None:
    flows = load_flows_from_file(FIXTURE)
    assert len(flows) == 2
    hits = match_flows_to_trackers(flows, blocklist_dir=BLOCKLIST)
    companies = {h["tracker_company"] for h in hits}
    assert "Umeng" in companies


def test_build_network_evidence_pii_keys() -> None:
    flows = load_flows_from_file(FIXTURE)
    mitm_result = {"flows": flows, "flow_count": len(flows), "duration_sec": 1.0}
    evidence = build_network_evidence(mitm_result, blocklist_dir=BLOCKLIST)
    assert evidence["summary"]["tracker_hit_count"] >= 1
    pii_types = evidence["summary"]["pii_types_observed"]
    assert "android_id" in pii_types or "imei" in pii_types


def test_resolve_dynamic_track_never_ready() -> None:
    probe_dir = Path(__file__).resolve().parent / "_probe_tmp"
    probe_dir.mkdir(exist_ok=True)
    (probe_dir / "diaoyuren.json").write_text(
        json.dumps({"java_bridge": "never_ready", "preferred_dynamic_tracks": ["mitm", "ui"]}),
        encoding="utf-8",
    )
    plan = resolve_capture_tracks("钓鱼人", dynamic_track="auto", probe_dir=probe_dir)
    assert plan["enable_frida"] is False
    assert plan["enable_mitm"] is True
    assert plan["dynamic_track_resolved"] == "mitm"


def test_resolve_auto_mitm_primary_tier1() -> None:
    plan = resolve_capture_tracks("小荷健康", dynamic_track="auto")
    assert plan["enable_mitm"] is True
    assert plan["enable_frida"] is False
    assert plan["dynamic_track_resolved"] == "mitm"


def test_resolve_lab42_track() -> None:
    plan = resolve_capture_tracks("小荷健康", dynamic_track="consent")
    assert plan["enable_mitm"] is True
    assert plan["enable_frida"] is False
    assert plan["dynamic_track_resolved"] == "mitm"


def test_lab42_track3_network_comparison(tmp_path: Path) -> None:
    flows = load_flows_from_file(FIXTURE)
    network = build_network_evidence({"flows": flows, "flow_count": len(flows)}, blocklist_dir=BLOCKLIST)
    before = {"network_result": network}
    after = {"network_result": {**network, "tracker_hits": [], "pii_key_hits": [], "summary": {**network["summary"], "tracker_hit_count": 0, "pii_key_hit_count": 0}}}
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(json.dumps(before), encoding="utf-8")
    after_path.write_text(json.dumps(after), encoding="utf-8")
    result = run_consent_audit("钓鱼人", before_off_file=before_path, after_off_file=after_path)
    assert result["primary_verdict_source"] == "track3_network"
    assert result["track3_network"]["status"] == "pass"
    assert result["summary"]["before_tracker_hits"] >= 1

