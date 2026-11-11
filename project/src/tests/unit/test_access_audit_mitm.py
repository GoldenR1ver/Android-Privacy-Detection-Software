"""Lab4-3 mitm runtime observation integration."""

from __future__ import annotations

import json
from pathlib import Path

from privacy_audit.analyzers.network.traffic_checker import build_network_evidence
from privacy_audit.analyzers.network.mitm_runner import load_flows_from_file
from privacy_audit.workflows.access_audit import run_access_audit

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mitm_diaoyuren_flows.json"
BLOCKLIST = Path(__file__).resolve().parents[2] / "input" / "tracker_blocklist"
TAXONOMY = Path(__file__).resolve().parents[2] / "privacy_audit" / "config" / "personal_info.yaml"
KEEP_BEFORE = (
    Path(__file__).resolve().parents[2] / "input" / "consent_before" / "Keep.json"
)


def test_lab4_3_mitm_observed_from_fixture(tmp_path: Path) -> None:
    flows = load_flows_from_file(FIXTURE)
    network = build_network_evidence({"flows": flows, "flow_count": len(flows)}, blocklist_dir=BLOCKLIST)
    dynamic_path = tmp_path / "dynamic.json"
    dynamic_path.write_text(json.dumps({"network_result": network}), encoding="utf-8")
    copy_path = tmp_path / "copy.json"
    copy_path.write_text(json.dumps({"手机号": "13800000000"}), encoding="utf-8")
    policy_path = tmp_path / "policy.txt"
    policy_path.write_text("本应用收集手机号、设备标识符等信息。", encoding="utf-8")

    result = run_access_audit(
        "钓鱼人",
        policy_file=policy_path,
        taxonomy_file=TAXONOMY,
        dynamic_evidence_file=dynamic_path,
        copy_file=copy_path,
        output_dir=tmp_path,
    )

    obs = result["runtime_observation"]
    assert obs["primary_source"] == "mitm"
    assert obs["mitm_observed_pii"]
    assert result["runtime_observed_pii"] == obs["mitm_observed_pii"]
    assert result["summary"]["coverage_rate_applicable"] is True
    assert result["summary"]["mitm_flow_count"] == len(flows)


def test_lab4_3_keep_before_mitm_has_runtime_types(tmp_path: Path) -> None:
    if not KEEP_BEFORE.exists():
        return
    copy_path = tmp_path / "copy.json"
    copy_path.write_text(json.dumps({"地区": "北京"}), encoding="utf-8")
    policy_path = tmp_path / "policy.txt"
    policy_path.write_text("收集地理位置、MAC地址、日志信息。", encoding="utf-8")

    result = run_access_audit(
        "Keep",
        policy_file=policy_path,
        taxonomy_file=TAXONOMY,
        dynamic_evidence_file=KEEP_BEFORE,
        copy_file=copy_path,
        output_dir=tmp_path,
    )

    mitm_types = set(result["runtime_observation"]["mitm_observed_pii"])
    assert mitm_types & {"location", "mac_address", "log_info"}
    assert result["summary"]["runtime_primary_source"] in {"mitm", "mitm+frida"}
    assert result["summary"]["coverage_rate_applicable"] is True
    assert "location" in result["missing_from_copy"] or "mac_address" in result["missing_from_copy"]
