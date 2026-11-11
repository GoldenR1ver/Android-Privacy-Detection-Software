"""Unit tests for R1 trace config sync."""

from __future__ import annotations

from pathlib import Path

from privacy_audit.config.trace_config import (
    build_txt_from_yaml,
    check_trace_yaml_consistency,
    clazz_to_pma_path,
    parse_trace_methods_txt,
    pma_line_to_target,
    target_to_pma_line,
)


def test_clazz_to_pma_path_inner_class() -> None:
    assert clazz_to_pma_path("android.provider.Settings$Secure") == "android/provider/Settings$Secure"


def test_pma_line_roundtrip() -> None:
    line = "-tracemethod android/telephony/TelephonyManager.getImei  # pii_type=imei"
    row = pma_line_to_target(line)
    assert row is not None
    assert row["clazz"] == "android.telephony.TelephonyManager"
    assert row["method"] == "getImei"
    assert row["pii_type"] == "imei"
    back = target_to_pma_line("imei", row)
    assert back is not None
    assert "TelephonyManager.getImei" in back


def test_build_txt_and_check_consistency(tmp_path: Path) -> None:
    taxonomy = Path(__file__).resolve().parents[2] / "privacy_audit" / "config" / "personal_info.yaml"
    trace_file = tmp_path / "privacy_trace_methods.txt"
    trace_file.write_text(build_txt_from_yaml(taxonomy), encoding="utf-8")
    report = check_trace_yaml_consistency(trace_file, taxonomy)
    assert report["consistent"] is True
    rows = parse_trace_methods_txt(trace_file)
    assert len(rows) >= 20
