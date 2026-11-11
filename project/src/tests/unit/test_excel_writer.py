"""Unit tests for PMA Excel export."""

from __future__ import annotations

import json
from pathlib import Path

from privacy_audit.reporting.excel_writer import PMA_COLUMNS, write_excel
from privacy_audit.workflows.excel_export import build_pma_excel_rows


def test_build_pma_rows_from_static(tmp_path: Path) -> None:
    static_dir = tmp_path / "static" / "diaoyuren"
    static_dir.mkdir(parents=True)
    static_dir.joinpath("static_evidence.json").write_text(
        json.dumps(
            {
                "androguard": {
                    "api_hits": {
                        "imei": [{"matched_api": "getImei", "context": "dex_hit"}]
                    }
                },
                "embedded_trackers": [
                    {"name": "Umeng", "matched_signatures": ["com.umeng"], "sample_classes": ["com.umeng.Foo"]}
                ],
            }
        ),
        encoding="utf-8",
    )
    bundle = build_pma_excel_rows(
        "钓鱼人",
        policy_text="我们收集 IMEI 与友盟统计。",
        static_evidence_dir=tmp_path / "static",
    )
    rows = bundle["rows"]
    assert len(rows) >= 2
    imei_rows = [r for r in rows if r["pii_type"] == "imei"]
    assert imei_rows and imei_rows[0]["declared_in_policy"] == "yes"
    assert any(r["evidence_level"] == "S1_static_embedded" for r in rows)


def test_write_excel_csv_fallback(tmp_path: Path) -> None:
    rows = [
        {
            "app_name": "Test",
            "pii_type": "imei",
            "api_name": "TelephonyManager.getImei",
            "stack_summary": "at com.example.Foo.bar",
            "declared_in_policy": "no",
            "evidence_level": "S3_runtime_frida",
            "args_summary": "",
            "hook_file": "hook_imei.js",
            "source": "dynamic.frida",
        }
    ]
    out = tmp_path / "test.xlsx"
    result = write_excel(rows, out)
    assert result["row_count"] == 1
    path = Path(result["path"])
    assert path.exists()
    if result["format"] == "csv":
        text = path.read_text(encoding="utf-8-sig")
        assert "stack_summary" in text
        assert PMA_COLUMNS[0] in text
