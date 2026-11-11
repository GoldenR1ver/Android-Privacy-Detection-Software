"""Unit tests for R2 tracker DEX scanner and SDK disclosure."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from privacy_audit.analyzers.static.tracker_blocklist_loader import (
    build_domain_index,
    normalize_tracker_entries,
)
from privacy_audit.analyzers.static.tracker_dex_scanner import scan_embedded_trackers
from privacy_audit.engine.tracker_disclosure import (
    build_sdk_disclosure_verdicts,
    compare_embedded_trackers_with_policy,
    merge_sdk_verdicts_into_compare_result,
)


def _make_apk_with_dex(tmp_path: Path, class_descriptors: list[str]) -> Path:
  """Build a minimal APK zip containing one dex with L...; class strings."""
  dex_body = b"".join(desc.encode("utf-8") for desc in class_descriptors)
  apk_path = tmp_path / "sample.apk"
  with zipfile.ZipFile(apk_path, "w") as zf:
    zf.writestr("classes.dex", dex_body)
  return apk_path


@pytest.fixture
def blocklist_dir(tmp_path: Path) -> Path:
  trackers = [
    {
      "id": "umeng",
      "name": "Umeng",
      "categories": ["analytics"],
      "code_signatures": ["com.umeng.analytics"],
      "network_hosts": ["umeng.com"],
    },
    {
      "id": "bugly",
      "name": "Bugly",
      "categories": ["crash_reporting"],
      "code_signatures": ["com.tencent.bugly"],
      "network_hosts": ["bugly.qq.com"],
    },
  ]
  root = tmp_path / "tracker_blocklist"
  root.mkdir()
  (root / "sdk_signatures.json").write_text(
    json.dumps({"schema_version": 1, "source": "test", "trackers": trackers}),
    encoding="utf-8",
  )
  return root


def test_scan_embedded_trackers_matches_dex_classes(
    tmp_path: Path, blocklist_dir: Path
) -> None:
  apk = _make_apk_with_dex(
    tmp_path,
    [
      "Lcom/umeng/analytics/MobclickAgent;",
      "Lcom/tencent/bugly/crashreport/CrashReport;",
      "Lcom/example/app/MainActivity;",
    ],
  )
  result = scan_embedded_trackers(apk, blocklist_dir=blocklist_dir)

  assert result["embedded_tracker_count"] == 2
  names = {row["name"] for row in result["embedded_trackers"]}
  assert names == {"Umeng", "Bugly"}
  umeng = next(r for r in result["embedded_trackers"] if r["name"] == "Umeng")
  assert "com.umeng.analytics" in umeng["matched_signatures"]


def test_build_domain_index_deduplicates_hosts() -> None:
  trackers = normalize_tracker_entries(
    {
      "trackers": [
        {"id": "a", "name": "A", "code_signatures": ["com.a"], "network_hosts": ["a.com"]},
        {"id": "b", "name": "B", "code_signatures": ["com.b"], "network_hosts": ["a.com", "b.com"]},
      ]
    }
  )
  rows = build_domain_index(trackers)
  hosts = [r["host"] for r in rows]
  assert hosts.count("a.com") == 1
  assert "b.com" in hosts


def test_compare_embedded_trackers_with_policy() -> None:
  policy = "我们使用友盟统计与 Bugly 崩溃分析服务。"
  embedded = [
    {"id": "umeng", "name": "Umeng", "matched_signatures": ["com.umeng"]},
    {"id": "adjust", "name": "Adjust", "matched_signatures": ["com.adjust"]},
  ]
  out = compare_embedded_trackers_with_policy(policy, embedded)
  assert out["declared_in_policy_count"] == 1
  assert "Adjust" in out["undeclared_trackers"]
  assert "Umeng" in out["declared_trackers"]


def test_merge_sdk_verdicts_into_compare_result() -> None:
  sdk = compare_embedded_trackers_with_policy(
    "第三方 SDK 清单：友盟",
    [{"id": "umeng", "name": "Umeng"}, {"id": "adjust", "name": "Adjust"}],
  )
  compare = {"verdicts": [], "summary": {"status_counts": {}, "total_compared": 0}}
  merged = merge_sdk_verdicts_into_compare_result(compare, sdk)
  assert merged["summary"]["sdk_disclosure_count"] == 2
  statuses = {v["status"] for v in merged["verdicts"]}
  assert "inconsistent_high_risk" in statuses
  assert "consistent" in statuses
  assert any(v["pii_type"] == "embedded_sdk:Adjust" for v in build_sdk_disclosure_verdicts(sdk))
