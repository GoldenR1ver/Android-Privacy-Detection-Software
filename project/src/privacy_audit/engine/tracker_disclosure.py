"""Compare embedded tracker/SDK detections against privacy policy disclosure."""

from __future__ import annotations

import re
from typing import Any

# Policy text aliases for common tracker brands (Chinese + English).
_TRACKER_POLICY_ALIASES: dict[str, list[str]] = {
    "umeng": ["友盟", "umeng"],
    "bugly": ["bugly", "腾讯 bugly"],
    "adjust": ["adjust"],
    "facebook": ["facebook", "脸书", "meta"],
    "firebase": ["firebase", "谷歌分析", "google analytics"],
    "branch": ["branch"],
    "sensors analytics": ["神策", "sensorsdata", "sensors analytics"],
    "talkingdata": ["talkingdata", "talking data"],
    "appsflyer": ["appsflyer"],
    "jpush": ["极光", "jpush", "jiguang"],
    "getui": ["个推", "getui", "igexin"],
    "alipay": ["支付宝", "alipay"],
    "wechat": ["微信", "wechat", "weixin"],
    "baidu mobads": ["百度", "baidu mobads", "百度统计"],
    "tencent gdt": ["广点通", "gdt"],
    "pangle": ["穿山甲", "pangle", "bytedance"],
    "msa oaid": ["移动安全联盟", "oaid", "miit"],
}


def _policy_mentions_tracker(policy_text: str, tracker_name: str) -> bool:
    text = (policy_text or "").lower()
    if not text or not tracker_name:
        return False
    name_low = tracker_name.lower().strip()
    if name_low and name_low in text:
        return True
    for alias in _TRACKER_POLICY_ALIASES.get(name_low, []):
        if alias.lower() in text:
            return True
    # Token match: "TalkingData" -> talkingdata
    compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", name_low)
    if len(compact) >= 4 and compact in re.sub(r"\s+", "", text):
        return True
    return False


def _policy_mentions_third_party_section(policy_text: str) -> bool:
    text = policy_text or ""
    markers = (
        "第三方",
        "SDK",
        "sdk",
        "共享清单",
        "合作伙伴",
        "嵌入",
    )
    return any(m in text for m in markers)


def compare_embedded_trackers_with_policy(
    policy_text: str,
    embedded_trackers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Disclosure supplement: SDK embedded in APK vs named in privacy policy."""
    rows: list[dict[str, Any]] = []
    declared: list[str] = []
    undeclared: list[str] = []

    for tr in embedded_trackers:
        if not isinstance(tr, dict):
            continue
        name = str(tr.get("name") or "").strip()
        if not name:
            continue
        mentioned = _policy_mentions_tracker(policy_text, name)
        row = {
            "tracker_id": str(tr.get("id", "")),
            "name": name,
            "categories": list(tr.get("categories") or []),
            "matched_signatures": list(tr.get("matched_signatures") or []),
            "sample_classes": list(tr.get("sample_classes") or [])[:3],
            "declared_in_policy": mentioned,
            "evidence_level": "S1_static_embedded",
        }
        rows.append(row)
        if mentioned:
            declared.append(name)
        else:
            undeclared.append(name)

    has_sdk_section = _policy_mentions_third_party_section(policy_text)
    risk = "low"
    if undeclared:
        risk = "medium" if has_sdk_section else "high"

    return {
        "schema_version": 1,
        "has_third_party_sdk_section": has_sdk_section,
        "embedded_count": len(rows),
        "declared_in_policy_count": len(declared),
        "undeclared_count": len(undeclared),
        "declared_trackers": sorted(declared),
        "undeclared_trackers": sorted(undeclared),
        "risk_level": risk,
        "trackers": rows,
        "summary_message": (
            f"{len(undeclared)} embedded tracker(s) not clearly named in policy"
            if undeclared
            else "All embedded trackers appear mentioned in policy text"
        ),
    }


def build_sdk_disclosure_verdicts(sdk_disclosure: dict[str, Any]) -> list[dict[str, Any]]:
    """Map embedded SDK rows into disclosure verdict entries."""
    verdicts: list[dict[str, Any]] = []
    for tr in sdk_disclosure.get("trackers") or []:
        if not isinstance(tr, dict):
            continue
        name = str(tr.get("name") or "").strip()
        if not name:
            continue
        declared = bool(tr.get("declared_in_policy"))
        verdicts.append(
            {
                "pii_type": f"embedded_sdk:{name}",
                "status": "consistent" if declared else "inconsistent_high_risk",
                "risk_level": "low" if declared else "high",
                "message": (
                    f"Embedded SDK '{name}' appears named in privacy policy."
                    if declared
                    else f"Embedded SDK '{name}' detected in APK but not clearly named in policy."
                ),
                "evidence_fields": ["sdk_disclosure.trackers", "embedded_trackers"],
                "verdict_basis": "sdk_disclosure",
                "evidence_level": str(tr.get("evidence_level") or "S1_static_embedded"),
            }
        )
    return verdicts


def merge_sdk_verdicts_into_compare_result(
    compare_result: dict[str, Any],
    sdk_disclosure: dict[str, Any],
) -> dict[str, Any]:
    """Append SDK disclosure verdicts and refresh summary counts."""
    sdk_verdicts = build_sdk_disclosure_verdicts(sdk_disclosure)
    if not sdk_verdicts:
        return compare_result

    verdicts = list(compare_result.get("verdicts") or [])
    verdicts.extend(sdk_verdicts)
    compare_result["verdicts"] = verdicts

    summary = dict(compare_result.get("summary") or {})
    status_counts = dict(summary.get("status_counts") or {})
    for row in sdk_verdicts:
        key = str(row.get("status", ""))
        status_counts[key] = status_counts.get(key, 0) + 1
    summary["status_counts"] = status_counts
    summary["sdk_disclosure_count"] = len(sdk_verdicts)
    summary["total_compared"] = int(summary.get("total_compared", 0)) + len(sdk_verdicts)
    compare_result["summary"] = summary
    return compare_result
