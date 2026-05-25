"""Test case generator: reads Part 1 audit_processed.csv and produces test_cases.json.

Bridges Part 1 (LLM text analysis) → Part 2 (Appium dynamic testing).
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from .models import TestCase
from .config import (
    TARGET_APPS,
    APP_PKG_MAP,
    CONSENT_RIGHTS,
    ACCESS_RIGHTS,
    DELETION_RIGHTS,
    DEFAULT_LIMIT_PER_APP_PER_RIGHT,
)


def _assign_priority(channel: str, risk: str, path_text: str) -> int:
    """Assign test priority 1-3 based on channel, risk, and path clarity."""
    if channel == "app_ui" and risk == "low" and path_text.strip():
        return 1
    if channel == "app_ui" and risk == "medium":
        return 2
    return 3


def _classify_right_bucket(right_types_str: str) -> list[str]:
    """Parse right_types JSON array and map to our three buckets."""
    buckets = []
    try:
        rights = json.loads(right_types_str) if isinstance(right_types_str, str) else right_types_str
    except (json.JSONDecodeError, TypeError):
        return buckets

    if not isinstance(rights, list):
        return buckets

    for r in rights:
        if r in CONSENT_RIGHTS:
            if "consent" not in buckets:
                buckets.append("consent")
        if r in ACCESS_RIGHTS:
            if "access" not in buckets:
                buckets.append("access")
        if r in DELETION_RIGHTS:
            if "deletion" not in buckets:
                buckets.append("deletion")
    return buckets


def _deduplicate_by_path(cases: list[TestCase]) -> list[TestCase]:
    """Merge test cases that share the same path_text for the same right_type."""
    seen: dict[str, TestCase] = {}
    for tc in cases:
        # De-duplicate key: same path_text + same right_type means same UI interaction
        key = f"{tc.right_type}|{tc.path_text}"
        if key not in seen:
            seen[key] = tc
        else:
            # Keep the higher-priority one (lower number = higher priority)
            if tc.priority < seen[key].priority:
                seen[key] = tc
    return list(seen.values())


_PATH_SEP_PATTERN = re.compile(r"\s*[-→>/]+\s*")
_QUOTE_PATTERN = re.compile(r"[「《]([^」》]+)[」》]|\"([^\"]+)\"")
_PATH_CLEANUP_PATTERN = re.compile(r"[（(][^）)]*[）)]")
_NON_AUTOMATABLE_HINTS = (
    "隐私政策",
    "第",
    "条",
    "本政策",
    "阅读",
    "客服热线",
    "发送邮件",
    "电话联系",
    "人工",
)


def _normalize_path_text(path_text: str) -> tuple[str, bool]:
    """Normalize Part1 path text into executable app navigation segments."""
    raw = (path_text or "").strip()
    if not raw:
        return "", False

    cleaned = _PATH_CLEANUP_PATTERN.sub("", raw).strip()
    quoted = _QUOTE_PATTERN.findall(cleaned)
    if quoted:
        tokens: list[str] = []
        for a, b in quoted:
            t = (a or b).strip()
            if not t:
                continue
            if len(t) > 18:
                pieces = [p.strip() for p in re.split(r"[，。、；：/\s]+", t) if 2 <= len(p.strip()) <= 12]
                tokens.extend(pieces[:3] or [t[:12]])
            else:
                tokens.append(t)
        uniq = []
        for t in tokens:
            if t not in uniq:
                uniq.append(t)
        if uniq:
            return " - ".join(uniq[:6]), True

    segments = [s.strip() for s in _PATH_SEP_PATTERN.split(cleaned) if s.strip()]
    short_segments = [s for s in segments if 1 <= len(s) <= 18]
    if len(short_segments) >= 1:
        normalized = " - ".join(short_segments[:6])
        executable = not any(h in normalized for h in _NON_AUTOMATABLE_HINTS)
        return normalized, executable

    pieces = [p.strip() for p in re.split(r"[，。、；：/\s]+", cleaned) if 2 <= len(p.strip()) <= 10]
    if pieces:
        normalized = " - ".join(pieces[:4])
        executable = not any(h in cleaned for h in _NON_AUTOMATABLE_HINTS)
        return normalized, executable

    return cleaned[:24], False


def generate_test_cases(
    pipeline_root: Path,
    apps: list[str] | None = None,
    rights: list[str] | None = None,
    limit_per_right: int = DEFAULT_LIMIT_PER_APP_PER_RIGHT,
) -> list[TestCase]:
    """Main entry: scan all apps' audit_processed.csv and generate TestCase list.

    Args:
        pipeline_root: Path to the Part 1 run output directory.
        apps: App names to include (default: TARGET_APPS from config).
        rights: Right buckets to include: consent, access, deletion (default: all).
        limit_per_right: Max test cases per app per right type.

    Returns:
        Sorted list of TestCase objects (P1 first).
    """
    target_apps = apps or TARGET_APPS
    target_rights = set(rights or ["consent", "access", "deletion"])

    all_cases: list[TestCase] = []

    for app_name in target_apps:
        app_dir = pipeline_root / app_name
        csv_path = app_dir / "audit_processed.csv"
        if not csv_path.exists():
            print(f"  [WARN] No audit_processed.csv for {app_name}, skipping")
            continue

        app_pkg = APP_PKG_MAP.get(app_name, "")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Partition by right bucket
        buckets: dict[str, list[TestCase]] = defaultdict(list)

        for idx, row in enumerate(rows):
            if row.get("app_test_candidate", "0") != "1":
                continue

            right_buckets = _classify_right_bucket(row.get("right_types", "[]"))

            for bucket in right_buckets:
                if bucket not in target_rights:
                    continue

                channel = "app_ui"
                channels_str = row.get("execution_channels", "[]")
                try:
                    channels = json.loads(channels_str)
                    # Use the most automatable channel
                    if "app_ui" in channels:
                        channel = "app_ui"
                    elif "web" in channels:
                        channel = "web"
                    else:
                        channel = channels[0] if channels else "unknown"
                except (json.JSONDecodeError, IndexError):
                    pass

                raw_path_text = row.get("path_text", "").strip()
                path_text, executable = _normalize_path_text(raw_path_text)
                risk = row.get("usability_risk", "unknown").strip()
                priority = _assign_priority(channel, risk, path_text)
                if channel == "app_ui" and not executable:
                    # Route policy-only or ambiguous instructions to manual review.
                    channel = "unknown"
                    priority = _assign_priority(channel, risk, path_text)

                tc = TestCase(
                    case_id=f"{app_name}-{bucket}-{idx:04d}",
                    app_name=app_name,
                    app_pkg=app_pkg,
                    right_type=bucket,
                    channel=channel,
                    path_text=path_text,
                    dynamic_test_goal=row.get("dynamic_test_goal", "").strip(),
                    usability_risk=risk,
                    target_data=row.get("target_data", "").strip(),
                    sentence=row.get("privacy_policy_content", "").strip(),
                    priority=priority,
                    access_copy_type=row.get("access_copy_type", "unknown").strip(),
                )
                buckets[bucket].append(tc)

        # Deduplicate and limit per bucket
        for bucket, cases in buckets.items():
            deduped = _deduplicate_by_path(cases)
            deduped.sort(key=lambda c: c.priority)
            all_cases.extend(deduped[:limit_per_right])

    # Global sort by priority then app name
    all_cases.sort(key=lambda c: (c.priority, c.app_name, c.right_type))

    return all_cases


def export_test_cases_json(cases: list[TestCase], output_path: Path) -> dict:
    """Write test cases to JSON and return summary stats."""
    data = {
        "total": len(cases),
        "by_priority": {
            "p1": sum(1 for c in cases if c.priority == 1),
            "p2": sum(1 for c in cases if c.priority == 2),
            "p3": sum(1 for c in cases if c.priority == 3),
        },
        "by_right": {
            "consent": sum(1 for c in cases if c.right_type == "consent"),
            "access": sum(1 for c in cases if c.right_type == "access"),
            "deletion": sum(1 for c in cases if c.right_type == "deletion"),
        },
        "by_channel": {
            ch: sum(1 for c in cases if c.channel == ch)
            for ch in sorted(set(c.channel for c in cases))
        },
        "by_app": {
            app: sum(1 for c in cases if c.app_name == app)
            for app in sorted(set(c.app_name for c in cases))
        },
        "test_cases": [tc.to_dict() for tc in cases],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def export_app_ui_only_json(cases: list[TestCase], output_path: Path) -> dict:
    """Export only app_ui channel test cases for quicker focused runs."""
    ui_cases = [c for c in cases if c.channel == "app_ui"]
    data = {
        "total": len(ui_cases),
        "test_cases": [tc.to_dict() for tc in ui_cases],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
