"""Frida code_prefix resolution and app_packages.json parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from privacy_audit.analyzers.dynamic.frida_runner import (
    _accept_java_probe_state,
    _app_ready_probe,
    _is_display_name_label,
    get_app_frida_tier_entry,
    load_app_frida_config,
    parse_app_package_entry,
    resolve_code_prefixes,
    resolve_frida_launch_mode,
    should_enable_frida_for_consent,
)


def test_resolve_code_prefixes_dedupes_and_orders_package_first() -> None:
    prefixes = resolve_code_prefixes(
        "com.lvsongguo",
        ["com.xiaohe", "com.lvsongguo", "com.bytedance.health"],
    )
    assert prefixes[0] == "com.lvsongguo"
    assert "com.xiaohe" in prefixes
    assert "com.bytedance.health" in prefixes
    assert len(prefixes) == len(set(prefixes))


def test_parse_string_entry_uses_builtin_hints() -> None:
    package, prefixes, _opts = parse_app_package_entry("com.lvsongguo")
    assert package == "com.lvsongguo"
    assert "com.xiaohe" in prefixes


def test_parse_object_entry_merges_explicit_prefixes() -> None:
    package, prefixes, opts = parse_app_package_entry(
        {
            "package": "com.lvsongguo",
            "code_prefixes": ["com.xiaohe", "com.custom"],
        }
    )
    assert opts == {}
    assert package == "com.lvsongguo"
    assert "com.custom" in prefixes
    assert "com.xiaohe" in prefixes


def test_load_app_frida_config_from_json(tmp_path: Path) -> None:
    path = tmp_path / "app_packages.json"
    path.write_text(
        json.dumps(
            {
                "小荷健康": {
                    "package": "com.lvsongguo",
                    "code_prefixes": ["com.xiaohe"],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    package, prefixes, _opts = load_app_frida_config("小荷健康", path)
    assert package == "com.lvsongguo"
    assert "com.xiaohe" in prefixes


def test_app_ready_probe_embeds_prefix_list() -> None:
    script = _app_ready_probe(["com.lvsongguo", "com.xiaohe"])
    assert '"com.xiaohe"' in script
    assert "maxAttempts" in script
    assert "enumerateLoadedClasses" in script
    assert "enumerateLoadedClassesSync" not in script


def test_display_name_label_detection() -> None:
    assert _is_display_name_label("小荷健康", "小荷健康", "com.lvsongguo")
    assert not _is_display_name_label("com.lvsongguo", "小荷健康", "com.lvsongguo")


def test_accept_java_on_mumu_display_process() -> None:
    assert _accept_java_probe_state("java_only", "小荷健康", "小荷健康", "com.lvsongguo")
    assert _accept_java_probe_state("java_timeout", "小荷健康", "小荷健康", "com.lvsongguo")
    assert _accept_java_probe_state(
        "java_timeout",
        "韭研公社",
        "韭研公社",
        "com.jiucaigongshe",
        accept_java_timeout_on_display=True,
    )
    assert not _accept_java_probe_state(
        "java_timeout",
        "pidof:12303",
        "钓鱼人",
        "com.lchr.diaoyu",
        accept_java_timeout_on_display=False,
    )
    assert not _accept_java_probe_state("java_timeout", "com.lvsongguo", "小荷健康", "com.lvsongguo")


def test_parse_invalid_entry_raises() -> None:
    with pytest.raises(ValueError):
        parse_app_package_entry(123)


def test_frida_tier_resolve_launch_mode(tmp_path: Path) -> None:
    tier = tmp_path / "frida_tier.json"
    tier.write_text(
        json.dumps(
            {
                "defaults": {"preferred_launch_mode": "attach"},
                "apps": {
                    "小荷健康": {"preferred_launch_mode": "spawn"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert resolve_frida_launch_mode("小荷健康", tier, "auto") == "spawn"
    assert resolve_frida_launch_mode("小荷健康", tier, "attach") == "attach"
    assert resolve_frida_launch_mode("未知", tier, "auto") == "attach"


def test_should_enable_frida_for_consent_from_tier(tmp_path: Path) -> None:
    tier = tmp_path / "frida_tier.json"
    tier.write_text(
        json.dumps(
            {
                "apps": {
                    "Keep": {"tier": "tier3", "enable_frida_consent": False},
                    "小荷健康": {"tier": "tier1", "enable_frida_consent": True},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert should_enable_frida_for_consent("小荷健康", tier) is False
    assert should_enable_frida_for_consent("Keep", tier) is False
    assert should_enable_frida_for_consent("Keep", tier, cli_enable=True) is True
    assert should_enable_frida_for_consent("小荷健康", tier, cli_enable=True) is True
    assert should_enable_frida_for_consent("小荷健康", tier, cli_disable=True) is False


def test_get_app_frida_tier_entry_merges_defaults(tmp_path: Path) -> None:
    tier = tmp_path / "frida_tier.json"
    tier.write_text(
        json.dumps(
            {
                "defaults": {"frida_grace_sec": 10},
                "apps": {"钓鱼人": {"tier": "unknown", "frida_grace_sec": 45}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    entry = get_app_frida_tier_entry("钓鱼人", tier)
    assert entry["frida_grace_sec"] == 45
    assert entry["tier"] == "unknown"


