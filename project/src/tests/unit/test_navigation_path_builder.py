"""Unit tests for policy navigation path evidence (sidebar rules)."""

from privacy_audit.analyzers.dynamic.navigation_path_builder import (
    SIDEBAR_RUNTIME_REASON,
    SIDEBAR_TOKEN,
    build_navigation_path_evidence,
    canonicalize_sidebar_aliases,
    hint_requires_sidebar,
    inject_sidebar_into_path,
    is_sidebar_token,
    path_dedupe_key,
    strip_sidebar_steps,
)


KEEP_HINT = (
    "提示您注意：您可以随时在Keep App 中的「我」的页面左上角-「设置」-「隐私设置」"
    "-「管理设备隐私授权」页面中，撤回/拒绝授予Keep访问的设备权限。"
)

XHS_HINT = (
    "您可在【我】-点击左上角-【设置】-【隐私设置】-【个性化选项】关闭个性化推荐机制"
)


def test_hint_requires_sidebar_markers() -> None:
    assert hint_requires_sidebar("点击左上角-设置")
    assert hint_requires_sidebar("界面内设置")
    assert hint_requires_sidebar("右上角入口")
    assert hint_requires_sidebar("点击更多进入设置")
    assert not hint_requires_sidebar("直接进入设置")


def test_more_is_sidebar_alias() -> None:
    assert is_sidebar_token("更多")
    assert is_sidebar_token("侧边栏")
    assert canonicalize_sidebar_aliases(["我", "更多", "设置"]) == [
        "我",
        SIDEBAR_TOKEN,
        "设置",
    ]


def test_no_proactive_sidebar_injection() -> None:
    path = ["我", "设置", "隐私设置"]
    expanded, record = inject_sidebar_into_path(path, KEEP_HINT)
    assert expanded == ["我", "设置", "隐私设置"]
    assert record["sidebar_injected"] is False
    assert record["sidebar_reason"] == SIDEBAR_RUNTIME_REASON
    assert record["policy_corner_marker"] == "左上角"


def test_strip_sidebar_for_dedupe() -> None:
    a = path_dedupe_key(["我", "设置", "隐私设置"])
    b = path_dedupe_key(["我", SIDEBAR_TOKEN, "设置", "隐私设置"])
    assert a == b


def test_build_navigation_path_evidence_no_sidebar_in_paths() -> None:
    bundle = build_navigation_path_evidence(
        [["我", "设置", "隐私设置"]],
        [KEEP_HINT, XHS_HINT],
        max_paths=1,
    )
    assert bundle["navigation_paths"]
    assert SIDEBAR_TOKEN not in bundle["navigation_paths"][0]
    rec = bundle["navigation_path_records"][0]
    assert rec["sidebar_injected"] is False
    assert rec["expanded_path"] == bundle["navigation_paths"][0]
