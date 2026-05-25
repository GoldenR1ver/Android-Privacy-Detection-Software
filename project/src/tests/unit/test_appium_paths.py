"""Unit tests for policy navigation path extraction."""

from privacy_audit.analyzers.dynamic.appium_runner import (
    _ME_TAB_LABELS,
    _candidate_paths_from_hints,
    _extract_navigation_tokens,
    _extract_policy_path_hints,
    _is_me_tab_token,
    _is_sidebar_token,
    _keyword_checks_for_step,
    _labels_for_token,
    _resolve_navigation_paths,
)
from privacy_audit.analyzers.dynamic.navigation_path_builder import SIDEBAR_TOKEN


KEEP_PRIVACY_HINT = (
    "提示您注意：您可以随时在Keep App 中的「我」的页面左上角-「设置」-「隐私设置」"
    "-「管理设备隐私授权」页面中，撤回/拒绝授予Keep访问的设备权限。"
)


def test_extract_tokens_from_keep_privacy_sentence() -> None:
    tokens = _extract_navigation_tokens(KEEP_PRIVACY_HINT)
    assert tokens[0] == "我"
    assert "设置" in tokens
    assert "隐私设置" in tokens
    assert "管理设备隐私授权" in tokens


def test_candidate_paths_start_with_me() -> None:
    hints = _extract_policy_path_hints(KEEP_PRIVACY_HINT + "\n" + KEEP_PRIVACY_HINT)
    bundle = _resolve_navigation_paths(hints)
    paths = bundle["navigation_paths"]
    assert paths
    assert paths[0][0] == "我"
    assert SIDEBAR_TOKEN not in paths[0]
    assert "设置" in paths[0]
    assert bundle["navigation_path_records"][0]["sidebar_injected"] is False


def test_is_sidebar_token() -> None:
    assert _is_sidebar_token("侧边栏")
    assert not _is_sidebar_token("设置")


def test_merge_privacy_setting_token() -> None:
    line = '您可以通过左上角「我」-左上角-「设置」-「隐私」设置、访问、更正隐私信息'
    tokens = _extract_navigation_tokens(line)
    assert tokens[0] == "我"
    assert "隐私设置" in tokens


def test_me_tab_labels_prefers_wo_de_and_aliases() -> None:
    labels = _labels_for_token("我")
    assert labels[0] == "我的"
    assert labels == ["我的", "我"]


def test_is_me_tab_token() -> None:
    assert _is_me_tab_token("我")
    assert _is_me_tab_token("我的")
    assert _is_me_tab_token("个人主页")
    assert not _is_me_tab_token("设置")


def test_keyword_checks_include_wo_de() -> None:
    kws = _keyword_checks_for_step("我")
    assert "我的" in kws
    assert "个人" in kws
