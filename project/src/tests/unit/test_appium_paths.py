"""Unit tests for policy navigation path extraction."""

from privacy_audit.analyzers.dynamic.appium_runner import (
    _ME_TAB_LABELS,
    _candidate_paths_from_hints,
    _expand_implicit_settings_steps,
    _extract_navigation_tokens,
    _extract_policy_path_hints,
    _intermediate_bridges_for_token,
    _is_me_tab_token,
    _is_sidebar_token,
    _keyword_checks_for_step,
    _labels_for_token,
    _llm_nav_eligible,
    _resolve_navigation_paths,
    _step_keyword_matched,
)
from privacy_audit.analyzers.dynamic.navigation_bridge_store import NavigationBridgeStore
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


XIAOHE_DOWNLOAD_HINT = (
    "您可以通过小荷健康APP【我的】-【设置】-【帐号与安全】-【个人信息下载】"
    "对您的个人信息进行复制，导出并下载您的头像、用户名、简介、手机号、生日、性别、地区等个人信息和发布的文章内容。"
)
XIAOHE_BASIC_HINT = (
    "您可以通过小荷健康APP【我的】-【设置】-【帐号与安全】-【切换基本版】了解并使用基本功能模式。"
)
XIAOHE_ACCOUNT_HINT = (
    "您可通过小荷健康APP【我的】-【设置】-【帐号与安全】对选择关闭或开启个性化推荐功能。"
)


def test_xiaohe_includes_personal_info_download_path() -> None:
    hints = _extract_policy_path_hints(
        "\n".join([XIAOHE_ACCOUNT_HINT, XIAOHE_BASIC_HINT, XIAOHE_DOWNLOAD_HINT]),
        max_count=12,
    )
    bundle = _resolve_navigation_paths(hints, max_paths=3)
    paths = bundle["navigation_paths"]
    assert len(paths) == 3
    assert any("个人信息下载" in step for path in paths for step in path)
    assert any("切换基本版" in step for path in paths for step in path)


XHS_NOTIFY_PRIVACY_HINT = (
    "您可通过【我-设置-通知设置/隐私/通用】变更或撤回授权范围，"
    "包括：1）微信账号、微博账号和QQ账号的绑定；"
)


def test_xhs_slash_token_keeps_single_step_with_label_aliases() -> None:
    tokens = _extract_navigation_tokens(XHS_NOTIFY_PRIVACY_HINT)
    assert tokens == ["我", "设置", "通知设置/隐私/通用"]
    labels = _labels_for_token("通知设置/隐私/通用")
    assert labels[:3] == ["通知设置", "隐私", "通用"]
    assert "隐私设置" in labels
    checks = {"隐私": True, "设置": False}
    assert _step_keyword_matched("通知设置/隐私/通用", checks)


def test_expand_implicit_settings_before_user_protection_center() -> None:
    assert _expand_implicit_settings_steps(["我", "用户保护中心", "隐私设置"]) == [
        "我",
        "设置",
        "用户保护中心",
        "隐私设置",
    ]
    assert _expand_implicit_settings_steps(["我", "设置", "用户保护中心"]) == [
        "我",
        "设置",
        "用户保护中心",
    ]


def test_llm_nav_eligible_after_bridge_fail_even_with_sidebar() -> None:
    assert _llm_nav_eligible(bridge_tried_failed=True, sidebar_has_trigger=True)
    assert _llm_nav_eligible(bridge_tried_failed=False, sidebar_has_trigger=False)
    assert not _llm_nav_eligible(bridge_tried_failed=False, sidebar_has_trigger=True)


def test_privacy_settings_includes_common_privacy_alias() -> None:
    labels = _labels_for_token("隐私设置")
    assert labels[0] == "隐私设置"
    assert "常用隐私设置" in labels
    assert labels.index("隐私设置") < labels.index("常用隐私设置")


def test_xhs_recommend_friend_negation_uses_affirmative_alias() -> None:
    labels = _labels_for_token("不把我推荐给可能认识的人")
    assert labels[0] == "不把我推荐给可能认识的人"
    assert "把我推荐给可能认识的人" in labels
    assert labels.index("不把我推荐给可能认识的人") < labels.index("把我推荐给可能认识的人")
    store = NavigationBridgeStore()
    store.load()
    assert _intermediate_bridges_for_token("不把我推荐给可能认识的人", store) == (
        "找到我的方式",
    )
    checks = {"把我推荐给可能认识的人": True}
    assert _step_keyword_matched("不把我推荐给可能认识的人", checks)
