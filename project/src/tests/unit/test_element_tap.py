"""Unit tests for hierarchy-aware tap helpers."""

from pathlib import Path

from privacy_audit.analyzers.dynamic.element_tap import (
    _collect_settings_gear_candidates,
    _pick_settings_gear_target,
    _toolbar_icon_role,
    bottom_tab_me_present,
    extract_bracket_policy_tokens,
    label_matches_query,
    profile_page_visible,
    score_element_for_query,
    settings_page_visible,
)
from privacy_audit.utils.ui_hierarchy import UiElement, parse_page_source
from privacy_audit.analyzers.dynamic.appium_runner import (
    _extract_navigation_tokens,
    _extract_policy_path_hints,
    _resolve_navigation_paths,
)
from privacy_audit.utils.ui_hierarchy import UiElement


def test_label_matches_query_rejects_my_credit_card_for_wo_de() -> None:
    assert label_matches_query("我的", "我的", "")
    assert not label_matches_query("我的", "我的信用卡", "")
    assert not label_matches_query("我的", "", "我的信用卡")
    assert label_matches_query("设置", "设置", "")


def test_label_matches_query_rejects_prefix_for_wo() -> None:
    assert label_matches_query("我", "我", "")
    assert label_matches_query("我", " 我 ", "")
    assert not label_matches_query("我", "我的", "")


def test_extract_bracket_policy_tokens_dash_and_arrow() -> None:
    assert extract_bracket_policy_tokens(
        "世纪佳缘APP【我】-【设置】-【隐私设置】"
    ) == ["我", "设置", "隐私设置"]
    assert extract_bracket_policy_tokens(
        "小荷健康APP【我的】-【设置】-【关于小荷】"
    ) == ["我的", "设置", "关于小荷"]
    assert extract_bracket_policy_tokens(
        "您可通过APP上【我 → 点击头像 → 维护资料】"
    ) == ["我", "点击头像", "维护资料"]


def test_extract_navigation_tokens_from_bracket_policy() -> None:
    hint = "您可以在小荷健康APP【我的】-【设置】-【关于小荷】查看本隐私政策"
    tokens = _extract_navigation_tokens(hint)
    assert tokens[:3] == ["我的", "设置", "关于小荷"]


def test_policy_hints_include_bracket_paths() -> None:
    policy = (
        "您可以在小荷健康APP【我的】-【设置】-【关于小荷】查看本隐私政策。\n"
        "世纪佳缘APP【我】-【设置】-【隐私设置】-【撤回授权】。"
    )
    hints = _extract_policy_path_hints(policy)
    assert any("【我的】" in h for h in hints)
    assert any("【我】" in h for h in hints)


def test_score_element_prefers_shorter_bottom_tab_label() -> None:
    bottom = UiElement(
        tag="node",
        text="我的",
        content_desc="",
        resource_id="tab_me",
        clazz="TextView",
        bounds=(900, 1800, 1000, 1900),
        clickable=True,
        displayed=True,
        enabled=True,
    )
    false_positive = UiElement(
        tag="node",
        text="我的信用卡",
        content_desc="",
        resource_id="card",
        clazz="TextView",
        bounds=(100, 400, 500, 500),
        clickable=True,
        displayed=True,
        enabled=True,
    )
    s_bottom = score_element_for_query(
        "我的",
        bottom,
        screen_width=1080,
        screen_height=1920,
        prefer_bottom_tab=True,
    )
    s_false = score_element_for_query(
        "我的",
        false_positive,
        screen_width=1080,
        screen_height=1920,
        prefer_bottom_tab=True,
    )
    assert s_bottom >= 0
    assert s_false < 0


def test_resolve_paths_for_xiaohe_bracket_policy() -> None:
    hint = "您可以通过小荷健康APP【我的】-【设置】-【帐号与安全】开启或关闭敏感权限。"
    bundle = _resolve_navigation_paths([hint])
    paths = bundle["navigation_paths"]
    assert paths
    assert paths[0][0] == "我的"
    assert "设置" in paths[0]


def test_profile_page_visible_eastmoney_user_page() -> None:
    xml = Path(
        "privacy_audit/utils/screenGrab/output/20260525_045855_UserPage/hierarchy.xml"
    )
    if not xml.exists():
        return
    assert profile_page_visible(xml.read_text(encoding="utf-8"))


def test_profile_page_not_visible_eastmoney_mainpage() -> None:
    xml = Path(
        "privacy_audit/utils/screenGrab/output/20260525_045822_Mainpage/hierarchy.xml"
    )
    if not xml.exists():
        return
    source = xml.read_text(encoding="utf-8")
    assert "riv_avatar" in source
    assert not profile_page_visible(source)


def test_jiayuan_recommend_has_bottom_me_not_profile() -> None:
    for rel in (
        "privacy_audit/utils/screenGrab/output/20260525_060258_RecommendPage/hierarchy.xml",
        "privacy_audit/utils/screenGrab/output/20260525_051338_RecommendPage/hierarchy.xml",
    ):
        xml = Path(rel)
        if not xml.exists():
            continue
        source = xml.read_text(encoding="utf-8")
        assert bottom_tab_me_present(source)
        assert not profile_page_visible(source)
        return


def test_jiayuan_user_page_is_profile() -> None:
    for rel in (
        "privacy_audit/utils/screenGrab/output/20260525_060311_UserPage/hierarchy.xml",
        "privacy_audit/utils/screenGrab/output/20260525_051350_UserPage/hierarchy.xml",
    ):
        xml = Path(rel)
        if not xml.exists():
            continue
        assert profile_page_visible(xml.read_text(encoding="utf-8"))
        return


def test_avatar_blocklist_rejects_card_avatar() -> None:
    from privacy_audit.analyzers.dynamic.element_tap import _avatar_resource_blocked

    assert _avatar_resource_blocked("com.jiayuan:id/img_card_avatar")
    assert not _avatar_resource_blocked("com.jiayuan:id/iv_mine_header")


def test_toolbar_icon_role_from_labels() -> None:
    assert _toolbar_icon_role("", "设置") == "settings"
    assert _toolbar_icon_role("", "扫一扫") == "scan"
    assert _toolbar_icon_role("", "我的收藏") == "skip"


def test_pick_settings_gear_prefers_rightmost_when_unlabeled() -> None:
    scan = UiElement(
        tag="node",
        text="",
        content_desc="",
        resource_id="",
        clazz="android.widget.ImageView",
        bounds=(919, 70, 968, 119),
        clickable=True,
        displayed=True,
        enabled=True,
    )
    settings = UiElement(
        tag="node",
        text="",
        content_desc="",
        resource_id="",
        clazz="android.widget.ImageView",
        bounds=(996, 70, 1045, 119),
        clickable=True,
        displayed=True,
        enabled=True,
    )
    picked = _pick_settings_gear_target([scan, settings])
    assert picked is settings
    assert picked.bounds[0] >= 996


def test_pick_settings_gear_prefers_labeled_settings_over_position() -> None:
    scan = UiElement(
        tag="node",
        text="",
        content_desc="扫一扫",
        resource_id="",
        clazz="android.widget.ImageView",
        bounds=(919, 70, 968, 119),
        clickable=True,
        displayed=True,
        enabled=True,
    )
    settings = UiElement(
        tag="node",
        text="",
        content_desc="设置",
        resource_id="",
        clazz="android.widget.ImageView",
        bounds=(996, 70, 1045, 119),
        clickable=True,
        displayed=True,
        enabled=True,
    )
    gears = _collect_settings_gear_candidates(
        [scan, settings], screen_width=1080, screen_height=1920
    )
    assert len(gears) == 1
    assert _pick_settings_gear_target(gears) is settings


def test_xiaohe_user_page_has_settings_gear_candidate() -> None:
    xml = Path(
        "privacy_audit/utils/screenGrab/output/20260525_061844_UserPage/hierarchy.xml"
    )
    if not xml.exists():
        return
    source = xml.read_text(encoding="utf-8")
    assert not settings_page_visible(source)
    elements = parse_page_source(source)
    gears = _collect_settings_gear_candidates(elements, screen_width=1080, screen_height=1920)
    assert len(gears) >= 2
    picked = _pick_settings_gear_target(gears)
    assert picked is not None
    assert picked.bounds
    assert picked.bounds[0] >= 996
    assert picked.bounds[1] <= 130


def test_xiaohe_setting_page_visible() -> None:
    xml = Path(
        "privacy_audit/utils/screenGrab/output/20260525_061859_SettingPage/hierarchy.xml"
    )
    if not xml.exists():
        return
    assert settings_page_visible(xml.read_text(encoding="utf-8"))
