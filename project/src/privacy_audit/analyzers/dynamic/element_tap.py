"""Hierarchy-aware tap helpers: exact label match, scroll, profile avatar."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from privacy_audit.utils.ui_hierarchy import UiElement, parse_page_source

logger = logging.getLogger("privacy_audit.appium")

_BOTTOM_TAB_Y_MIN_RATIO = 0.78
_TOP_PROFILE_Y_MAX_RATIO = 0.28
_LEFT_AVATAR_X_MAX_RATIO = 0.22

# Labels where a longer string starting with the query is a false positive (e.g. 我的信用卡).
_ME_QUERY_FALSE_PREFIXES: dict[str, tuple[str, ...]] = {
    "我的": ("我的信用卡", "我的钱包", "我的订单", "我的收藏", "我的主页", "我的服务"),
    "我": ("我们是", "我在", "我的"),
}

_BOTTOM_TAB_ME_LABELS: tuple[str, ...] = ("我", "我的")

_SCROLLABLE_TOKENS = frozenset(
    {
        "设置",
        "隐私设置",
        "隐私",
        "账号与安全",
        "帐号与安全",
        "个人资料",
        "管理设备隐私授权",
        "第三方数据合作",
        "消息与提醒",
        "用户保护中心",
        "关于小荷",
        "关于我们",
        "撤回授权",
        "个人信息收集设置",
    }
)
SCROLLABLE_TOKENS = _SCROLLABLE_TOKENS

_AVATAR_RESOURCE_HINTS = (
    "avatar",
    "riv_avatar",
    "iv_avatar",
    "head",
    "portrait",
    "user_icon",
    "profile",
    "mine_head",
    "user_info",
)
_AVATAR_RESOURCE_IDS_PRIORITY: tuple[str, ...] = (
    "riv_avatar",
    "iv_mine_header",
    "iv_avatar",
    "img_avatar",
)
# Feed / card avatars must not be used for app-level profile entry (世纪佳缘推荐页等).
_AVATAR_RESOURCE_ID_BLOCKLIST: tuple[str, ...] = (
    "card_avatar",
    "img_card",
    "feed_",
    "recommend_",
    "user_card",
)
_IMAGE_CLASS_HINTS = ("ImageView", "ImageButton", "CircleImageView")

# Signals that the profile / "我的" hub page is open (no bottom-tab "我的" label required).
_PROFILE_PAGE_TEXT_MARKERS: tuple[str, ...] = (
    "我的收藏",
    "我的评论",
    "我的服务",
    "我的钱包",
    "我的卡包",
    "个人主页",
    "隐私管理",
    "隐私设置",
    "账号与安全",
    "帐号与安全",
    "个性化推送",
)
_PROFILE_PAGE_RESOURCE_MARKERS: tuple[str, ...] = (
    "container_user_info",
    "container_user_name",
    "tv_my_collection",
    "tv_comment",
    "tv_right_action",
    "tv_user_name",
)
# Main page may also expose riv_avatar — do not treat avatar id alone as profile hub.
_PROFILE_PAGE_STRONG_MARKERS: tuple[str, ...] = (
    "container_user_info",
    "tv_my_collection",
    "iv_mine_header",
    "tv_new_mine_title",
)


def visible_labels(el: UiElement) -> tuple[str, str]:
    return el.text.strip(), el.content_desc.strip()


def _normalize_visible_label(raw: str) -> str:
    """Trim UI text; 世纪佳缘 bottom tab uses padded labels like `` 我 ``."""
    return raw.strip()


def _is_bottom_me_label(text: str, content_desc: str) -> bool:
    for raw in (text, content_desc):
        if not raw:
            continue
        if _normalize_visible_label(raw) in _BOTTOM_TAB_ME_LABELS:
            return True
    return False


def label_matches_query(query: str, text: str, content_desc: str) -> bool:
    """True when element text/desc equals query, not a longer false-positive phrase."""
    q = query.strip()
    if not q:
        return False
    for raw in (text, content_desc):
        if not raw:
            continue
        candidate = _normalize_visible_label(raw)
        if candidate == q:
            return True
        if len(q) == 1 and q == "我" and candidate == "我":
            return True
        if q == "我的" and candidate == "我的":
            return True
        if candidate.startswith(q) and len(candidate) > len(q):
            if any(candidate.startswith(bad) or bad in candidate for bad in _ME_QUERY_FALSE_PREFIXES.get(q, ())):
                return False
            if q in {"我", "我的"}:
                return False
    return False


def score_element_for_query(
    query: str,
    el: UiElement,
    *,
    screen_width: int,
    screen_height: int,
    prefer_bottom_tab: bool = False,
) -> float:
    text, desc = visible_labels(el)
    if not label_matches_query(query, text, desc):
        return -1.0
    if not el.clickable or not el.displayed:
        return -1.0
    score = 100.0
    label_len = len(text or desc)
    score -= max(0, label_len - len(query)) * 3.0
    center = el.center
    if center and prefer_bottom_tab and center[1] >= screen_height * _BOTTOM_TAB_Y_MIN_RATIO:
        score += 25.0
    if center and prefer_bottom_tab and center[1] < screen_height * _BOTTOM_TAB_Y_MIN_RATIO:
        score -= 15.0
    if query in {"我", "我的"} and text and len(text) > len(query) + 1:
        score -= 50.0
    return score


def collect_clickable_candidates(
    page_source: str,
    query: str,
    *,
    screen_width: int = 1080,
    screen_height: int = 1920,
    prefer_bottom_tab: bool = False,
) -> list[tuple[float, UiElement]]:
    elements = parse_page_source(page_source)
    if not elements:
        return []
    width = screen_width
    height = screen_height
    for el in elements:
        if el.bounds:
            width = max(width, el.bounds[2])
            height = max(height, el.bounds[3])
    ranked: list[tuple[float, UiElement]] = []
    for el in elements:
        score = score_element_for_query(
            query,
            el,
            screen_width=width,
            screen_height=height,
            prefer_bottom_tab=prefer_bottom_tab,
        )
        if score >= 0:
            ranked.append((score, el))
    ranked.sort(key=lambda item: (-item[0], item[1].center[1] if item[1].center else 0))
    return ranked


def tap_best_element(driver: Any, ranked: list[tuple[float, UiElement]], steps: list[str], tag: str) -> bool:
    if not ranked:
        return False
    score, el = ranked[0]
    label = el.label or el.text or el.content_desc
    try:
        bounds = el.bounds
        if bounds:
            x = int((bounds[0] + bounds[2]) / 2)
            y = int((bounds[1] + bounds[3]) / 2)
            if hasattr(driver, "tap"):
                driver.tap([(x, y)], 200)
            else:
                driver.execute_script("mobile: clickGesture", {"x": x, "y": y})
            steps.append(f"{tag}[bounds]=ok({x},{y},{label})")
            logger.info("Tapped %s at (%s,%s) score=%.1f label=%s", tag, x, y, score, label)
            return True
        if hasattr(driver, "find_element"):
            from appium.webdriver.common.appiumby import AppiumBy  # type: ignore

            if el.resource_id:
                nodes = driver.find_elements(AppiumBy.ID, el.resource_id)
                if len(nodes) == 1:
                    nodes[0].click()
                    steps.append(f"{tag}[resource_id]=ok({label})")
                    logger.info("Tapped %s via resource-id score=%.1f label=%s", tag, score, label)
                    return True
                if len(nodes) > 1:
                    steps.append(f"{tag}[resource_id]=ambiguous({len(nodes)})")
                    logger.warning(
                        "Skipped ambiguous resource-id tap id=%s matches=%d",
                        el.resource_id,
                        len(nodes),
                    )
    except Exception as exc:
        steps.append(f"{tag}=error({exc})")
        logger.warning("tap_best_element failed: %s", exc)
    return False


def tap_label_via_hierarchy(
    driver: Any,
    label: str,
    steps: list[str],
    *,
    prefer_bottom_tab: bool = False,
    tag: str = "hierarchy_tap",
) -> bool:
    try:
        source = driver.page_source or ""
    except Exception:
        return False
    if not source:
        return False
    width, height = 1080, 1920
    try:
        size = driver.get_window_size()
        width = int(size.get("width", width))
        height = int(size.get("height", height))
    except Exception:
        pass
    ranked = collect_clickable_candidates(
        source,
        label,
        screen_width=width,
        screen_height=height,
        prefer_bottom_tab=prefer_bottom_tab,
    )
    return tap_best_element(driver, ranked, steps, f"{tag}[{label}]")


def scroll_to_label(driver: Any, label: str, steps: list[str], appium_by: Any) -> bool:
    """Scroll scrollable container until label may become visible."""
    if label not in _SCROLLABLE_TOKENS and len(label) > 12:
        return False
    safe = label.replace('"', '\\"')
    queries = [
        (
            appium_by.ANDROID_UIAUTOMATOR,
            f'new UiScrollable(new UiSelector().scrollable(true).instance(0))'
            f'.scrollIntoView(new UiSelector().text("{safe}"))',
        ),
        (
            appium_by.ANDROID_UIAUTOMATOR,
            f'new UiScrollable(new UiSelector().scrollable(true).instance(0))'
            f'.scrollIntoView(new UiSelector().textContains("{safe}"))',
        ),
    ]
    for by, query in queries:
        try:
            el = driver.find_element(by, query)
            if el:
                steps.append(f"scroll_into_view[{label}]=ok")
                logger.info("Scrolled into view for label=%s", label)
                return True
        except Exception:
            continue
    try:
        size = driver.get_window_size()
        w, h = int(size.get("width", 1080)), int(size.get("height", 1920))
        driver.execute_script(
            "mobile: scrollGesture",
            {
                "left": int(w * 0.1),
                "top": int(h * 0.35),
                "width": int(w * 0.8),
                "height": int(h * 0.45),
                "direction": "down",
                "percent": 0.75,
            },
        )
        steps.append(f"scroll_gesture_down[{label}]=ok")
        return True
    except Exception as exc:
        steps.append(f"scroll_gesture_down[{label}]=fail({exc})")
        return False


def _bounds_contain_point(bounds: tuple[int, int, int, int], x: float, y: float) -> bool:
    x1, y1, x2, y2 = bounds
    return x1 <= x <= x2 and y1 <= y <= y2


def bottom_tab_me_present(page_source: str, *, screen_height: int = 1920) -> bool:
    """True when a bottom navigation tab labeled 我/我的 exists (世纪佳缘、支付宝等)."""
    elements = parse_page_source(page_source or "")
    if not elements:
        return False
    height = screen_height
    for el in elements:
        if el.bounds:
            height = max(height, el.bounds[3])
    y_min = height * _BOTTOM_TAB_Y_MIN_RATIO
    for el in elements:
        text, desc = visible_labels(el)
        if _is_bottom_me_label(text, desc) and el.center and el.center[1] >= y_min:
            return True
    return False


def tap_bottom_tab_me(driver: Any, steps: list[str]) -> bool:
    """
    Tap bottom-nav 我 / 我的 tab (世纪佳缘 abt_bottom_tab_item, 支付宝 bottom bar, etc.).

    Prefer clickable ancestor that contains the tab label; avoid feed card avatars.
    """
    try:
        source = driver.page_source or ""
    except Exception:
        return False
    if not source:
        return False
    elements = parse_page_source(source)
    if not elements:
        return False
    width, height = 1080, 1920
    for el in elements:
        if el.bounds:
            width = max(width, el.bounds[2])
            height = max(height, el.bounds[3])
    y_min = height * _BOTTOM_TAB_Y_MIN_RATIO

    label_els: list[UiElement] = []
    for el in elements:
        text, desc = visible_labels(el)
        if not _is_bottom_me_label(text, desc):
            continue
        if not el.center or el.center[1] < y_min:
            continue
        label_els.append(el)
    if not label_els:
        for el in elements:
            rid = (el.resource_id or "").lower()
            if "abt_bottom_tab_text" not in rid:
                continue
            text, desc = visible_labels(el)
            if _is_bottom_me_label(text, desc) and el.center and el.center[1] >= y_min:
                label_els.append(el)
    if not label_els:
        tab_hosts = [
            el
            for el in elements
            if el.clickable
            and el.displayed
            and el.bounds
            and el.center
            and el.center[1] >= y_min
            and "abt_bottom_tab_item" in (el.resource_id or "")
        ]
        if tab_hosts:
            tab_hosts.sort(key=lambda e: e.bounds[2], reverse=True)
            host = tab_hosts[0]
            if tap_best_element(driver, [(100.0, host)], steps, "bottom_tab_me_rightmost"):
                time.sleep(0.7)
                return True
        return False

    label_els.sort(key=lambda e: (e.center[0] if e.center else 0), reverse=True)
    target = label_els[0]
    if not target.center:
        return False
    mx, my = target.center

    clickable_hosts: list[tuple[float, UiElement]] = []
    for el in elements:
        if not el.clickable or not el.displayed or not el.bounds or not el.center:
            continue
        if el.center[1] < y_min:
            continue
        if _bounds_contain_point(el.bounds, mx, my):
            area = (el.bounds[2] - el.bounds[0]) * (el.bounds[3] - el.bounds[1])
            clickable_hosts.append((area, el))

    if clickable_hosts:
        clickable_hosts.sort(key=lambda item: -item[0])
        _, host = clickable_hosts[0]
        if tap_best_element(driver, [(100.0, host)], steps, "bottom_tab_me"):
            time.sleep(0.7)
            return True

    if target.clickable and tap_best_element(driver, [(90.0, target)], steps, "bottom_tab_me_label"):
        time.sleep(0.7)
        return True

    try:
        x, y = int(mx), int(my)
        if hasattr(driver, "tap"):
            driver.tap([(x, y)], 200)
        else:
            driver.execute_script("mobile: clickGesture", {"x": x, "y": y})
        steps.append(f"bottom_tab_me_coord=ok({x},{y})")
        logger.info("Bottom tab 我 coordinate tap at (%s, %s)", x, y)
        time.sleep(0.7)
        return True
    except Exception as exc:
        steps.append(f"bottom_tab_me_coord=fail({exc})")
        return False


def _avatar_resource_blocked(resource_id: str) -> bool:
    rid = resource_id.lower()
    return any(block in rid for block in _AVATAR_RESOURCE_ID_BLOCKLIST)


def profile_page_visible(page_source: str) -> bool:
    """True when hierarchy looks like a profile / 我的 hub (incl. avatar-only entry apps)."""
    if not page_source:
        return False
    low = page_source.lower()
    if any(marker in low for marker in _PROFILE_PAGE_STRONG_MARKERS):
        return True
    if "iv_mine_header" in low and ("我的账户" in low or "我的认证" in low or "我的动态" in low):
        return True
    elements = parse_page_source(page_source)
    text_hits = 0
    for el in elements:
        blob = f"{el.text} {el.content_desc}"
        for marker in _PROFILE_PAGE_TEXT_MARKERS:
            if marker in blob:
                text_hits += 1
                break
    return text_hits >= 2


def profile_page_checks(page_source: str) -> dict[str, bool]:
    """Keyword-style checks for _step_keyword_matched after 我/我的 step."""
    low = (page_source or "").lower()
    checks: dict[str, bool] = {}
    for marker in _PROFILE_PAGE_TEXT_MARKERS:
        checks[marker] = marker in low
    for marker in ("我的", "我", "个人主页", "个人", "用户"):
        checks[marker] = marker in low
    checks["profile_resource_marker"] = any(
        rid in low for rid in _PROFILE_PAGE_RESOURCE_MARKERS
    )
    checks["profile_page"] = profile_page_visible(page_source)
    return checks


def tap_avatar_by_resource_id(driver: Any, steps: list[str]) -> bool:
    """Tap known avatar resource-ids (e.g. eastmoney riv_avatar on Mainpage)."""
    try:
        from appium.webdriver.common.appiumby import AppiumBy  # type: ignore
    except Exception:
        return False
    try:
        source = driver.page_source or ""
    except Exception:
        source = ""
    if not source:
        return False
    width, height = 1080, 1920
    try:
        size = driver.get_window_size()
        width = int(size.get("width", width))
        height = int(size.get("height", height))
    except Exception:
        pass
    y_max = height * _TOP_PROFILE_Y_MAX_RATIO
    x_max = width * 0.35
    for rid_suffix in _AVATAR_RESOURCE_IDS_PRIORITY:
        for el in parse_page_source(source):
            if not el.resource_id or rid_suffix not in el.resource_id:
                continue
            if _avatar_resource_blocked(el.resource_id):
                continue
            if not el.bounds:
                continue
            cx = (el.bounds[0] + el.bounds[2]) / 2
            cy = (el.bounds[1] + el.bounds[3]) / 2
            if cy > y_max or cx > x_max:
                continue
            full_rid = el.resource_id
            try:
                nodes = driver.find_elements(AppiumBy.ID, full_rid)
                if nodes:
                    nodes[0].click()
                    steps.append(f"avatar_resource_id[{rid_suffix}]=ok")
                    logger.info("Tapped avatar via resource-id %s", full_rid)
                    return True
            except Exception:
                if tap_best_element(driver, [(80.0, el)], steps, f"avatar_rid[{rid_suffix}]"):
                    return True
    return False


def tap_me_entry(
    driver: Any,
    steps: list[str],
    *,
    appium_by: Any | None = None,
) -> bool:
    """
    Open profile / 我的 hub.

    Order: already on profile -> bottom tab 我 (世纪佳缘等) -> text labels ->
    top-left app avatar only when no bottom 我 tab (东方财富等).
    """
    try:
        source = driver.page_source or ""
    except Exception:
        source = ""
    if profile_page_visible(source):
        steps.append("me_entry_already_on_profile_page")
        logger.info("Profile page already visible; skip entry tap")
        return True

    has_bottom_me = bottom_tab_me_present(source)
    if has_bottom_me:
        steps.append("me_entry_bottom_tab_detected")
        if tap_bottom_tab_me(driver, steps):
            try:
                if profile_page_visible(driver.page_source or ""):
                    return True
            except Exception:
                return True

    if appium_by is not None:
        for label in ("我的", "我"):
            if tap_label_via_hierarchy(
                driver,
                label,
                steps,
                prefer_bottom_tab=True,
                tag="me_entry_hierarchy",
            ):
                time.sleep(0.6)
                try:
                    if profile_page_visible(driver.page_source or ""):
                        return True
                except Exception:
                    return True
            try:
                from appium.webdriver.common.appiumby import AppiumBy  # type: ignore

                el = driver.find_element(
                    AppiumBy.ANDROID_UIAUTOMATOR,
                    f'new UiSelector().text("{label}")',
                )
                if el:
                    el.click()
                    steps.append(f"me_entry_label[{label}]=ok")
                    time.sleep(0.6)
                    if profile_page_visible(driver.page_source or ""):
                        return True
            except Exception:
                continue

    if not has_bottom_me:
        if tap_avatar_by_resource_id(driver, steps):
            time.sleep(0.8)
            try:
                if profile_page_visible(driver.page_source or ""):
                    return True
            except Exception:
                return True
        if tap_profile_avatar(driver, steps):
            time.sleep(0.8)
            try:
                if profile_page_visible(driver.page_source or ""):
                    return True
            except Exception:
                return True

    return False


def tap_profile_avatar(driver: Any, steps: list[str]) -> bool:
    """Tap top-left avatar / profile image when there is no 我的 tab label."""
    try:
        source = driver.page_source or ""
    except Exception:
        return False
    elements = parse_page_source(source)
    if not elements:
        return False
    width, height = 1080, 1920
    for el in elements:
        if el.bounds:
            width = max(width, el.bounds[2])
            height = max(height, el.bounds[3])
    candidates: list[tuple[float, UiElement]] = []
    for el in elements:
        if not el.clickable or not el.displayed or not el.center:
            continue
        cx, cy = el.center
        if cy > height * _TOP_PROFILE_Y_MAX_RATIO:
            continue
        score = 0.0
        rid = el.resource_id.lower()
        if _avatar_resource_blocked(rid):
            continue
        if any(h in rid for h in _AVATAR_RESOURCE_HINTS):
            score += 40.0
        if any(h in el.clazz for h in _IMAGE_CLASS_HINTS):
            score += 20.0
        if cx <= width * _LEFT_AVATAR_X_MAX_RATIO:
            score += 25.0
        elif cx <= width * 0.35:
            score += 10.0
        text, desc = visible_labels(el)
        if text or desc:
            score -= 30.0
        if score > 0:
            candidates.append((score, el))
    candidates.sort(key=lambda x: -x[0])
    if candidates and tap_best_element(driver, candidates, steps, "avatar_tap"):
        return True
    # Eastmoney Mainpage: riv_avatar center ~(49, 81) on 1080x1920
    try:
        x = int(width * 0.05)
        y = int(height * 0.05)
        if hasattr(driver, "tap"):
            driver.tap([(x, y)], 200)
        else:
            driver.execute_script("mobile: clickGesture", {"x": x, "y": y})
        steps.append(f"avatar_coord=ok({x},{y})")
        logger.info("Avatar coordinate tap at (%s, %s)", x, y)
        return True
    except Exception:
        steps.append("avatar_coord=miss")
        return False


_TOP_SETTINGS_Y_MAX_RATIO = 0.12
_TOP_SETTINGS_X_MIN_RATIO = 0.82
_SETTINGS_GEAR_MAX_ICON_PX = 80
_SETTINGS_GEAR_MIN_ICON_PX = 20
_SETTINGS_PAGE_STRONG_MARKERS: tuple[str, ...] = (
    'pane-title="设置"',
    "pane-title=\"设置\"",
    'content-desc="设置"',
)
_SETTINGS_PAGE_MENU_MARKERS: tuple[str, ...] = (
    "帐号与安全",
    "账号与安全",
    "关于小荷",
    "关于小荷健康",
    "隐私设置",
    "切换基本版",
)
# When content-desc / text is present, classify toolbar icons before position heuristics.
_SETTINGS_TOOLBAR_LABEL_SETTINGS: tuple[str, ...] = ("设置",)
_SETTINGS_TOOLBAR_LABEL_SCAN: tuple[str, ...] = (
    "扫一扫",
    "扫码",
    "扫描",
    "scan",
    "qr",
)
# Icons with these descs are never the settings gear (profile / nav labels).
_SETTINGS_GEAR_DESC_SKIP: tuple[str, ...] = (
    "我的",
    "首页",
    "消息",
    "医疗",
    "搜索",
    "返回",
    "分享",
    "更多",
    *_SETTINGS_TOOLBAR_LABEL_SCAN,
)


def settings_page_visible(page_source: str) -> bool:
    """True when hierarchy looks like 设置页 (小荷: pane-title / 帐号与安全 等)."""
    if not page_source:
        return False
    if any(marker in page_source for marker in _SETTINGS_PAGE_STRONG_MARKERS):
        if any(m in page_source for m in _SETTINGS_PAGE_MENU_MARKERS):
            return True
    elements = parse_page_source(page_source)
    header = False
    menu = False
    for el in elements:
        text, desc = visible_labels(el)
        blob = f"{text} {desc}"
        if desc == "设置" or text == "设置":
            header = True
        if any(m in blob for m in _SETTINGS_PAGE_MENU_MARKERS):
            menu = True
    return header and menu


def _toolbar_icon_role(text: str, content_desc: str) -> str:
    """
    Classify a top-right toolbar control when the app exposes text/desc.

    Returns: settings | scan | skip | unknown
    """
    for raw in (text, content_desc):
        if not raw:
            continue
        label = _normalize_visible_label(raw).lower()
        if any(s in label for s in _SETTINGS_TOOLBAR_LABEL_SETTINGS) or label == "设置":
            return "settings"
        if any(s in label for s in _SETTINGS_TOOLBAR_LABEL_SCAN):
            return "scan"
        if any(raw == skip or raw.startswith(skip) for skip in _SETTINGS_GEAR_DESC_SKIP):
            return "skip"
    return "unknown"


def _collect_settings_gear_candidates(
    elements: list[UiElement],
    *,
    screen_width: int,
    screen_height: int,
) -> list[UiElement]:
    """Small clickable ImageView in top-right toolbar (小荷 UserPage 齿轮 / 扫一扫)."""
    y_max = screen_height * _TOP_SETTINGS_Y_MAX_RATIO
    x_min = screen_width * _TOP_SETTINGS_X_MIN_RATIO
    found: list[UiElement] = []
    for el in elements:
        if not el.clickable or not el.displayed or not el.bounds or not el.center:
            continue
        cx, cy = el.center
        if cy > y_max or cx < x_min:
            continue
        if "Image" not in el.clazz and "Button" not in el.clazz:
            continue
        text, desc = visible_labels(el)
        role = _toolbar_icon_role(text, desc)
        if role == "skip" or role == "scan":
            continue
        if role != "settings" and text:
            continue
        if role == "unknown" and desc and any(
            desc == skip or desc.startswith(skip) for skip in _SETTINGS_GEAR_DESC_SKIP
        ):
            continue
        w = el.bounds[2] - el.bounds[0]
        h = el.bounds[3] - el.bounds[1]
        if w > _SETTINGS_GEAR_MAX_ICON_PX or h > _SETTINGS_GEAR_MAX_ICON_PX:
            continue
        if w < _SETTINGS_GEAR_MIN_ICON_PX or h < _SETTINGS_GEAR_MIN_ICON_PX:
            continue
        found.append(el)
    found.sort(key=lambda e: (e.center[0] if e.center else 0))
    return found


def _pick_settings_gear_target(candidates: list[UiElement]) -> UiElement | None:
    """
    Choose settings gear: labeled 设置 first, else rightmost icon (小荷: 右=设置, 左=扫一扫).
    """
    if not candidates:
        return None
    labeled: list[UiElement] = []
    for el in candidates:
        text, desc = visible_labels(el)
        if _toolbar_icon_role(text, desc) == "settings":
            labeled.append(el)
    if labeled:
        labeled.sort(key=lambda e: (e.center[0] if e.center else 0))
        return labeled[-1]
    return candidates[-1]


def tap_settings_gear_icon(driver: Any, steps: list[str]) -> bool:
    """
    Tap top-right settings gear when there is no 设置 text (小荷健康 UserPage).

    Uses content-desc/text when present; otherwise the rightmost top-right icon
    (left icon is 扫一扫 on 小荷).
    """
    try:
        source = driver.page_source or ""
    except Exception:
        return False
    if settings_page_visible(source):
        steps.append("settings_gear_already_on_page")
        logger.info("Settings page already visible; skip gear tap")
        return True

    elements = parse_page_source(source)
    if not elements:
        return False
    width, height = 1080, 1920
    for el in elements:
        if el.bounds:
            width = max(width, el.bounds[2])
            height = max(height, el.bounds[3])

    gears = _collect_settings_gear_candidates(
        elements, screen_width=width, screen_height=height
    )
    gear = _pick_settings_gear_target(gears)
    if gear is not None:
        text, desc = visible_labels(gear)
        role = _toolbar_icon_role(text, desc)
        pick = "label=设置" if role == "settings" else "rightmost_toolbar_icon"
        steps.append(f"settings_gear_pick={pick}")
        if tap_best_element(driver, [(100.0, gear)], steps, "settings_gear"):
            time.sleep(0.65)
            try:
                if settings_page_visible(driver.page_source or ""):
                    return True
            except Exception:
                return True

    # 小荷 UserPage: gear at [996,70][1045,119] when hierarchy has no desc
    x = int(width * 0.945)
    y = int(height * 0.049)
    try:
        if hasattr(driver, "tap"):
            driver.tap([(x, y)], 200)
        else:
            driver.execute_script("mobile: clickGesture", {"x": x, "y": y})
        steps.append(f"settings_gear_coord=ok({x},{y})")
        logger.info("Settings gear coordinate tap at (%s, %s)", x, y)
        time.sleep(0.65)
        try:
            return settings_page_visible(driver.page_source or "")
        except Exception:
            return True
    except Exception as exc:
        steps.append(f"settings_gear_coord=fail({exc})")
        return False


def extract_bracket_policy_tokens(hint: str) -> list[str]:
    """Parse 【我的】-【设置】 and 【我 → 点击头像 → 维护资料】 style policy paths."""
    parts = re.findall(r"【([^】]+)】", hint)
    tokens: list[str] = []
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        if re.search(r"[-－→>]", chunk):
            sub = re.split(r"[-－→>]+", chunk)
            for piece in sub:
                piece = piece.strip()
                if piece and len(piece) <= 16:
                    tokens.append(piece)
        elif len(chunk) <= 16:
            tokens.append(chunk)
    return tokens
