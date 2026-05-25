"""Appium execution layer for dynamic stage."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from privacy_audit.analyzers.dynamic.navigation_path_builder import (
    SIDEBAR_TAP_LABELS,
    SIDEBAR_TOKEN,
    build_navigation_path_evidence,
    hint_requires_sidebar,
    is_sidebar_token,
)
from privacy_audit.analyzers.dynamic.app_launch import (
    PACKAGES_SKIP_APPIUM_AUTO_LAUNCH,
    adb_foreground_package,
    adb_launch_package,
    ensure_target_app_foreground,
)
from privacy_audit.analyzers.dynamic.element_tap import (
    SCROLLABLE_TOKENS,
    bottom_tab_me_present,
    extract_bracket_policy_tokens,
    label_matches_query,
    profile_page_checks,
    profile_page_visible,
    scroll_to_label,
    settings_page_visible,
    tap_bottom_tab_me,
    tap_label_via_hierarchy,
    tap_me_entry,
    tap_profile_avatar,
    tap_settings_gear_icon,
)
from privacy_audit.analyzers.dynamic.sidebar_detector import (
    detect_drawer_open,
    detect_sidebar_trigger_on_screen,
    format_sidebar_state_step,
    get_sidebar_ui_state,
)
from privacy_audit.utils.runtime_preflight import check_appium_server

logger = logging.getLogger("privacy_audit.appium")

# Implicit find timeout (seconds). Keep low to avoid ~60–70s waits per selector on miss.
DEFAULT_IMPLICIT_WAIT_SEC = 2
DEFAULT_SELECTOR_TIMEOUT_MS = 3000
DEFAULT_IDLE_TIMEOUT_MS = 500
# Max wall-clock time per navigation token (e.g. 侧边栏 / 设置) before giving up.
DEFAULT_TAP_ATTEMPT_TIMEOUT_SEC = 15.0

_NAV_STOPWORDS = frozenset(
    {
        "左上角",
        "页面",
        "通过",
        "中的",
        "中",
        "可",
        "可以",
        "提示您注意",
        "Keep App",
        "App",
    }
)

# Profile / home tab labels (UI often shows 我的, not policy wording 我).
_ME_TAB_LABELS: tuple[str, ...] = (
    "我的",
    "我",
    "个人",
    "主页",
    "用户",
    "个人主页",
    "用户主页",
)
# Text taps for 我/我的 — avoid 个人/主页/用户 on main feed (false positives, wastes timeout).
_ME_ENTRY_TAP_LABELS: tuple[str, ...] = ("我的", "我")
_ME_TAB_TOKENS = frozenset({"我", "我的", *_ME_TAB_LABELS})

# Bottom tab bar: fraction of screen height from top where tab bar lives.
_BOTTOM_TAB_Y_MIN_RATIO = 0.82
_RIGHTMOST_TAB_X_RATIO = 0.9
_RIGHTMOST_TAB_Y_RATIO = 0.92

_SIDEBAR_CONTENT_DESCS: tuple[str, ...] = (
    "菜单",
    "更多",
    "打开导航抽屉",
    "导航抽屉",
    "drawer",
    "menu",
    "Menu",
    "Open navigation drawer",
)
_TOP_REGION_Y_MAX_RATIO = 0.2
_LEFT_REGION_X_MAX_RATIO = 0.3
_RIGHT_REGION_X_MIN_RATIO = 0.7
_TOP_LEFT_TAP_X_RATIO = 0.08
_TOP_LEFT_TAP_Y_RATIO = 0.08
_TOP_RIGHT_TAP_X_RATIO = 0.92
_TOP_RIGHT_TAP_Y_RATIO = 0.08
_RESET_BACK_PRESS_MAX = 8

# Extra labels to try for each navigation token (first match wins).
_TOKEN_TAP_ALIASES: dict[str, tuple[str, ...]] = {
    "我": _ME_ENTRY_TAP_LABELS,
    "我的": _ME_ENTRY_TAP_LABELS,
    "设置": ("设置",),
    "隐私设置": ("隐私设置", "隐私"),
    "隐私": ("隐私设置", "隐私"),
    "账号与安全": ("账号与安全", "账号与安全"),
    "个人资料": ("个人资料",),
    "管理设备隐私授权": ("管理设备隐私授权", "管理设备隐私授权", "隐私设置"),
    "第三方数据合作": ("第三方数据合作", "第三方数据合作", "隐私设置"),
    "消息与提醒": ("消息与提醒",),
}

_DEFAULT_KEEP_PATHS: list[list[str]] = [
    ["我", "设置", "隐私设置"],
    ["我", "设置", "账号与安全"],
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_nav_token(token: str) -> str:
    """Strip quotes/brackets so 「设置」 and 设置 both match UI text."""
    text = token.strip().strip(" \"'“”[]（）()")
    text = text.replace("「", "").replace("」", "")
    return text.strip()


def _extract_navigation_tokens(hint: str) -> list[str]:
    """
    Extract UI path tokens from a policy sentence.

    Prefer quoted segments 「...」 (Keep policy style). Fallback to arrow separators only,
    NOT bare hyphen — hyphens appear inside Chinese prose and would drop 「我」 when the
    prefix segment exceeds length limits.
    """
    quoted = [_normalize_nav_token(m) for m in re.findall(r"「([^」]+)」", hint)]
    quoted.extend(_normalize_nav_token(m) for m in extract_bracket_policy_tokens(hint))
    quoted = [t for t in quoted if t and t not in _NAV_STOPWORDS and len(t) <= 16]

    if quoted:
        return _merge_privacy_tokens(quoted)

    if "->" in hint or "→" in hint:
        parts = re.split(r"->|→", hint)
        tokens = [_normalize_nav_token(p) for p in parts]
        tokens = [t for t in tokens if t and t not in _NAV_STOPWORDS and len(t) <= 16]
        return _merge_privacy_tokens(tokens)

    return []


def _merge_privacy_tokens(tokens: list[str]) -> list[str]:
    """Merge 隐私/设置 variants into 隐私设置 for Keep policy wording."""
    merged: list[str] = []
    idx = 0
    while idx < len(tokens):
        if idx + 1 < len(tokens) and tokens[idx] == "隐私" and tokens[idx + 1] == "设置":
            merged.append("隐私设置")
            idx += 2
            continue
        if tokens[idx] == "隐私" and "隐私设置" not in tokens:
            merged.append("隐私设置")
        else:
            merged.append(tokens[idx])
        idx += 1
    return merged


def _extract_policy_path_hints(policy_text: str, max_count: int = 8) -> list[str]:
    hints: list[str] = []
    for line in policy_text.splitlines():
        text = line.strip().strip("；;。")
        if not text:
            continue
        if "->" in text or "→" in text:
            hints.append(text)
        elif "设置" in text and ("隐私" in text or "账号" in text or "个人资料" in text):
            hints.append(text)
        elif "「我」" in text and "设置" in text:
            hints.append(text)
        elif re.search(r"【[^】]+】", text) and (
            "设置" in text or "隐私" in text or "账号" in text or "我的" in text
        ):
            hints.append(text)
        elif hint_requires_sidebar(text) and ("设置" in text or "「" in text):
            hints.append(text)
        if len(hints) >= max_count:
            break
    return hints


def _ensure_me_first(path: list[str]) -> list[str]:
    """Keep paths should start from the 我 tab when targeting settings/privacy."""
    if not path:
        return path
    if path[0] in {"我", "我的"}:
        return path
    if any(k in tok for tok in path for k in ("设置", "隐私", "账号", "个人资料", "消息")):
        return ["我", *path]
    return path


def _candidate_paths_from_hints(policy_hints: list[str], max_paths: int = 2) -> list[list[str]]:
    keep_keywords = ("我", "我的", "设置", "隐私", "账号", "安全", "授权", "第三方", "管理", "个人资料")
    paths: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    for hint in policy_hints:
        if not any(k in hint for k in keep_keywords):
            continue
        tokens = _extract_navigation_tokens(hint)
        if len(tokens) < 2:
            continue
        path = _ensure_me_first(tokens[:5])
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
        if len(paths) >= max_paths:
            break

    if not paths:
        paths = [list(p) for p in _DEFAULT_KEEP_PATHS]

    # Prefer privacy-settings paths for consent-related navigation.
    paths.sort(
        key=lambda p: (
            0 if any("隐私" in t for t in p) else 1,
            len(p),
        )
    )
    return [_ensure_me_first(list(p)) for p in paths[:max_paths]]


def _resolve_navigation_paths(policy_hints: list[str], max_paths: int = 2) -> dict[str, Any]:
    """Build navigation_paths + navigation_path_records for dynamic evidence."""
    raw_paths = _candidate_paths_from_hints(policy_hints, max_paths=max_paths)
    return build_navigation_path_evidence(raw_paths, policy_hints, max_paths=max_paths)


def _to_bool_text(value: bool) -> str:
    return "true" if value else "false"


def _build_uiautomator2_options(
    *,
    device_name: str,
    app_package: str,
    app_activity: str,
    no_reset: bool,
    bind_package: bool,
) -> Any:
    from appium.options.android import UiAutomator2Options  # type: ignore

    options = UiAutomator2Options()
    options.set_capability("platformName", "Android")
    options.set_capability("automationName", "UiAutomator2")
    options.set_capability("deviceName", device_name)
    if bind_package and app_package:
        options.set_capability("appium:appPackage", app_package)
        options.set_capability("appium:appWaitActivity", "*")
        if app_package not in PACKAGES_SKIP_APPIUM_AUTO_LAUNCH:
            options.set_capability("appium:autoLaunch", True)
        else:
            options.set_capability("appium:autoLaunch", False)
    if app_activity:
        options.set_capability("appium:appActivity", app_activity)
    options.set_capability("appium:noReset", _to_bool_text(no_reset))
    options.set_capability("appium:dontStopAppOnReset", _to_bool_text(no_reset))
    options.set_capability("appium:newCommandTimeout", 90)
    options.set_capability("appium:adbExecTimeout", 60000)
    options.set_capability("appium:uiautomator2ServerLaunchTimeout", 60000)
    options.set_capability("appium:settings[waitForSelectorTimeout]", DEFAULT_SELECTOR_TIMEOUT_MS)
    options.set_capability("appium:settings[waitForIdleTimeout]", DEFAULT_IDLE_TIMEOUT_MS)
    return options


def _create_appium_driver(
    webdriver: Any,
    appium_server_url: str,
    *,
    device_name: str,
    app_package: str,
    app_activity: str,
    no_reset: bool,
    steps: list[str],
) -> Any:
    """
    Create Appium session bound to app_package.

    Never falls back to a package-less session (that would automate whatever is
    already foreground, e.g. Keep). Uses adb launch + retries instead.
    """
    if app_package:
        fg_before = adb_foreground_package()
        steps.append(f"adb_foreground_before={fg_before or 'unknown'}")
        if fg_before and fg_before != app_package:
            logger.info(
                "Foreground before session is %s; launching target %s via adb",
                fg_before,
                app_package,
            )
        adb_launch_package(app_package)

    last_exc: Exception | None = None
    for attempt in range(1, 4):
        bind_package = bool(app_package)
        options = _build_uiautomator2_options(
            device_name=device_name,
            app_package=app_package,
            app_activity=app_activity,
            no_reset=no_reset,
            bind_package=bind_package,
        )
        try:
            driver = webdriver.Remote(appium_server_url, options=options)
            steps.append(f"session_created_attempt{attempt}=ok")
            return driver
        except Exception as exc:
            last_exc = exc
            logger.warning("Appium session attempt %d failed: %s", attempt, exc)
            steps.append(f"session_created_attempt{attempt}=fail")
            if app_package:
                adb_launch_package(app_package)
                time.sleep(1.0)
            else:
                break
    raise RuntimeError(f"Appium session failed after retries: {last_exc}") from last_exc


def _seconds_remaining(deadline: float | None) -> float:
    if deadline is None:
        return float("inf")
    return max(0.0, deadline - time.monotonic())


def _timed_out(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _apply_driver_timeouts(
    driver: Any,
    implicit_wait_sec: float,
    deadline: float | None = None,
) -> None:
    """Short waits so each missed selector fails quickly; tightens when deadline is set."""
    remaining = _seconds_remaining(deadline)
    if deadline is None:
        imp = implicit_wait_sec
        sel_ms = DEFAULT_SELECTOR_TIMEOUT_MS
        idle_ms = DEFAULT_IDLE_TIMEOUT_MS
    else:
        imp = min(implicit_wait_sec, max(0.35, remaining / 5.0))
        sel_ms = int(min(DEFAULT_SELECTOR_TIMEOUT_MS, max(400, remaining * 1000 / 5.0)))
        idle_ms = min(DEFAULT_IDLE_TIMEOUT_MS, sel_ms)
    try:
        driver.implicitly_wait(imp)
    except Exception as exc:
        logger.debug("implicitly_wait not applied: %s", exc)
    try:
        driver.update_settings(
            {
                "waitForIdleTimeout": idle_ms,
                "waitForSelectorTimeout": sel_ms,
            }
        )
    except Exception as exc:
        logger.debug("update_settings not applied: %s", exc)


def _configure_driver_timeouts(driver: Any, implicit_wait_sec: float) -> None:
    """Session-level defaults (per-token navigation may tighten via deadline)."""
    _apply_driver_timeouts(driver, implicit_wait_sec, deadline=None)


def _capture_step_shot(
    driver: Any,
    screenshot_dir: Path,
    app_name: str,
    step_name: str,
    screenshots: list[str],
) -> None:
    safe_step = re.sub(r"[^a-zA-Z0-9_]+", "_", step_name).strip("_") or "step"
    shot = screenshot_dir / f"{app_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_step}.png"
    driver.save_screenshot(str(shot))
    screenshots.append(str(shot))


def _is_me_tab_token(token: str) -> bool:
    return _normalize_nav_token(token) in _ME_TAB_TOKENS


def _is_sidebar_token(token: str) -> bool:
    return is_sidebar_token(token)


def _labels_for_token(token: str) -> list[str]:
    normalized = _normalize_nav_token(token)
    if _is_me_tab_token(normalized):
        seed = _ME_ENTRY_TAP_LABELS
    else:
        aliases = _TOKEN_TAP_ALIASES.get(normalized, (normalized,))
        seed = (token, normalized, *aliases)
    labels: list[str] = []
    for label in seed:
        label = _normalize_nav_token(label)
        if label and label not in labels:
            labels.append(label)
    return labels


def _keyword_checks_for_step(token: str) -> list[str]:
    """Keywords for page_source probe after each tap."""
    if _is_sidebar_token(token):
        return ["更多", "菜单", "设置", "隐私", "账号", "个人资料", "消息与提醒", "账号与安全"]
    if _is_me_tab_token(token):
        return list(
            dict.fromkeys(
                [
                    _normalize_nav_token(token),
                    "我的",
                    "我",
                    *_ME_TAB_LABELS,
                    "隐私",
                    "设置",
                    "profile_page",
                    "profile_resource_marker",
                ]
            )
        )
    normalized = _normalize_nav_token(token)
    if normalized == "设置":
        return list(
            dict.fromkeys(
                [
                    token,
                    "设置",
                    "settings_page",
                    "帐号与安全",
                    "账号与安全",
                    "隐私",
                    "关于小荷",
                    "关于小荷健康",
                    "我的",
                ]
            )
        )
    return [_normalize_nav_token(token), "我的", "隐私", "设置", "我"]


def _step_keyword_matched(
    token: str,
    checks: dict[str, bool],
    sidebar_state: Any | None = None,
) -> bool:
    if _is_sidebar_token(token):
        if sidebar_state is not None and sidebar_state.sidebar_step_satisfied:
            return True
        return any(
            checks.get(label, False)
            for label in ("更多", "菜单", "设置", "隐私", "账号", "个人资料", "消息与提醒")
        )
    if _is_me_tab_token(token):
        if checks.get("profile_page") or checks.get("profile_resource_marker"):
            return True
        return any(checks.get(label, False) for label in _ME_TAB_LABELS)
    normalized = _normalize_nav_token(token)
    if normalized == "设置":
        return bool(
            checks.get("settings_page")
            or checks.get(normalized)
            or checks.get("帐号与安全")
            or checks.get("账号与安全")
            or checks.get("关于小荷")
            or checks.get("关于小荷健康")
        )
    return bool(checks.get(normalized, False) or checks.get("我的", False))


def _element_center_y(element: Any) -> float:
    try:
        loc = element.location
        size = element.size
        return float(loc.get("y", 0)) + float(size.get("height", 0)) / 2.0
    except Exception:
        return 0.0


def _element_center_x(element: Any) -> float:
    try:
        loc = element.location
        size = element.size
        return float(loc.get("x", 0)) + float(size.get("width", 0)) / 2.0
    except Exception:
        return 0.0


def _try_tap_single_label(
    driver: Any,
    appium_by: Any,
    label: str,
    *,
    deadline: float | None = None,
    implicit_wait_sec: float = DEFAULT_IMPLICIT_WAIT_SEC,
    prefer_bottom_tab: bool = False,
) -> bool:
    if _timed_out(deadline):
        return False
    _apply_driver_timeouts(driver, implicit_wait_sec, deadline=deadline)
    steps_probe: list[str] = []
    if tap_label_via_hierarchy(
        driver,
        label,
        steps_probe,
        prefer_bottom_tab=prefer_bottom_tab or label in {"我", "我的"},
        tag="precise_tap",
    ):
        for s in steps_probe:
            logger.info("%s", s)
        return True

    # Exact UiAutomator / XPath only (no textContains — avoids 我的 → 我的信用卡).
    selectors = [
        (appium_by.ANDROID_UIAUTOMATOR, f'new UiSelector().text("{label}")'),
        (appium_by.ANDROID_UIAUTOMATOR, f'new UiSelector().description("{label}")'),
        (
            appium_by.XPATH,
            (
                f'//*[@text="{label}" or @content-desc="{label}"]'
            ),
        ),
    ]
    for by, query in selectors:
        if _timed_out(deadline):
            logger.info("Tap label=%s stopped: attempt timeout", label)
            return False
        _apply_driver_timeouts(driver, implicit_wait_sec, deadline=deadline)
        try:
            elements = driver.find_elements(by, query)
            for element in elements:
                try:
                    text = str(getattr(element, "text", "") or "")
                    desc = str(getattr(element, "get_attribute", lambda _n: "")("content-desc") or "")
                except Exception:
                    text, desc = "", ""
                if label_matches_query(label, text, desc):
                    element.click()
                    logger.info("Tapped label=%s via exact %s text=%r", label, by, text or desc)
                    return True
        except Exception:
            continue
    return False


def _try_tap_text(
    driver: Any,
    appium_by: Any,
    text: str,
    *,
    deadline: float | None = None,
    implicit_wait_sec: float = DEFAULT_IMPLICIT_WAIT_SEC,
    prefer_bottom_tab: bool = False,
) -> bool:
    for label in _labels_for_token(text):
        if _timed_out(deadline):
            return False
        if _try_tap_single_label(
            driver,
            appium_by,
            label,
            deadline=deadline,
            implicit_wait_sec=implicit_wait_sec,
            prefer_bottom_tab=prefer_bottom_tab and not _is_me_tab_token(text),
        ):
            return True
    return False


def _window_size(driver: Any) -> tuple[int, int]:
    try:
        size = driver.get_window_size()
        return int(size.get("width", 1080)), int(size.get("height", 1920))
    except Exception:
        return 1080, 1920


def _try_tap_bottom_bar_clickable(
    driver: Any,
    appium_by: Any,
    steps: list[str],
    *,
    target_label: str = "我的",
) -> bool:
    """Tap bottom tab matching target_label exactly (not 我的信用卡)."""
    if tap_label_via_hierarchy(
        driver,
        target_label,
        steps,
        prefer_bottom_tab=True,
        tag="bottom_tab",
    ):
        return True
    width, height = _window_size(driver)
    y_min = height * _BOTTOM_TAB_Y_MIN_RATIO
    candidates: list[tuple[float, float, Any]] = []
    queries = [
        (appium_by.ANDROID_UIAUTOMATOR, "new UiSelector().clickable(true)"),
        (
            appium_by.XPATH,
            "//*[@clickable='true' and (@text!='' or @content-desc!='')]",
        ),
    ]
    for by, query in queries:
        try:
            for element in driver.find_elements(by, query):
                cy = _element_center_y(element)
                if cy < y_min:
                    continue
                try:
                    text = str(getattr(element, "text", "") or "")
                    desc = str(element.get_attribute("content-desc") or "")
                except Exception:
                    text, desc = "", ""
                if label_matches_query(target_label, text, desc):
                    cx = _element_center_x(element)
                    candidates.append((cx, cy, element))
        except Exception:
            continue
    if not candidates:
        steps.append(f"fallback_bottom_exact[{target_label}]=miss")
        return False
    candidates.sort(key=lambda item: item[0])
    rightmost_x, _, element = candidates[-1]
    try:
        element.click()
        steps.append(f"fallback_bottom_exact[{target_label}]=ok(x={int(rightmost_x)})")
        logger.info("Fallback: bottom tab label=%s at x=%.0f", target_label, rightmost_x)
        return True
    except Exception as exc:
        steps.append(f"fallback_bottom_exact[{target_label}]=error({exc})")
        return False


def _tap_coordinates(driver: Any, x: int, y: int, steps: list[str], step_tag: str) -> bool:
    try:
        if hasattr(driver, "tap"):
            driver.tap([(x, y)], 200)
        else:
            driver.execute_script("mobile: clickGesture", {"x": x, "y": y})
        steps.append(f"{step_tag}=ok({x},{y})")
        logger.info("Coordinate tap %s at (%s, %s)", step_tag, x, y)
        return True
    except Exception as exc:
        steps.append(f"{step_tag}=error({exc})")
        logger.warning("Coordinate tap %s failed: %s", step_tag, exc)
        return False


def _collect_corner_clickables(
    driver: Any,
    appium_by: Any,
    corner: str,
) -> list[tuple[float, float, Any]]:
    width, height = _window_size(driver)
    y_max = height * _TOP_REGION_Y_MAX_RATIO
    candidates: list[tuple[float, float, Any]] = []
    try:
        elements = driver.find_elements(
            appium_by.ANDROID_UIAUTOMATOR, "new UiSelector().clickable(true)"
        )
    except Exception:
        return candidates
    for element in elements:
        cx = _element_center_x(element)
        cy = _element_center_y(element)
        if cy > y_max:
            continue
        if corner == "top_left" and cx <= width * _LEFT_REGION_X_MAX_RATIO:
            candidates.append((cx, cy, element))
        elif corner == "top_right" and cx >= width * _RIGHT_REGION_X_MIN_RATIO:
            candidates.append((cx, cy, element))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if corner == "top_right":
        candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates


def _try_tap_corner_clickable(
    driver: Any,
    appium_by: Any,
    steps: list[str],
    corner: str,
) -> bool:
    candidates = _collect_corner_clickables(driver, appium_by, corner)
    if not candidates:
        steps.append(f"sidebar_{corner}_clickable=miss")
        return False
    element = candidates[0][2]
    try:
        element.click()
        steps.append(f"sidebar_{corner}_clickable=ok(x={int(candidates[0][0])})")
        logger.info("Sidebar: tapped %s clickable", corner)
        return True
    except Exception as exc:
        steps.append(f"sidebar_{corner}_clickable=error({exc})")
        return False


def _try_tap_sidebar_quick(
    driver: Any,
    appium_by: Any,
    steps: list[str],
    *,
    deadline: float | None = None,
    implicit_wait_sec: float = DEFAULT_IMPLICIT_WAIT_SEC,
) -> bool:
    """Fast path when XML shows no drawer trigger: only try visible menu labels."""
    for label in SIDEBAR_TAP_LABELS:
        if _timed_out(deadline):
            steps.append("sidebar_quick=timeout")
            return False
        if _try_tap_single_label(
            driver,
            appium_by,
            label,
            deadline=deadline,
            implicit_wait_sec=implicit_wait_sec,
        ):
            steps.append(f"sidebar_quick_text[{label}]=ok")
            logger.info("Sidebar quick: tapped label %s", label)
            return True
    steps.append("sidebar_quick=miss")
    return False


def _try_tap_sidebar(
    driver: Any,
    appium_by: Any,
    steps: list[str],
    prefer_corner: str = "top_left",
    *,
    deadline: float | None = None,
    implicit_wait_sec: float = DEFAULT_IMPLICIT_WAIT_SEC,
    quick_only: bool = False,
) -> bool:
    """Open drawer / menu (often labeled 更多 on screen)."""
    if _timed_out(deadline):
        steps.append("sidebar=timeout")
        return False
    if quick_only:
        return _try_tap_sidebar_quick(
            driver,
            appium_by,
            steps,
            deadline=deadline,
            implicit_wait_sec=implicit_wait_sec,
        )

    _apply_driver_timeouts(driver, implicit_wait_sec, deadline=deadline)
    keep_trigger_ids = (
        "com.gotokeep.keep:id/viewToolbarSidebar",
        "viewToolbarSidebar",
    )
    for res_id in keep_trigger_ids:
        if _timed_out(deadline):
            steps.append("sidebar=timeout")
            return False
        try:
            query = f'new UiSelector().resourceId("{res_id}")'
            elements = driver.find_elements(appium_by.ANDROID_UIAUTOMATOR, query)
            if elements:
                elements[0].click()
                steps.append(f"sidebar_resource_id[{res_id}]=ok")
                logger.info("Sidebar: tapped resource-id %s", res_id)
                return True
        except Exception:
            continue
    for label in SIDEBAR_TAP_LABELS:
        if _timed_out(deadline):
            steps.append("sidebar=timeout")
            return False
        if _try_tap_single_label(
            driver,
            appium_by,
            label,
            deadline=deadline,
            implicit_wait_sec=implicit_wait_sec,
        ):
            steps.append(f"sidebar_text[{label}]=ok")
            logger.info("Sidebar: tapped visible label %s", label)
            return True
    if _seconds_remaining(deadline) < 4.0:
        steps.append("sidebar=skip_desc_corners_low_budget")
        return False
    for desc in _SIDEBAR_CONTENT_DESCS:
        if _timed_out(deadline):
            steps.append("sidebar=timeout")
            return False
        _apply_driver_timeouts(driver, implicit_wait_sec, deadline=deadline)
        try:
            query = f'new UiSelector().descriptionContains("{desc}")'
            elements = driver.find_elements(appium_by.ANDROID_UIAUTOMATOR, query)
            if elements:
                elements[0].click()
                steps.append(f"sidebar_desc[{desc}]=ok")
                logger.info("Sidebar: tapped content-desc containing %s", desc)
                return True
        except Exception:
            continue
    corners = [prefer_corner, "top_left", "top_right"]
    seen_corners: set[str] = set()
    for corner in corners:
        if _timed_out(deadline):
            steps.append("sidebar=timeout")
            return False
        if corner in seen_corners:
            continue
        seen_corners.add(corner)
        if _try_tap_corner_clickable(driver, appium_by, steps, corner):
            return True
    if _timed_out(deadline):
        steps.append("sidebar=timeout")
        return False
    width, height = _window_size(driver)
    if prefer_corner == "top_right":
        x = int(width * _TOP_RIGHT_TAP_X_RATIO)
    else:
        x = int(width * _TOP_LEFT_TAP_X_RATIO)
    y = int(height * _TOP_LEFT_TAP_Y_RATIO)
    tag = "sidebar_coord_top_right" if prefer_corner == "top_right" else "sidebar_coord_top_left"
    return _tap_coordinates(driver, x, y, steps, tag)


def _navigate_back_to_home(driver: Any, steps: list[str], presses: int = _RESET_BACK_PRESS_MAX) -> None:
    for _ in range(presses):
        try:
            driver.press_keycode(4)
        except Exception:
            try:
                driver.back()
            except Exception:
                break
        time.sleep(0.35)
    steps.append(f"reset_back_presses={presses}")


def _reset_app_for_path(driver: Any, app_package: str, steps: list[str], path_idx: int) -> None:
    """Fresh start for each navigation path (avoid continuing from prior path's screen)."""
    steps.append(f"path_{path_idx}_reset_start")
    if app_package:
        try:
            driver.terminate_app(app_package)
            time.sleep(0.6)
        except Exception as exc:
            steps.append(f"path_{path_idx}_reset_terminate_failed={exc}")
            logger.warning("terminate failed for path %d: %s", path_idx, exc)
        if ensure_target_app_foreground(app_package, steps=steps, driver=driver):
            steps.append(f"path_{path_idx}_reset=terminate_activate")
            logger.info("Reset path %d: terminate + foreground %s", path_idx, app_package)
            return
    _navigate_back_to_home(driver, steps)
    if app_package:
        ensure_target_app_foreground(app_package, steps=steps, driver=driver)
    steps.append(f"path_{path_idx}_reset=back_fallback")


def _try_tap_coordinate_rightmost_tab(driver: Any, steps: list[str]) -> bool:
    """Coordinate tap on the rightmost bottom tab (icon-only bars)."""
    width, height = _window_size(driver)
    x = int(width * _RIGHTMOST_TAB_X_RATIO)
    y = int(height * _RIGHTMOST_TAB_Y_RATIO)
    return _tap_coordinates(driver, x, y, steps, "fallback_coord_tab")


def _try_tap_me_tab(
    driver: Any,
    appium_by: Any,
    steps: list[str],
    *,
    deadline: float | None = None,
    implicit_wait_sec: float = DEFAULT_IMPLICIT_WAIT_SEC,
) -> bool:
    if _timed_out(deadline):
        return False
    if tap_me_entry(driver, steps, appium_by=appium_by):
        steps.append("fallback_me_entry=ok")
        logger.info("Fallback: opened profile via avatar/me_entry")
        return True
    if _timed_out(deadline):
        return False
    for label in _ME_ENTRY_TAP_LABELS:
        if _try_tap_single_label(
            driver,
            appium_by,
            label,
            deadline=deadline,
            implicit_wait_sec=implicit_wait_sec,
            prefer_bottom_tab=False,
        ):
            steps.append(f"fallback_tap[{label}]=ok")
            logger.info("Fallback: opened profile tab via label=%s", label)
            return True
    if _timed_out(deadline):
        return False
    if _try_tap_bottom_bar_clickable(driver, appium_by, steps, target_label="我的"):
        return True
    if _timed_out(deadline):
        return False
    if _try_tap_coordinate_rightmost_tab(driver, steps):
        return True
    steps.append("fallback_me_tab=all_miss")
    logger.warning("Fallback: could not open profile (avatar + labels + bottom bar)")
    return False


def _navigate_tokens(
    driver: Any,
    appium_by: Any,
    path_idx: int,
    tokens: list[str],
    screenshot_dir: Path,
    app_name: str,
    screenshots: list[str],
    steps: list[str],
    keyword_checks: list[dict[str, Any]],
    path_record: dict[str, Any] | None = None,
    tap_attempt_timeout_sec: float = DEFAULT_TAP_ATTEMPT_TIMEOUT_SEC,
    implicit_wait_sec: float = DEFAULT_IMPLICIT_WAIT_SEC,
) -> None:
    exec_tokens = [t for t in tokens if not _is_sidebar_token(t)]
    steps.append(f"navigate_path_{path_idx}={' > '.join(exec_tokens)}")
    if len(exec_tokens) != len(tokens):
        steps.append("path_skip_sidebar_tokens=legacy_path_data")
    me_fallback_used = False
    prefer_corner = "top_right"
    if path_record:
        marker = str(
            path_record.get("policy_corner_marker", "")
            or path_record.get("sidebar_reason", "")
        )
        if "右上角" in marker:
            prefer_corner = "top_right"
        elif "左上角" in marker:
            prefer_corner = "top_left"

    for token in exec_tokens:
        norm = _normalize_nav_token(token)
        token_deadline = time.monotonic() + tap_attempt_timeout_sec
        logger.info(
            "Tap attempt path=%d token=%s timeout=%ss",
            path_idx,
            token,
            tap_attempt_timeout_sec,
        )
        _apply_driver_timeouts(driver, implicit_wait_sec, deadline=token_deadline)
        sidebar_state_after: Any | None = None

        if _is_me_tab_token(token):
            try:
                if profile_page_visible(driver.page_source or ""):
                    steps.append(f"tap[{token}]=already_on_profile_page")
                    tapped = True
                else:
                    tapped = tap_me_entry(driver, steps, appium_by=appium_by)
                    if tapped:
                        steps.append(f"tap[{token}]=me_entry_avatar")
            except Exception:
                tapped = False
        else:
            tapped = False

        if not tapped:
            tapped = _try_tap_text(
                driver,
                appium_by,
                token,
                deadline=token_deadline,
                implicit_wait_sec=implicit_wait_sec,
            )

        if not tapped and norm == "设置":
            steps.append(f"tap[{token}]=miss_try_settings_gear")
            logger.info("Tap miss on 设置; trying top-right gear icon (小荷等)")
            if tap_settings_gear_icon(driver, steps):
                tapped = True

        if not tapped and _is_me_tab_token(token):
            steps.append(f"tap[{token}]=miss_try_me_tab_fallback")
            logger.info("Profile tab miss on %s; running me-tab fallback chain", token)
            tapped = _try_tap_me_tab(
                driver,
                appium_by,
                steps,
                deadline=token_deadline,
                implicit_wait_sec=implicit_wait_sec,
            )
        elif (
            not tapped
            and not me_fallback_used
            and not _is_me_tab_token(token)
            and not _is_sidebar_token(token)
        ):
            me_fallback_used = True
            steps.append(f"tap[{token}]=miss_try_me_first")
            logger.info("Tap miss on %s; trying fallback profile tab", token)
            if _try_tap_me_tab(
                driver,
                appium_by,
                steps,
                deadline=token_deadline,
                implicit_wait_sec=implicit_wait_sec,
            ):
                time.sleep(0.5)
                tapped = _try_tap_text(
                    driver,
                    appium_by,
                    token,
                    deadline=token_deadline,
                    implicit_wait_sec=implicit_wait_sec,
                )

        timed_out = _timed_out(token_deadline)
        if not tapped and not timed_out:
            if norm in SCROLLABLE_TOKENS or any(k in norm for k in ("设置", "隐私", "账号")):
                steps.append(f"tap[{token}]=miss_try_scroll")
                logger.info("Step miss on %s; scrolling to find label", token)
                if scroll_to_label(driver, norm, steps, appium_by):
                    time.sleep(0.55)
                    tapped = _try_tap_text(
                        driver,
                        appium_by,
                        token,
                        deadline=token_deadline,
                        implicit_wait_sec=implicit_wait_sec,
                    )

        if not tapped and not timed_out:
            steps.append(f"tap[{token}]=miss_try_sidebar_fallback")
            state_before = get_sidebar_ui_state(driver)
            steps.append(f"sidebar_xml_before:{format_sidebar_state_step(state_before)}")
            if not state_before.trigger_present and not state_before.drawer_open:
                steps.append("sidebar_skip=no_drawer_trigger_in_xml")
                logger.info(
                    "Skip sidebar fallback for %s: no drawer trigger in hierarchy",
                    token,
                )
            else:
                logger.info(
                    "Step miss on %s; trying sidebar/drawer fallback (corner=%s)",
                    token,
                    prefer_corner,
                )
                quick_only = False
                sidebar_opened = state_before.sidebar_step_satisfied
                if not sidebar_opened:
                    sidebar_opened = _try_tap_sidebar(
                        driver,
                        appium_by,
                        steps,
                        prefer_corner=prefer_corner,
                        deadline=token_deadline,
                        implicit_wait_sec=implicit_wait_sec,
                        quick_only=quick_only,
                    )
                    if not _timed_out(token_deadline):
                        time.sleep(0.4)
                        sidebar_state_after = get_sidebar_ui_state(driver)
                        steps.append(
                            f"sidebar_xml_after:{format_sidebar_state_step(sidebar_state_after)}"
                        )
                        sidebar_opened = sidebar_opened or sidebar_state_after.sidebar_step_satisfied
                if sidebar_opened and not _timed_out(token_deadline):
                    if norm in SCROLLABLE_TOKENS or any(k in norm for k in ("设置", "隐私", "账号")):
                        scroll_to_label(driver, norm, steps, appium_by)
                    tapped = _try_tap_text(
                        driver,
                        appium_by,
                        token,
                        deadline=token_deadline,
                        implicit_wait_sec=implicit_wait_sec,
                    )
                if not tapped:
                    steps.append(f"sidebar_fallback_after_{token}=failed")

        if timed_out and not tapped:
            steps.append(f"tap[{token}]=timeout_{int(tap_attempt_timeout_sec)}s")
        else:
            steps.append(f"tap[{token}]={'ok' if tapped else 'miss'}")
        logger.info("Tap result path=%d token=%s ok=%s", path_idx, token, tapped)
        time.sleep(0.8)

        check_keywords = _keyword_checks_for_step(token)
        checks = _check_keywords(driver, check_keywords)
        if sidebar_state_after is not None:
            checks["xml_drawer_open"] = sidebar_state_after.drawer_open
            checks["xml_trigger_present"] = sidebar_state_after.trigger_present
        keyword_checks.append(
            {
                "path_index": path_idx,
                "step_token": token,
                "matched": _step_keyword_matched(token, checks, sidebar_state_after),
                "checks": checks,
            }
        )
        _capture_step_shot(driver, screenshot_dir, app_name, f"path{path_idx}_{token}", screenshots)

        if not tapped:
            steps.append(f"nav_path_{path_idx}_stopped_at={token}")
            break


def _check_keywords(driver: Any, keywords: list[str]) -> dict[str, bool]:
    try:
        source = driver.page_source or ""
    except Exception:
        source = ""
    low_source = source.lower()
    result: dict[str, bool] = {}
    for kw in keywords:
        key = _normalize_nav_token(kw).lower()
        if key == "profile_page":
            result[kw] = profile_page_visible(source)
        elif key == "settings_page":
            result[kw] = settings_page_visible(source)
        elif key == "profile_resource_marker":
            from privacy_audit.analyzers.dynamic.element_tap import (
                _PROFILE_PAGE_STRONG_MARKERS,
            )

            result[kw] = any(rid in low_source for rid in _PROFILE_PAGE_STRONG_MARKERS)
        else:
            result[kw] = key in low_source if low_source else False
    return result


def run_ui_flow(
    app_name: str,
    policy_text: str = "",
    appium_server_url: str = "http://127.0.0.1:4723",
    device_name: str = "Android",
    app_package: str = "",
    app_activity: str = "",
    screenshot_dir: Path | None = None,
    dry_run: bool = True,
    no_reset: bool = True,
    connect_timeout_sec: float = 5.0,
    fail_fast_without_appium: bool = True,
    implicit_wait_sec: float = DEFAULT_IMPLICIT_WAIT_SEC,
    tap_attempt_timeout_sec: float = DEFAULT_TAP_ATTEMPT_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Run Appium UI flow and return navigation evidence."""
    screenshot_dir = screenshot_dir or Path("output/stage3_dynamic/screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    policy_hints = _extract_policy_path_hints(policy_text)
    path_bundle = _resolve_navigation_paths(policy_hints)
    nav_paths = path_bundle["navigation_paths"]
    navigation_path_records = path_bundle["navigation_path_records"]
    logger.info(
        "Appium UI flow start app=%s package=%s dry_run=%s server=%s "
        "implicit_wait=%ss tap_attempt_timeout=%ss",
        app_name,
        app_package or "(none)",
        dry_run,
        appium_server_url,
        implicit_wait_sec,
        tap_attempt_timeout_sec,
    )
    logger.info("Policy path hints (%d): %s", len(policy_hints), policy_hints[:3])
    logger.info("Resolved navigation paths: %s", nav_paths)

    if dry_run:
        simulated_shot = screenshot_dir / f"{app_name}_dryrun_screen.png"
        steps = [
            f"[dry_run] launch app {app_name}",
            "[dry_run] navigate to candidate privacy settings page",
            f"[dry_run] paths={nav_paths}",
            "[dry_run] capture page text and path hint evidence",
        ]
        return {
            "runner": "appium",
            "mode": "dry_run",
            "timestamp_utc": _utc_now_iso(),
            "steps": steps,
            "policy_path_hints": policy_hints,
            "navigation_paths": nav_paths,
            "navigation_path_records": navigation_path_records,
            "screenshots": [str(simulated_shot)],
            "keyword_checks": [],
            "error": "",
        }

    appium_check = check_appium_server(appium_server_url, timeout_sec=connect_timeout_sec)
    if not appium_check.get("ok"):
        msg = str(appium_check.get("error", "Appium unreachable"))
        logger.error("Appium preflight failed: %s", msg)
        if fail_fast_without_appium:
            return {
                "runner": "appium",
                "mode": "real",
                "timestamp_utc": _utc_now_iso(),
                "steps": ["preflight_appium_failed"],
                "policy_path_hints": policy_hints,
                "navigation_paths": nav_paths,
                "navigation_path_records": navigation_path_records,
                "screenshots": [],
                "keyword_checks": [],
                "error": f"Appium preflight failed: {msg}",
            }

    try:
        logger.info("Loading Appium Python client...")
        from appium import webdriver  # type: ignore
        from appium.options.android import UiAutomator2Options  # type: ignore
        from appium.webdriver.common.appiumby import AppiumBy  # type: ignore
    except Exception as exc:
        logger.error("Appium dependencies unavailable: %s", exc)
        return {
            "runner": "appium",
            "mode": "real",
            "timestamp_utc": _utc_now_iso(),
            "steps": [],
            "policy_path_hints": policy_hints,
            "navigation_paths": nav_paths,
            "navigation_path_records": navigation_path_records,
            "screenshots": [],
            "keyword_checks": [],
            "error": f"Appium dependencies unavailable: {exc}",
        }

    driver = None
    steps: list[str] = []
    screenshots: list[str] = []
    keyword_checks: list[dict[str, Any]] = []
    error = ""
    try:
        logger.info(
            "Creating Appium session (may take 30-90s on first connect)... url=%s package=%s",
            appium_server_url,
            app_package,
        )
        t0 = time.perf_counter()
        driver = _create_appium_driver(
            webdriver,
            appium_server_url,
            device_name=device_name,
            app_package=app_package,
            app_activity=app_activity,
            no_reset=no_reset,
            steps=steps,
        )
        logger.info("Appium session created in %.1fs", time.perf_counter() - t0)
        _configure_driver_timeouts(driver, implicit_wait_sec)
        steps.append("session_created")
        if app_package:
            if not ensure_target_app_foreground(app_package, steps=steps, driver=driver):
                fg = adb_foreground_package()
                error = (
                    f"Could not bring {app_package} to foreground "
                    f"(adb sees {fg or 'unknown'}). Close other apps (e.g. Keep) and retry."
                )
                logger.error(error)
            else:
                current_pkg = getattr(driver, "current_package", "") or ""
                if current_pkg:
                    steps.append(f"foreground_package={current_pkg}")
                    if current_pkg != app_package:
                        error = (
                            f"Expected foreground package {app_package}, but got {current_pkg}"
                        )
        _capture_step_shot(driver, screenshot_dir, app_name, "launch", screenshots)

        if error:
            steps.append("navigate_skipped=wrong_foreground_package")
            logger.warning("Skipping UI navigation: %s", error)
        else:
            path_records_by_idx = {
                int(rec.get("path_index", 0)): rec for rec in navigation_path_records
            }
            for path_idx, tokens in enumerate(nav_paths, start=1):
                if path_idx > 1 and app_package:
                    _reset_app_for_path(driver, app_package, steps, path_idx)
                _navigate_tokens(
                    driver,
                    AppiumBy,
                    path_idx,
                    tokens,
                    screenshot_dir,
                    app_name,
                    screenshots,
                    steps,
                    keyword_checks,
                    path_record=path_records_by_idx.get(path_idx),
                    tap_attempt_timeout_sec=tap_attempt_timeout_sec,
                    implicit_wait_sec=implicit_wait_sec,
                )
            steps.append("screenshot_captured")
    except Exception as exc:  # pragma: no cover - runtime dependent
        error = f"Appium runtime failed: {exc}"
        logger.exception("Appium runtime failed")
    finally:
        if driver is not None:
            logger.info("Closing Appium session")
            try:
                driver.quit()
            except Exception:
                pass

    if error:
        logger.warning("Appium UI flow finished with error: %s", error)
    else:
        logger.info(
            "Appium UI flow finished steps=%d screenshots=%d",
            len(steps),
            len(screenshots),
        )

    return {
        "runner": "appium",
        "mode": "real",
        "timestamp_utc": _utc_now_iso(),
        "steps": steps,
        "policy_path_hints": policy_hints,
        "navigation_paths": nav_paths,
        "navigation_path_records": navigation_path_records,
        "screenshots": screenshots,
        "keyword_checks": keyword_checks,
        "error": error,
    }
