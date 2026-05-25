"""UI automation helpers for Appium — Chinese text element finding, scrolling, waiting."""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from selenium.webdriver.common.by import By

from .config import ELEMENT_WAIT_SEC

logger = logging.getLogger(__name__)


class UIAutomator:
    """Wraps Appium WebDriver for Chinese-language Android UI automation."""

    def __init__(self, driver):
        self.driver = driver
        self.wait_sec = ELEMENT_WAIT_SEC

    # ── Element finders ───────────────────────────────────────────────────

    def find_by_text(self, text: str, exact: bool = False) -> Optional[object]:
        """Find a UI element by its text content (Chinese-friendly)."""
        try:
            # Prefer UiAutomator textContains for partial matching
            escaped = text.replace('"', '\\"')
            if exact:
                selector = f'new UiSelector().text("{escaped}")'
            else:
                selector = f'new UiSelector().textContains("{escaped}")'
            el = self.driver.find_element(By.ANDROID_UIAUTOMATOR,selector)
            return el
        except Exception:
            return None

    def find_by_description(self, desc: str, exact: bool = False) -> Optional[object]:
        """Find element by content-description (accessibility label)."""
        try:
            escaped = desc.replace('"', '\\"')
            if exact:
                selector = f'new UiSelector().description("{escaped}")'
            else:
                selector = f'new UiSelector().descriptionContains("{escaped}")'
            return self.driver.find_element(By.ANDROID_UIAUTOMATOR,selector)
        except Exception:
            return None

    def find_by_id(self, resource_id: str) -> Optional[object]:
        """Find element by resource-id (Selenium 4 API)."""
        try:
            from selenium.webdriver.common.by import By
            return self.driver.find_element(By.ID, resource_id)
        except Exception:
            return None

    def find_any_by_ids(self, resource_ids: list[str]) -> Optional[object]:
        """Try multiple resource-ids and return the first matched element."""
        for rid in resource_ids:
            rid = (rid or "").strip()
            if not rid:
                continue
            el = self.find_by_id(rid)
            if el is not None:
                return el
        return None

    def find_by_class(self, class_name: str) -> Optional[object]:
        """Find element by class name (Selenium 4 API)."""
        try:
            from selenium.webdriver.common.by import By
            return self.driver.find_element(By.CLASS_NAME, class_name)
        except Exception:
            return None

    def find_by_xpath(self, xpath: str) -> Optional[object]:
        """Find element by XPath (Selenium 4 API)."""
        try:
            from selenium.webdriver.common.by import By
            return self.driver.find_element(By.XPATH, xpath)
        except Exception:
            return None

    def find_clickable_parent(self, element) -> Optional[object]:
        """Given a non-clickable element, find the nearest clickable ancestor."""
        try:
            # Try up to 3 levels of ancestry
            for _ in range(3):
                element = element.find_element(By.XPATH, "..")
                clickable = element.get_attribute("clickable")
                if clickable == "true":
                    return element
            return None
        except Exception:
            return None

    # ── Multi-match variant ───────────────────────────────────────────────

    def find_any(self, text_list: list[str], exact: bool = False) -> Optional[object]:
        """Try to find any element matching one of the given texts."""
        for text in text_list:
            el = self.find_by_text(text, exact=exact)
            if el is not None:
                return el
        return None

    def find_any_description(self, desc_list: list[str], exact: bool = False) -> Optional[object]:
        """Try to find any element by accessibility description."""
        for desc in desc_list:
            desc = (desc or "").strip()
            if not desc:
                continue
            el = self.find_by_description(desc, exact=exact)
            if el is not None:
                return el
        return None

    # ── Scroll helpers ────────────────────────────────────────────────────

    def scroll_to_text(self, text: str, max_scrolls: int = 5) -> Optional[object]:
        """Scroll through the screen looking for an element with the given text."""
        for _ in range(max_scrolls):
            el = self.find_by_text(text)
            if el is not None:
                return el
            self.swipe_up()
            time.sleep(0.3)
        return None

    def scroll_to_text_uiautomator(self, text: str) -> Optional[object]:
        """Use UiScrollable to scroll to text. More reliable than manual scroll."""
        try:
            escaped = text.replace('"', '\\"')
            selector = (
                f'new UiScrollable(new UiSelector().scrollable(true).instance(0))'
                f'.scrollTextIntoView("{escaped}")'
            )
            return self.driver.find_element(By.ANDROID_UIAUTOMATOR,selector)
        except Exception:
            return None

    # ── Gesture helpers ───────────────────────────────────────────────────

    def swipe_up(self) -> None:
        """Swipe from bottom to top (scroll down)."""
        try:
            size = self.driver.get_window_size()
            start_x = size["width"] // 2
            start_y = int(size["height"] * 0.7)
            end_y = int(size["height"] * 0.3)
            self.driver.swipe(start_x, start_y, start_x, end_y, 500)
        except Exception as e:
            logger.debug("Swipe up failed: %s", e)

    def swipe_down(self) -> None:
        """Swipe from top to bottom (scroll up)."""
        try:
            size = self.driver.get_window_size()
            start_x = size["width"] // 2
            start_y = int(size["height"] * 0.3)
            end_y = int(size["height"] * 0.7)
            self.driver.swipe(start_x, start_y, start_x, end_y, 500)
        except Exception as e:
            logger.debug("Swipe down failed: %s", e)

    def tap(self, x: int, y: int) -> None:
        """Tap at the specified screen coordinates."""
        try:
            self.driver.tap([(x, y)])
        except Exception as e:
            logger.debug("Tap failed at (%d,%d): %s", x, y, e)

    def tap_bottom_tab(self, tab_index: int, total_tabs: int = 5) -> bool:
        """Tap one of bottom tab slots by index as icon-only fallback."""
        if total_tabs <= 0:
            return False
        try:
            idx = max(0, min(tab_index, total_tabs - 1))
            size = self.driver.get_window_size()
            y = int(size["height"] * 0.94)
            step = size["width"] / total_tabs
            x = int(step * idx + step / 2)
            self.tap(x, y)
            time.sleep(0.4)
            return True
        except Exception as e:
            logger.debug("tap_bottom_tab failed: %s", e)
            return False

    def back(self) -> None:
        """Press the Android back button."""
        try:
            self.driver.back()
        except Exception as e:
            logger.debug("Back failed: %s", e)

    # ── Wait helpers ──────────────────────────────────────────────────────

    def wait_and_find_by_text(self, text: str, timeout: int = None) -> Optional[object]:
        """Poll until an element with the given text appears, or timeout."""
        timeout = timeout or self.wait_sec
        deadline = time.time() + timeout
        while time.time() < deadline:
            el = self.find_by_text(text)
            if el is not None:
                return el
            time.sleep(0.2)
        return None

    def is_element_present(self, text: str) -> bool:
        """Quick check if an element with the given text is present."""
        return self.find_by_text(text) is not None

    def is_on_screen(self, text: str) -> bool:
        """Alias for is_element_present."""
        return self.is_element_present(text)

    # ── Info helpers ──────────────────────────────────────────────────────

    def get_current_activity(self) -> str:
        """Get the current Activity name."""
        try:
            return self.driver.current_activity or ""
        except Exception:
            return ""

    # ── Login detection ──────────────────────────────────────────────────

    LOGIN_KEYWORDS = [
        "登录", "注册", "手机号登录", "短信验证码登录", "密码登录",
        "请输入手机号", "获取验证码", "同意并登录", "一键登录",
        "微信登录", "QQ登录", "Apple登录", "本机号码一键登录",
        "Sign in", "Login", "Phone number",
    ]

    HOME_INDICATORS = [
        "首页", "推荐", "发现", "消息", "我的", "Home",
        "Recommend", "Discover", "Messages", "Me",
    ]

    def is_login_screen(self) -> bool:
        """Check if the current screen is a login/registration page."""
        # Strong login indicators: phone input + verification code
        has_phone = self.is_element_present("请输入手机") or self.is_element_present("手机号")
        has_code = self.is_element_present("验证码") or self.is_element_present("获取验证码")
        if has_phone and has_code:
            return True

        # Check login keywords
        login_count = sum(1 for kw in self.LOGIN_KEYWORDS if self.is_element_present(kw))
        if login_count >= 2:
            return True

        # Check for login button presence with no home indicators
        has_login_btn = (
            self.is_element_present("登录") or self.is_element_present("Login")
        )
        has_home = any(self.is_element_present(kw) for kw in self.HOME_INDICATORS)
        if has_login_btn and not has_home:
            return True

        return False

    def is_logged_in(self) -> bool:
        """Check if the user appears to be logged in (on home/main screen).

        Strategy: if we're NOT on a login screen, assume we're logged in.
        This handles the wide variety of bottom-nav labels across Chinese apps
        (首页/推荐/热门/消息/我/我的/购物/发现/直播 等).
        """
        # If we're clearly on a login screen, definitely not logged in
        if self.is_login_screen():
            return False

        # The app launched and we're 3+ seconds in — if no login elements
        # are visible, we're likely on the main app interface (logged in).
        # The noReset=True capability preserves app state across sessions.
        return True

    def get_page_source(self) -> str:
        """Get the full page XML source."""
        try:
            return self.driver.page_source
        except Exception:
            return ""

    def wait_for_app_ready(self, timeout_sec: int = 12, poll_sec: float = 0.5) -> tuple[bool, str]:
        """Wait until the app UI becomes meaningfully interactive.

        Readiness heuristics (any one can pass):
        1. Login screen detected (UI is loaded, though auth may be needed).
        2. Home indicators present.
        3. XML source contains enough visible text/description nodes.
        """
        deadline = time.time() + timeout_sec
        last_reason = "等待应用界面加载中"

        while time.time() < deadline:
            try:
                source = self.get_page_source() or ""
                if len(source) < 120:
                    last_reason = "page_source 过短，疑似仍在启动页"
                    time.sleep(poll_sec)
                    continue

                source_lower = source.lower()
                # Use page_source text checks to avoid implicit-wait-heavy element queries.
                login_tokens = [
                    "登录", "注册", "验证码", "password", "signin", "login", "手机号",
                ]
                if sum(1 for tk in login_tokens if tk in source_lower or tk in source) >= 2:
                    return True, "检测到登录相关关键词，应用已完成渲染"

                if any((kw in source) or (kw.lower() in source_lower) for kw in self.HOME_INDICATORS):
                    return True, "检测到首页/底栏关键元素"

                text_nodes = re.findall(r'text="([^"]+)"', source)
                desc_nodes = re.findall(r'content-desc="([^"]+)"', source)
                non_empty_text = [t for t in text_nodes if t and t.strip()]
                non_empty_desc = [d for d in desc_nodes if d and d.strip()]

                if len(non_empty_text) >= 3 or len(non_empty_desc) >= 2:
                    return True, "检测到足够数量的可见节点，界面已可交互"

                last_reason = (
                    f"界面节点不足(text={len(non_empty_text)}, desc={len(non_empty_desc)})"
                )
            except Exception as e:
                last_reason = f"读取界面状态失败: {e}"

            time.sleep(poll_sec)

        return False, f"超时未就绪: {last_reason}"
