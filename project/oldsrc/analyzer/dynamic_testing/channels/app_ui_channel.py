"""App UI channel — full Appium-based in-app automation testing."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from ..models import TestCase, TestResult, TestStatus
from ..navigator import Navigator
from ..ui_automator import UIAutomator

logger = logging.getLogger(__name__)


class AppUIChannel:
    """Handles app_ui channel testing via Appium UIAutomator2."""

    def __init__(self, ui: UIAutomator, llm=None):
        self.ui = ui
        self.navigator = Navigator(ui)
        self.llm = llm

    def execute(self, driver, test_case: TestCase) -> TestResult:
        """Execute a single app_ui test case.

        Navigates to the target feature described in path_text,
        verifies its existence, and records the result.
        """
        result = TestResult(
            case_id=test_case.case_id,
            app_name=test_case.app_name,
            right_type=test_case.right_type,
            channel=test_case.channel,
        )

        steps_taken: list[str] = []
        start_time = time.time()

        try:
            # Step 1: Fresh session — normalize to predictable starting screen.
            self.navigator.set_app_context(test_case.app_name)
            self._reset_to_main_tabs()
            if self._detect_login_screen():
                result.status = TestStatus.SKIP
                result.error_message = "检测到登录界面，需手动登录后再测"
                result.notes = "登录态缺失：跳过自动化执行"
                result.duration_sec = round(time.time() - start_time, 2)
                return result

            # Step 2: Parse and execute the navigation path
            nav_steps = self.navigator.parse_path(test_case.path_text)
            if not nav_steps:
                result.status = TestStatus.SKIP
                result.error_message = f"无法解析路径文本: '{test_case.path_text}'"
                result.notes = "路径文本为空或无法解析"
                result.duration_sec = time.time() - start_time
                return result

            logger.info("Executing %d nav steps for %s", len(nav_steps), test_case.case_id)
            success, step_msgs = self.navigator.execute(driver, nav_steps)
            steps_taken.extend(step_msgs)

            # Step 3: Determine outcome
            if success:
                result.status = TestStatus.PASS
                result.notes = "导航成功，目标功能/入口已定位"
                result.findability_score = min(5, max(1, 5 - len(nav_steps) + 1))
                result.functionality_score = 4  # entry found
                result.efficiency_score = min(5, max(1, 5 - len(nav_steps) + 1))
            else:
                result.status = TestStatus.FAIL
                last_step = step_msgs[-1] if step_msgs else "未知"
                result.error_message = f"导航失败: {last_step}"
                result.findability_score = 1
                result.functionality_score = 1
                result.efficiency_score = 1

            # Step 4: Check login barrier
            login_detected = self._detect_login_screen()
            if login_detected:
                result.notes += "；检测到登录界面，部分功能可能需登录后验证"
                result.requires_login = True

            # Step 5: Score consistency —— compare path_text with actual UI labels found
            consistency = self._score_consistency(nav_steps, step_msgs)
            result.consistency_score = consistency

        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)[:300]
            logger.exception("Error in app_ui test %s: %s", test_case.case_id, e)

        result.duration_sec = round(time.time() - start_time, 2)
        result.steps_taken = steps_taken

        return result

    def _detect_login_screen(self) -> bool:
        """Heuristic: check if current screen looks like a login page."""
        login_keywords = ["登录", "注册", "手机号", "验证码", "密码", "Sign in", "Login"]
        for kw in login_keywords:
            if self.ui.is_element_present(kw):
                return True
        # Check for phone number / password input patterns
        source = self.ui.get_page_source().lower()
        login_hints = ["edittext", "password"]
        return sum(1 for h in login_hints if h in source) >= 2

    def _reset_to_main_tabs(self) -> None:
        """Robust reset to app main screen: tap tab → press back → restart activity."""
        import time
        for tab in ["首页", "推荐", "热门"]:
            el = self.ui.find_by_xpath(f'//*[@text="{tab}"]')
            if el is not None:
                try:
                    el.click()
                    time.sleep(0.5)
                    return
                except Exception:
                    pass
            try:
                escaped = tab.replace('"', '\\"')
                el = self.ui.driver.find_element("android", f'new UiSelector().text("{escaped}")')
                if el:
                    el.click()
                    time.sleep(0.5)
                    return
            except Exception:
                pass
        main_tabs = ["首页", "推荐", "热门", "消息", "购物", "我", "发现"]
        for _ in range(4):
            if any(self.ui.is_element_present(t) for t in main_tabs):
                return
            try:
                self.ui.driver.press_keycode(4)
                time.sleep(0.3)
            except Exception:
                break
        try:
            self.ui.driver.start_activity(
                self.ui.driver.capabilities.get("appPackage", ""),
                self.ui.driver.capabilities.get("appActivity", ".index.v2.IndexActivityV2"),
            )
            time.sleep(1.0)
        except Exception:
            pass

    def _score_consistency(self, nav_steps, step_msgs) -> int:
        """Score path_text vs actual UI consistency (1-5)."""
        total_steps = len(nav_steps)
        if total_steps == 0:
            return 1

        found_count = sum(1 for m in step_msgs if "未找到" not in m and "失败" not in m)
        ratio = found_count / max(total_steps, 1)

        if ratio >= 0.9:
            return 5
        if ratio >= 0.7:
            return 4
        if ratio >= 0.5:
            return 3
        if ratio >= 0.3:
            return 2
        return 1
