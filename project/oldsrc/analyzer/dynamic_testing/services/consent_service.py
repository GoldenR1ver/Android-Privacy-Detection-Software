"""ConsentService — tests consent/withdraw consent rights via Appium."""

from __future__ import annotations

import logging
import time
from typing import List

from ..models import TestCase, TestResult, TestStatus
from ..navigator import Navigator
from ..ui_automator import UIAutomator

logger = logging.getLogger(__name__)


class ConsentService:
    """Tests consent (同意权) and withdraw consent (撤回同意权) in Android apps.

    Test scenarios:
      1. Permission toggles exist and are functional
      2. Privacy settings screen is reachable
      3. Withdraw consent entry path exists
      4. Personalized recommendation toggle exists
    """

    # Keywords for privacy-related UI elements
    PRIVACY_KEYWORDS = [
        "隐私", "权限", "个性化", "推荐", "同意", "授权", "设置",
        "隐私设置", "权限管理", "个人信息", "隐私政策",
    ]

    TOGGLE_KEYWORDS = [
        "个性化推荐", "个性化广告", "定向推送", "精准推荐",
        "消息推送", "通知",
    ]

    WITHDRAW_KEYWORDS = [
        "撤回同意", "撤回", "撤销", "取消授权", "关闭",
    ]

    def __init__(self, ui: UIAutomator, llm=None):
        self.ui = ui
        self.navigator = Navigator(ui)
        self.llm = llm

    def execute(self, driver, test_case: TestCase) -> TestResult:
        """Run consent-related test for the given test case."""
        steps_taken: list[str] = []
        start_time = time.time()

        result = TestResult(
            case_id=test_case.case_id,
            app_name=test_case.app_name,
            right_type=test_case.right_type,
            channel=test_case.channel,
        )

        try:
            self.navigator.set_app_context(test_case.app_name)
            self._reset_to_main_tabs()
            if self.ui.is_login_screen():
                result.status = TestStatus.SKIP
                result.error_message = "检测到登录界面，需手动登录后再测"
                result.notes = "登录态缺失：跳过同意权自动化执行"
                result.duration_sec = round(time.time() - start_time, 2)
                result.steps_taken = steps_taken
                return result

            # Wait for app main screen (retry for splash/ads)
            for attempt in range(8):
                main_tabs = ["首页", "我", "推荐", "热门", "消息", "发现"]
                if any(self.ui.is_element_present(t) for t in main_tabs):
                    break
                time.sleep(0.5)

            # Step 1: Navigate to privacy/consent area
            nav_steps = self.navigator.parse_path(test_case.path_text)
            if nav_steps:
                ok, msgs = self.navigator.execute(driver, nav_steps)
                steps_taken.extend(msgs)
                if not ok:
                    result.status = TestStatus.FAIL
                    result.error_message = "导航到目标路径失败"
                    result.notes = "; ".join(msgs[-3:])
                    result.duration_sec = time.time() - start_time
                    result.steps_taken = steps_taken
                    return result
            else:
                # No structured path — try keyword-based probing
                steps_taken.append("无结构化路径文本，执行关键词搜索式导航")

            # Step 2: Check for privacy settings screen
            privacy_found = self._verify_privacy_settings_screen()
            if privacy_found:
                steps_taken.append("隐私/权限设置页面 → 已找到")
            else:
                steps_taken.append("隐私/权限设置页面 → 未明确检测到相关控件")

            # Step 3: Check for toggles
            toggle_found = self._verify_toggle_exists()
            if toggle_found:
                steps_taken.append("个性化推荐/权限开关 → 已找到")
            else:
                steps_taken.append("个性化推荐/权限开关 → 未找到")

            # Step 4: If withdrawing consent, check for withdraw entry
            if test_case.right_type == "withdraw_consent":
                withdraw_found = self._verify_withdraw_entry()
                if withdraw_found:
                    steps_taken.append("撤回同意入口 → 已找到")
                else:
                    steps_taken.append("撤回同意入口 → 未明确找到")
            else:
                steps_taken.append("同意权验证 → 隐私设置页面可访达")

            # Score and determine status
            result = self._score_consent_test(
                result, privacy_found, toggle_found, test_case, steps_taken
            )

        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)[:300]
            logger.exception("ConsentService error for %s: %s", test_case.case_id, e)

        result.duration_sec = round(time.time() - start_time, 2)
        result.steps_taken = steps_taken
        return result

    def _verify_privacy_settings_screen(self) -> bool:
        """Check if the current screen appears to be a privacy-related screen."""
        for kw in self.PRIVACY_KEYWORDS:
            if self.ui.is_element_present(kw):
                return True
        return False

    def _verify_toggle_exists(self) -> bool:
        """Check for the presence of a toggle or switch control."""
        for kw in self.TOGGLE_KEYWORDS:
            if self.ui.is_element_present(kw):
                return True
        # Also check for Switch/Toggle widgets
        try:
            from selenium.webdriver.common.by import By
            switches = self.ui.driver.find_elements(By.CLASS_NAME, "android.widget.Switch")
            if switches:
                return True
        except Exception:
            pass
        return False

    def _verify_withdraw_entry(self) -> bool:
        """Check for withdraw consent options."""
        for kw in self.WITHDRAW_KEYWORDS:
            if self.ui.is_element_present(kw):
                return True
        return False

    def _reset_to_main_tabs(self) -> None:
        """Robust reset to app main screen: tap tab -> press back -> restart activity."""
        import time
        from selenium.webdriver.common.by import By

        for tab in ["首页", "推荐", "热门"]:
            try:
                el = self.ui.driver.find_element(By.XPATH, '//*[@text="' + tab + '"]')
                el.click()
                time.sleep(0.5)
                return
            except Exception:
                pass
            try:
                sel = 'new UiSelector().text("' + tab + '")'
                el = self.ui.driver.find_element(By.ANDROID_UIAUTOMATOR, sel)
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
            pkg = self.ui.driver.capabilities.get("appPackage", "")
            act = self.ui.driver.capabilities.get("appActivity", ".index.v2.IndexActivityV2")
            self.ui.driver.start_activity(pkg, act)
            time.sleep(1.0)
        except Exception:
            pass

    def _score_consent_test(
        self,
        result: TestResult,
        privacy_found: bool,
        toggle_found: bool,
        test_case: TestCase,
        steps: list[str],
    ) -> TestResult:
        """Apply scoring based on findings."""
        if privacy_found and toggle_found:
            result.status = TestStatus.PASS
            result.findability_score = 5
            result.functionality_score = 5
            result.notes = "隐私设置页面和权限开关均已找到，功能可用"
        elif privacy_found:
            result.status = TestStatus.PASS
            result.findability_score = 4
            result.functionality_score = 3
            result.notes = "隐私设置页面可访问，但未检测到具体开关控件"
        elif toggle_found:
            result.status = TestStatus.PASS
            result.findability_score = 3
            result.functionality_score = 3
            result.notes = "检测到开关控件，但无法确认所在页面为隐私设置页面"
        else:
            result.status = TestStatus.FAIL
            result.findability_score = 1
            result.functionality_score = 1
            result.notes = "未找到隐私设置页面或相关控件"

        # Efficiency: count steps taken
        nav_step_count = sum(1 for s in steps if "tap" in s.lower())
        if nav_step_count <= 2:
            result.efficiency_score = 5
        elif nav_step_count <= 4:
            result.efficiency_score = 4
        elif nav_step_count <= 7:
            result.efficiency_score = 3
        else:
            result.efficiency_score = 2

        # Consistency: check if any step mentions "未找到"
        has_mismatch = any("未找到" in s for s in steps)
        result.consistency_score = 3 if has_mismatch else 4

        return result
