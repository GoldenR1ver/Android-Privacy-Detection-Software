"""DeletionService — tests deletion/account cancellation rights via Appium.

Test scenarios:
  1. Navigate to deletion / account cancellation entry point
  2. Verify the entry exists and is clickable
  3. STOP before confirmation — NEVER execute actual deletion

Safety principle: test existence and accessibility, NOT the destructive action.
"""

from __future__ import annotations

import logging
import time

from ..models import TestCase, TestResult, TestStatus
from ..navigator import Navigator
from ..ui_automator import UIAutomator

logger = logging.getLogger(__name__)


class DeletionService:
    """Tests deletion rights (删除权) and account cancellation (注销账号).

    CRITICAL: This service NEVER performs the final confirmation.
    It only verifies that the entry point to the deletion flow exists.
    """

    DELETION_KEYWORDS = [
        "删除", "清除", "清空", "移除",
    ]

    CANCEL_KEYWORDS = [
        "注销", "注销账号", "注销账户", "账号注销", "账户注销",
        "删除账号", "删除账户", "永久删除",
    ]

    CONFIRMATION_KEYWORDS = [
        "确认", "确定", "确认注销", "确认删除",
        "我已知晓", "已知风险",
    ]

    SAFETY_STOP_KEYWORDS = [
        "注销", "删除", "不再使用", "我确认",
    ]

    def __init__(self, ui: UIAutomator, llm=None):
        self.ui = ui
        self.navigator = Navigator(ui)
        self.llm = llm

    def execute(self, driver, test_case: TestCase) -> TestResult:
        """Run deletion-related test. Stops BEFORE any destructive action."""
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
                result.notes = "登录态缺失：跳过删除权自动化执行"
                result.duration_sec = round(time.time() - start_time, 2)
                result.steps_taken = steps_taken
                return result

            # Wait for app main screen (retry for splash/ads)
            for attempt in range(8):
                main_tabs = ["首页", "我", "推荐", "热门", "消息", "发现"]
                if any(self.ui.is_element_present(t) for t in main_tabs):
                    break
                time.sleep(0.5)

            # Step 1: Navigate to the deletion/cancellation entry
            nav_steps = self.navigator.parse_path(test_case.path_text)
            if nav_steps:
                ok, msgs = self.navigator.execute(driver, nav_steps)
                steps_taken.extend(msgs)
                if not ok:
                    result.status = TestStatus.FAIL
                    result.error_message = "导航到删除/注销入口失败"
                    result.notes = "; ".join(msgs[-3:])
                    result.duration_sec = time.time() - start_time
                    result.steps_taken = steps_taken
                    return result

            # Step 2: Check for deletion-related UI elements
            deletion_found = self._verify_deletion_entry()
            steps_taken.append(
                f"删除/注销入口 → {'已找到' if deletion_found else '未找到'}"
            )

            # Step 3: Check for confirmation dialog (but DO NOT click confirm!)
            confirmation_present = self._detect_confirmation_screen()
            if confirmation_present:
                steps_taken.append(
                    "安全提示: 检测到确认弹窗 — 已停止，未执行删除操作"
                )
            else:
                steps_taken.append("未进入确认弹窗 — 仅验证入口可达性")

            # Step 4: Safety check — ensure we didn't accidentally do anything destructive
            safety_note = self._safety_check()
            steps_taken.append(safety_note)

            # Score
            result = self._score_deletion_test(
                result, deletion_found, confirmation_present,
                test_case, steps_taken,
            )

        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)[:300]
            logger.exception("DeletionService error for %s: %s", test_case.case_id, e)

        result.duration_sec = round(time.time() - start_time, 2)
        result.steps_taken = steps_taken
        return result

    def _verify_deletion_entry(self) -> bool:
        """Check if deletion/cancellation related UI elements exist on screen."""
        all_keywords = self.DELETION_KEYWORDS + self.CANCEL_KEYWORDS
        for kw in all_keywords:
            if self.ui.is_element_present(kw):
                return True
        return False

    def _detect_confirmation_screen(self) -> bool:
        """Check if we're on a confirmation dialog/popup.

        Used as a safety check — do NOT click confirm buttons.
        """
        for kw in self.CONFIRMATION_KEYWORDS:
            if self.ui.is_element_present(kw):
                return True
        return False

    def _safety_check(self) -> str:
        """Verify no accidental destructive action was performed."""
        # Check if any safety-stop keywords are on screen
        found: list[str] = []
        for kw in self.SAFETY_STOP_KEYWORDS:
            if self.ui.is_element_present(kw):
                found.append(kw)

        if found:
            return f"安全状态: 停留在确认页面（{'、'.join(found[:3])}）— 未执行实际删除"
        return "安全状态: 未进入确认流程"

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

    def _score_deletion_test(
        self,
        result: TestResult,
        entry_found: bool,
        confirmation_present: bool,
        test_case: TestCase,
        steps: list[str],
    ) -> TestResult:
        """Apply scoring based on findings."""
        if entry_found and confirmation_present:
            result.status = TestStatus.PASS
            result.findability_score = 5
            result.functionality_score = 5
            result.notes = "删除/注销入口存在，确认弹窗已出现（未执行），流程完整"
        elif entry_found:
            result.status = TestStatus.PASS
            result.findability_score = 4
            result.functionality_score = 3
            result.notes = "删除/注销入口可访问，但未触发确认弹窗（可能在更深层级）"
        else:
            result.status = TestStatus.FAIL
            result.findability_score = 1
            result.functionality_score = 1
            result.notes = "未找到删除/注销入口"

        # Efficiency
        nav_step_count = sum(1 for s in steps if "tap" in s.lower())
        if nav_step_count <= 2:
            result.efficiency_score = 5
        elif nav_step_count <= 4:
            result.efficiency_score = 4
        elif nav_step_count <= 7:
            result.efficiency_score = 3
        else:
            result.efficiency_score = 2

        # Consistency
        has_mismatch = any("未找到" in s for s in steps)
        result.consistency_score = 4 if not has_mismatch else 3

        return result
