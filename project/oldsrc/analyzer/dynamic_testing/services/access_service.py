"""AccessService — tests access/copy rights (访问权) via Appium.

Test scenarios:
  1. Navigate to personal info / account page
  2. Verify personal info is displayed
  3. Check data export / download function entry
  4. Compare visible data categories with policy claims
"""

from __future__ import annotations

import logging
import time

from ..models import TestCase, TestResult, TestStatus
from ..navigator import Navigator
from ..ui_automator import UIAutomator
from ..config import ACCESS_COPY_BASELINES

logger = logging.getLogger(__name__)


class AccessService:
    """Tests access rights (访问权) — personal data viewing, export, download."""

    ACCOUNT_KEYWORDS = [
        "个人信息", "账号", "我的", "资料", "账户", "个人中心",
        "账号与安全", "设置", "个人资料", "账户信息",
    ]

    EXPORT_KEYWORDS = [
        "导出", "下载", "获取副本", "数据导出", "个人信息副本",
        "下载个人信息", "导出个人信息", "数据下载",
    ]

    DATA_TYPE_KEYWORDS = [
        "手机号", "邮箱", "姓名", "身份证", "地址", "头像",
        "昵称", "性别", "生日",
    ]

    def __init__(self, ui: UIAutomator, llm=None):
        self.ui = ui
        self.navigator = Navigator(ui)
        self.llm = llm  # LLMVerifier or None

    def execute(self, driver, test_case: TestCase) -> TestResult:
        """Run access-related test for the given test case."""
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
                result.notes = "登录态缺失：跳过访问权自动化执行"
                result.duration_sec = round(time.time() - start_time, 2)
                result.steps_taken = steps_taken
                return result

            # Wait for app main screen to appear (retry for splash/ads)
            for attempt in range(8):
                main_tabs = ["首页", "我", "推荐", "热门", "消息", "发现"]
                if any(self.ui.is_element_present(t) for t in main_tabs):
                    break
                time.sleep(0.5)
            else:
                # Still can't find tabs — try anyway
                logger.warning("App main screen not detected, proceeding anyway")

            # Step 1: Navigate
            nav_steps = self.navigator.parse_path(test_case.path_text)
            if nav_steps:
                ok, msgs = self.navigator.execute(driver, nav_steps)
                steps_taken.extend(msgs)

            # Step 2: Wait for page to render, then dump screen text for LLM
            time.sleep(1.0)
            screen_text = self._dump_screen_text()
            structured = self._collect_structured_signals(test_case, screen_text)
            if structured["account_page_found"] and not structured["export_found"]:
                alt_probe = self._probe_alternative_channels(test_case, steps_taken)
                if alt_probe["probed"]:
                    steps_taken.append(alt_probe["message"])
                if alt_probe["screen_text"]:
                    screen_text = f"{screen_text}\n{alt_probe['screen_text']}".strip()
                structured["alt_channel_found"] = alt_probe["found"]
            else:
                structured["alt_channel_found"] = False

            llm_result = None
            if self.llm and self.llm.available:
                llm_result = self.llm.verify(
                    dynamic_test_goal=test_case.dynamic_test_goal,
                    path_text=test_case.path_text,
                    right_type=test_case.right_type,
                    screen_text=screen_text,
                )
                if llm_result["pass"] is not None:
                    steps_taken.append(f"LLM评估: {llm_result['summary']}")

            # Step 3: Score — prefer LLM, fall back to keywords
            if llm_result and llm_result["pass"] is not None:
                result.status = TestStatus.PASS if llm_result["pass"] else TestStatus.FAIL
                result.findability_score = llm_result["findability"]
                result.functionality_score = llm_result["functionality"]
                result.efficiency_score = llm_result["efficiency"]
                result.consistency_score = llm_result["consistency"]
                result.notes = f"LLM: {llm_result['summary']}"
                self._attach_failure_reason_label(result, test_case, structured)
            else:
                # Keyword-based fallback
                if not nav_steps:
                    result.status = TestStatus.SKIP
                    result.notes = "无导航路径，无法执行"
                else:
                    info_found = self._verify_personal_info_visible()
                    export_found = self._verify_export_function()
                    steps_taken.append(
                        f"个人信息→{'找到' if info_found else '未找到'}, "
                        f"导出功能→{'找到' if export_found else '未找到'}"
                    )
                    result = self._score_access_test(
                        result, info_found, export_found,
                        self._list_visible_data_types(), test_case, steps_taken,
                    )
                    self._attach_failure_reason_label(result, test_case, structured)

        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)[:300]
            logger.exception("AccessService error for %s: %s", test_case.case_id, e)

        result.duration_sec = round(time.time() - start_time, 2)
        result.steps_taken = steps_taken
        return result

    def _dump_screen_text(self) -> str:
        """Extract all visible text from the current screen for LLM analysis."""
        import re
        try:
            source = self.ui.get_page_source()
            texts = set()
            for m in re.finditer(r'text="([^"]+)"', source):
                t = m.group(1).strip()
                if t and 3 <= len(t) <= 100:
                    texts.add(t)
            for m in re.finditer(r'content-desc="([^"]+)"', source):
                t = m.group(1).strip()
                if t and 2 <= len(t) <= 50:
                    texts.add(t)
            return "\n".join(sorted(texts))
        except Exception:
            return ""

    def _collect_structured_signals(self, test_case: TestCase, screen_text: str) -> dict:
        """Fast rule-based signal extraction before/alongside LLM scoring."""
        source = ""
        try:
            source = self.ui.get_page_source() or ""
        except Exception:
            source = ""

        text_blob = "\n".join([screen_text or "", source]).lower()
        account_page_found = any(kw.lower() in text_blob for kw in self.ACCOUNT_KEYWORDS)
        export_found = any(kw.lower() in text_blob for kw in self.EXPORT_KEYWORDS)
        baseline = ACCESS_COPY_BASELINES.get(test_case.app_name, {})

        return {
            "account_page_found": account_page_found,
            "export_found": export_found,
            "export_copy_in_app": baseline.get("export_copy_in_app"),
            "alt_channel_found": False,
        }

    def _probe_alternative_channels(self, test_case: TestCase, steps_taken: list[str]) -> dict:
        """Try probing fallback channels (客服/工单/邮箱) when export button is missing."""
        baseline = ACCESS_COPY_BASELINES.get(test_case.app_name, {})
        keywords = baseline.get("alt_channel_keywords", [])
        paths = baseline.get("alt_channel_paths", [])

        def _scan_current() -> tuple[bool, str]:
            source = self.ui.get_page_source() or ""
            blob = source.lower()
            for kw in keywords:
                if kw and kw.lower() in blob:
                    return True, kw
            return False, ""

        found_now, kw_now = _scan_current()
        if found_now:
            return {
                "probed": True,
                "found": True,
                "message": f"替代通道探测: 当前页发现「{kw_now}」入口",
                "screen_text": self._dump_screen_text(),
            }

        for p in paths:
            nav_steps = self.navigator.parse_path(p)
            if not nav_steps:
                continue
            ok, msgs = self.navigator.execute(self.ui.driver, nav_steps)
            if msgs:
                steps_taken.extend([f"替代通道探测路径: {m}" for m in msgs])
            if not ok:
                continue
            found, kw = _scan_current()
            if found:
                return {
                    "probed": True,
                    "found": True,
                    "message": f"替代通道探测: 通过路径「{p}」发现「{kw}」入口",
                    "screen_text": self._dump_screen_text(),
                }

        return {
            "probed": bool(keywords or paths),
            "found": False,
            "message": "替代通道探测: 未发现客服/工单/邮箱类入口",
            "screen_text": "",
        }

    def _attach_failure_reason_label(self, result: TestResult, test_case: TestCase, structured: dict) -> None:
        """Attach layered failure reason labels to distinguish product gaps."""
        if result.status != TestStatus.FAIL:
            return

        account_page_found = bool(structured.get("account_page_found"))
        export_found = bool(structured.get("export_found"))
        export_copy_in_app = structured.get("export_copy_in_app")
        alt_channel_found = bool(structured.get("alt_channel_found"))

        reason = ""
        if account_page_found and not export_found:
            if export_copy_in_app is False:
                reason = "归因: 产品缺口（App内缺少导出副本入口）"
            else:
                reason = "归因: 功能缺失或未命中导出入口"
        elif not account_page_found:
            reason = "归因: 导航未达目标页"
        else:
            reason = "归因: 规则/模型判定未通过"

        alt_note = (
            "替代通道: 已发现客服/工单/邮箱入口"
            if alt_channel_found
            else "替代通道: 未发现明确入口"
        )
        if result.notes:
            result.notes = f"{result.notes} | {reason} | {alt_note}"
        else:
            result.notes = f"{reason} | {alt_note}"

    def _verify_personal_info_visible(self) -> bool:
        """Check if personal info fields are visible on current screen."""
        found_any = False
        for kw in self.ACCOUNT_KEYWORDS + self.DATA_TYPE_KEYWORDS:
            if self.ui.is_element_present(kw):
                found_any = True
        return found_any

    def _verify_export_function(self) -> bool:
        """Check if a data export/download button exists."""
        for kw in self.EXPORT_KEYWORDS:
            if self.ui.is_element_present(kw):
                return True
        return False

    def _list_visible_data_types(self) -> list[str]:
        """Scan the screen for visible personal data type labels."""
        found: list[str] = []
        for kw in self.DATA_TYPE_KEYWORDS:
            if self.ui.is_element_present(kw):
                found.append(kw)
        return found

    def _compare_data_types(
        self, claimed: str, seen: list[str]
    ) -> str:
        """Compare policy-claimed data types with what's visible in UI."""
        if not claimed:
            return "隐私政策未明确列出可访问的数据类型"
        if not seen:
            return "UI中未检测到明确的数据类型标签"
        return (
            f"政策声称可访问: {claimed[:100]}...，"
            f"UI可见: {', '.join(seen)}"
        )

    def _reset_to_main_tabs(self) -> None:
        """Robust reset to app main screen.

        Strategy: tap 首页 tab → press back → restart main activity.
        """
        import time

        # Strategy 1: Tap 首页 / 推荐 tab directly
        for tab in ["首页", "推荐", "热门"]:
            el = self.ui.find_by_xpath(f'//*[@text="{tab}"]')
            if el is not None:
                try:
                    el.click()
                    time.sleep(0.5)
                    return
                except Exception:
                    pass
            # Also try Android UI Automator
            try:
                escaped = tab.replace('"', '\\"')
                el = self.ui.driver.find_element(
                    "android", f'new UiSelector().text("{escaped}")'
                )
                if el:
                    el.click()
                    time.sleep(0.5)
                    return
            except Exception:
                pass

        # Strategy 2: Press back until we see tab indicators
        main_tabs = ["首页", "推荐", "热门", "消息", "购物", "我", "发现"]
        for _ in range(4):
            if any(self.ui.is_element_present(t) for t in main_tabs):
                return
            try:
                self.ui.driver.press_keycode(4)  # Android BACK
                time.sleep(0.3)
            except Exception:
                break

        # Strategy 3: Restart the app's main activity
        try:
            self.ui.driver.start_activity(
                self.ui.driver.capabilities.get("appPackage", ""),
                self.ui.driver.capabilities.get("appActivity", ".index.v2.IndexActivityV2"),
            )
            time.sleep(1.0)
        except Exception:
            pass

    def _score_access_test(
        self,
        result: TestResult,
        info_found: bool,
        export_found: bool,
        data_types: list[str],
        test_case: TestCase,
        steps: list[str],
    ) -> TestResult:
        """Apply scoring based on findings."""
        if info_found and export_found:
            result.status = TestStatus.PASS
            result.findability_score = 5
            result.functionality_score = 5
            result.notes = "个人信息页面和数据导出功能均可用"
        elif info_found and not export_found:
            result.status = TestStatus.PASS
            result.findability_score = 4
            result.functionality_score = 3
            result.notes = "个人信息页面可访问，但未找到数据导出/下载功能"
        elif not info_found and export_found:
            result.status = TestStatus.PASS
            result.findability_score = 3
            result.functionality_score = 3
            result.notes = "找到数据导出功能，但未确认个人信息展示"
        else:
            result.status = TestStatus.FAIL
            result.findability_score = 1
            result.functionality_score = 1
            result.notes = "未找到个人信息页面和数据导出功能"

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
        result.consistency_score = 3 if has_mismatch else 4

        return result
