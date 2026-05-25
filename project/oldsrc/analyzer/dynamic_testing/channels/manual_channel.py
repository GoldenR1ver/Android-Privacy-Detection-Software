"""Manual channel — verification stubs for email/phone/offline/customer_service channels.

These channels cannot be fully automated. This module documents what was found
and flags items for manual human verification.
"""

from __future__ import annotations

import logging

from ..models import TestCase, TestResult, TestStatus

logger = logging.getLogger(__name__)


class ManualChannel:
    """Handles email, phone, mail, offline, customer_service, and unknown channels."""

    def execute(self, test_case: TestCase) -> TestResult:
        """Document the expected verification path for manual review."""
        result = TestResult(
            case_id=test_case.case_id,
            app_name=test_case.app_name,
            right_type=test_case.right_type,
            channel=test_case.channel,
        )

        channel = test_case.channel

        if channel == "email":
            return self._handle_email(test_case, result)
        elif channel == "phone":
            return self._handle_phone(test_case, result)
        elif channel == "customer_service":
            return self._handle_customer_service(test_case, result)
        elif channel in ("mail", "offline"):
            return self._handle_offline(test_case, result)
        else:
            return self._handle_unknown(test_case, result)

    def _handle_email(self, tc: TestCase, result: TestResult) -> TestResult:
        result.status = TestStatus.SKIP
        result.error_message = f"邮件渠道需要人工验证: {tc.path_text}"
        result.notes = (
            f"请向 {tc.path_text} 发送验证邮件，"
            f"验证目标: {tc.dynamic_test_goal}，"
            f"并记录响应时间和内容完整性。"
        )
        # Can't score automated channels properly — mark as manual
        result.findability_score = 3  # email address was found in policy
        result.functionality_score = 0  # not verified
        result.efficiency_score = 0
        result.consistency_score = 3
        return result

    def _handle_phone(self, tc: TestCase, result: TestResult) -> TestResult:
        result.status = TestStatus.SKIP
        result.error_message = f"电话渠道需要人工验证: {tc.path_text}"
        result.notes = (
            f"请拨打 {tc.path_text}，"
            f"验证目标: {tc.dynamic_test_goal}，"
            f"记录接通情况、客服响应和操作指引。"
        )
        result.findability_score = 3
        result.functionality_score = 0
        result.efficiency_score = 0
        result.consistency_score = 3
        return result

    def _handle_customer_service(self, tc: TestCase, result: TestResult) -> TestResult:
        result.status = TestStatus.SKIP
        result.error_message = f"在线客服渠道需要人工验证: {tc.path_text}"
        result.notes = (
            f"请在APP内找到客服入口（{tc.path_text}），"
            f"验证目标: {tc.dynamic_test_goal}，"
            f"记录客服入口位置、响应速度和问题解决能力。"
        )
        result.findability_score = 3
        result.functionality_score = 0
        result.efficiency_score = 2
        result.consistency_score = 3
        return result

    def _handle_offline(self, tc: TestCase, result: TestResult) -> TestResult:
        result.status = TestStatus.SKIP
        result.error_message = f"线下/邮寄渠道无法自动化验证: {tc.path_text}"
        result.notes = (
            f"线下验证: {tc.dynamic_test_goal}，"
            f"地址信息: {tc.path_text}，"
            f"需前往线下网点确认。"
        )
        result.findability_score = 2
        result.functionality_score = 0
        result.efficiency_score = 0
        result.consistency_score = 3
        return result

    def _handle_unknown(self, tc: TestCase, result: TestResult) -> TestResult:
        result.status = TestStatus.SKIP
        result.error_message = f"未知渠道，无法自动化验证: {tc.path_text}"
        result.notes = (
            f"渠道信息不足: {tc.path_text}，"
            f"建议人工阅读隐私政策原文确认执行方式。"
            f"政策原文: {tc.sentence[:200]}"
        )
        result.findability_score = 1
        result.functionality_score = 0
        result.efficiency_score = 0
        result.consistency_score = 1
        return result
