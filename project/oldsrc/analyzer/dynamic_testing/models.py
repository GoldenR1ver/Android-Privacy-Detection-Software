"""Core data models for Part 2 dynamic testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class TestStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIP = "SKIP"


class Priority(int, Enum):
    P1 = 1  # app_ui + low risk + non-empty path_text
    P2 = 2  # app_ui + medium risk
    P3 = 3  # web/email/phone or high risk


@dataclass
class TestCase:
    case_id: str
    app_name: str
    app_pkg: str
    right_type: str  # consent | withdraw_consent | access_copy | deletion | account_cancellation
    channel: str  # app_ui | web | email | phone | customer_service | mail | offline | unknown
    path_text: str
    dynamic_test_goal: str
    usability_risk: str  # low | medium | high | unknown
    target_data: str
    sentence: str  # privacy_policy_content
    priority: int = 3
    requires_login: bool = True
    access_copy_type: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "app_name": self.app_name,
            "app_pkg": self.app_pkg,
            "right_type": self.right_type,
            "channel": self.channel,
            "path_text": self.path_text,
            "dynamic_test_goal": self.dynamic_test_goal,
            "usability_risk": self.usability_risk,
            "target_data": self.target_data,
            "sentence": self.sentence,
            "priority": self.priority,
            "requires_login": self.requires_login,
            "access_copy_type": self.access_copy_type,
        }


@dataclass
class TestResult:
    case_id: str
    app_name: str
    right_type: str
    channel: str
    status: TestStatus = TestStatus.SKIP
    findability_score: int = 1
    functionality_score: int = 1
    efficiency_score: int = 1
    consistency_score: int = 1
    usability_score: float = 0.0
    steps_taken: List[str] = field(default_factory=list)
    error_message: str = ""
    screenshot_paths: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "app_name": self.app_name,
            "right_type": self.right_type,
            "channel": self.channel,
            "status": self.status.value,
            "findability_score": self.findability_score,
            "functionality_score": self.functionality_score,
            "efficiency_score": self.efficiency_score,
            "consistency_score": self.consistency_score,
            "usability_score": self.usability_score,
            "steps_taken": self.steps_taken,
            "error_message": self.error_message,
            "screenshot_paths": self.screenshot_paths,
            "duration_sec": self.duration_sec,
            "notes": self.notes,
        }


@dataclass
class AppEvalReport:
    app_name: str
    app_pkg: str
    consent_results: List[TestResult] = field(default_factory=list)
    access_results: List[TestResult] = field(default_factory=list)
    deletion_results: List[TestResult] = field(default_factory=list)
    app_ui_results: List[TestResult] = field(default_factory=list)
    web_results: List[TestResult] = field(default_factory=list)
    manual_results: List[TestResult] = field(default_factory=list)

    @property
    def all_results(self) -> List[TestResult]:
        return (
            self.consent_results
            + self.access_results
            + self.deletion_results
        )
