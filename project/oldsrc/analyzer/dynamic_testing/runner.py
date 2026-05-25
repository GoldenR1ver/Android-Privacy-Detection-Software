"""Test runner orchestrator — discover, execute, score, report."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from .models import TestCase, TestResult, TestStatus
from .config import TARGET_APPS, DEFAULT_LIMIT_PER_APP_PER_RIGHT
from .test_case_generator import generate_test_cases, export_test_cases_json, export_app_ui_only_json
from .driver_manager import DriverManager
from .ui_automator import UIAutomator
from .screenshot import ScreenshotCapture
from .channels.app_ui_channel import AppUIChannel
from .channels.web_channel import WebChannel
from .channels.manual_channel import ManualChannel
from .services.consent_service import ConsentService
from .services.access_service import AccessService
from .services.deletion_service import DeletionService
from .scoring.usability_scorer import UsabilityScorer
from .reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class DynamicTestRunner:
    """Orchestrates the full Part 2 pipeline: discover → execute → score → report."""

    def __init__(
        self,
        pipeline_root: Path,
        dry_run: bool = False,
    ):
        self.pipeline_root = Path(pipeline_root)
        self.dry_run = dry_run
        self.output_dir = self.pipeline_root / "dynamic_testing"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components (lazy init)
        self._driver_mgr: Optional[DriverManager] = None
        self._ui: Optional[UIAutomator] = None
        self._screenshots: Optional[ScreenshotCapture] = None

    # ── Phase 1: Discover ─────────────────────────────────────────────────

    def discover(
        self,
        apps: list[str] | None = None,
        rights: list[str] | None = None,
        limit: int = DEFAULT_LIMIT_PER_APP_PER_RIGHT,
    ) -> list[TestCase]:
        """Generate test cases from Part 1 output."""
        logger.info("Phase 1: Discover — generating test cases from Part 1 output")
        cases = generate_test_cases(
            pipeline_root=self.pipeline_root,
            apps=apps,
            rights=rights,
            limit_per_right=limit,
        )
        logger.info("Generated %d test cases", len(cases))

        # Export
        all_path = self.output_dir / "test_cases.json"
        summary = export_test_cases_json(cases, all_path)
        logger.info("Test cases saved to %s (total: %d)", all_path, summary["total"])

        ui_path = self.output_dir / "test_cases_app_ui_only.json"
        export_app_ui_only_json(cases, ui_path)

        # Save manifest
        manifest = {
            "phase": "discover",
            "pipeline_root": str(self.pipeline_root),
            "total_test_cases": summary["total"],
            "by_priority": summary["by_priority"],
            "by_right": summary["by_right"],
            "by_channel": summary["by_channel"],
            "by_app": summary["by_app"],
            "app_ui_only_count": summary["by_channel"].get("app_ui", 0),
        }
        with open(self.output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return cases

    # ── Phase 2: Execute ──────────────────────────────────────────────────

    def execute(
        self,
        test_cases: list[TestCase],
        apps: list[str] | None = None,
        resume: bool = False,
    ) -> list[TestResult]:
        """Execute test cases via Appium (or dry-run mock)."""
        logger.info("Phase 2: Execute — running %d test cases", len(test_cases))

        results: list[TestResult] = []
        target_apps = set(apps or TARGET_APPS)

        # Filter to target apps
        cases = [tc for tc in test_cases if tc.app_name in target_apps]

        if self.dry_run:
            logger.info("DRY-RUN mode — using mock results (no Appium connection)")
            results = self._dry_run_execute(cases)
            # Save results
            self._save_results(results)
            return results

        # Real execution: requires Appium + device
        self._init_driver()

        if not self._driver_mgr or not self._driver_mgr.ensure_mumu_connected():
            logger.error("Cannot connect to MuMu emulator. Falling back to dry-run.")
            results = self._dry_run_execute(cases)
            self._save_results(results)
            return results

        # Group cases by app for session efficiency
        app_cases: dict[str, list[TestCase]] = {}
        for tc in cases:
            if tc.channel == "app_ui":
                app_cases.setdefault(tc.app_name, []).append(tc)

        for app_name, app_test_cases in app_cases.items():
            # Check resume: skip if already completed
            if resume and self._app_already_tested(app_name):
                logger.info("Skipping %s — already tested", app_name)
                continue

            logger.info("Testing %s (%d cases)", app_name, len(app_test_cases))
            app_pkg = app_test_cases[0].app_pkg if app_test_cases else ""
            if not app_pkg:
                logger.warning("No app_pkg for %s, skipping", app_name)
                continue

            # Ensure app is installed before launching
            if not self._driver_mgr.ensure_app_installed(app_name, app_pkg):
                logger.error(
                    "App %s (%s) is not installed and cannot be auto-installed. "
                    "Please install it manually or put the APK in the APK_DIR.",
                    app_name, app_pkg,
                )
                for tc in app_test_cases:
                    results.append(
                        self._make_error_result(tc, f"APP未安装且无法自动安装: {app_pkg}")
                    )
                continue

            # Initialize shared components
            screen_dir = self.output_dir / "per_app" / app_name / "screenshots"
            from .llm_verifier import LLMVerifier
            llm = LLMVerifier()
            web_channel = WebChannel()
            manual_channel = ManualChannel()

            for tc in app_test_cases:
                logger.debug("Running %s", tc.case_id)

                # Open fresh session for each test case
                driver = self._driver_mgr.start_session(app_pkg)
                if driver is None:
                    results.append(self._make_error_result(tc, "无法启动Appium会话"))
                    continue

                try:
                    self._screenshots = ScreenshotCapture(screen_dir)
                    self._ui = UIAutomator(driver)
                    app_ui_channel = AppUIChannel(self._ui, llm)
                    consent_svc = ConsentService(self._ui, llm)
                    access_svc = AccessService(self._ui, llm)
                    deletion_svc = DeletionService(self._ui, llm)

                    # Wait until app UI is truly ready before running a case.
                    ready, ready_msg = self._ui.wait_for_app_ready(timeout_sec=12)
                    if not ready:
                        logger.warning("%s: 首次就绪检测失败: %s", tc.case_id, ready_msg)
                        # Recovery: try foregrounding app once, then re-check.
                        try:
                            driver.activate_app(app_pkg)
                        except Exception:
                            pass
                        time.sleep(0.5)
                        ready, ready_msg = self._ui.wait_for_app_ready(timeout_sec=8)

                    if not ready:
                        logger.warning("%s: 应用未就绪，跳过该用例: %s", tc.case_id, ready_msg)
                        result = TestResult(
                            case_id=tc.case_id,
                            app_name=tc.app_name,
                            right_type=tc.right_type,
                            channel=tc.channel,
                            status=TestStatus.SKIP,
                            error_message=f"应用未完全打开: {ready_msg}",
                            notes="应用启动后界面未达到可交互状态，已跳过本条",
                        )
                        results.append(result)
                        self._save_results(results)
                        continue

                    # Check login
                    logged_in = self._ui.is_logged_in()
                    if not logged_in:
                        logger.warning("%s: 检测到登录界面", app_name)
                        result = TestResult(
                            case_id=tc.case_id, app_name=tc.app_name,
                            right_type=tc.right_type, channel=tc.channel,
                            status=TestStatus.SKIP,
                            error_message="未登录：请先在APP中手动登录后再运行测试",
                        )
                        results.append(result)
                        self._save_results(results)
                        continue

                    # Screenshot before
                    self._screenshots.capture(driver, f"{tc.case_id}_start")

                    # Dispatch
                    if tc.channel != "app_ui":
                        if tc.channel == "web":
                            result = web_channel.execute(tc)
                        else:
                            result = manual_channel.execute(tc)
                    elif tc.right_type == "consent":
                        result = consent_svc.execute(driver, tc)
                    elif tc.right_type == "access":
                        result = access_svc.execute(driver, tc)
                    elif tc.right_type == "deletion":
                        result = deletion_svc.execute(driver, tc)
                    else:
                        result = app_ui_channel.execute(driver, tc)

                    # Screenshot after
                    if tc.channel == "app_ui":
                        result.screenshot_paths.append(
                            self._screenshots.capture(driver, f"{tc.case_id}_end") or ""
                        )

                    results.append(result)
                    self._save_results(results)

                except Exception as e:
                    logger.exception("Error in test %s: %s", tc.case_id, e)
                finally:
                    self._driver_mgr.force_stop_app(app_pkg)
                    self._driver_mgr.stop_session()
                    time.sleep(0.5)  # Short cool-down between test cases

        # Handle non-app_ui cases
        non_ui_cases = [tc for tc in cases if tc.channel != "app_ui"]
        web_channel = WebChannel()
        manual_channel = ManualChannel()
        for tc in non_ui_cases:
            if tc.channel == "web":
                result = web_channel.execute(tc)
            else:
                result = manual_channel.execute(tc)
            results.append(result)

        self._save_results(results)
        return results

    # ── Phase 3: Report ───────────────────────────────────────────────────

    def report(self, results: list[TestResult], apps: list[str] | None = None) -> dict:
        """Generate all reports from test results."""
        logger.info("Phase 3: Report — generating reports from %d results", len(results))

        reporter = ReportGenerator(self.output_dir)
        aggregate = reporter.generate(results, apps or [])

        logger.info("Reports generated in %s", self.output_dir)
        logger.info(
            "Overall: %d cases, pass_rate=%.1f%%, avg_score=%.1f",
            aggregate.get("total", 0),
            aggregate.get("overall_pass_rate", 0) * 100,
            aggregate.get("overall_avg_score", 0),
        )

        return aggregate

    # ── Full pipeline ─────────────────────────────────────────────────────

    def run_all(
        self,
        apps: list[str] | None = None,
        rights: list[str] | None = None,
        limit: int = DEFAULT_LIMIT_PER_APP_PER_RIGHT,
        resume: bool = False,
    ) -> dict:
        """Run the complete pipeline: discover → execute → report."""
        cases = self.discover(apps=apps, rights=rights, limit=limit)
        if not cases:
            logger.warning("No test cases generated")
            return {"total": 0}

        results = self.execute(cases, apps=apps, resume=resume)
        aggregate = self.report(results, apps=apps)
        return aggregate

    # ── Internal helpers ──────────────────────────────────────────────────

    def _init_driver(self) -> None:
        """Initialize the driver manager."""
        if self._driver_mgr is None:
            self._driver_mgr = DriverManager()

    def _dry_run_execute(self, cases: list[TestCase]) -> list[TestResult]:
        """Generate mock results for development/testing without a device."""
        results: list[TestResult] = []
        for tc in cases:
            import random
            # Simulate mostly-pass with some failures
            is_pass = random.random() < 0.65
            scores = {
                "findability": random.randint(3, 5) if is_pass else random.randint(1, 2),
                "functionality": random.randint(3, 5) if is_pass else random.randint(1, 2),
                "efficiency": random.randint(3, 5) if is_pass else random.randint(1, 2),
                "consistency": random.randint(3, 5) if is_pass else random.randint(1, 2),
            }
            result = TestResult(
                case_id=tc.case_id,
                app_name=tc.app_name,
                right_type=tc.right_type,
                channel=tc.channel,
                status=TestStatus.PASS if is_pass else TestStatus.FAIL,
                findability_score=scores["findability"],
                functionality_score=scores["functionality"],
                efficiency_score=scores["efficiency"],
                consistency_score=scores["consistency"],
                usability_score=0.0,  # computed later
                steps_taken=[f"mock: {'成功' if is_pass else '失败'} - 模拟执行"],
                notes=f"[DRY-RUN] 模拟测试{'通过' if is_pass else '失败'}",
                duration_sec=round(random.uniform(1.0, 5.0), 2),
            )
            results.append(result)
        return results

    def _save_results(self, results: list[TestResult]) -> None:
        """Save current results to JSON."""
        filepath = self.output_dir / "results.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)

    def _app_already_tested(self, app_name: str) -> bool:
        """Check if results already exist for the given app."""
        results_path = self.output_dir / "results.json"
        if not results_path.exists():
            return False
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            return any(r.get("app_name") == app_name for r in existing)
        except Exception:
            return False

    def _make_error_result(self, tc: TestCase, msg: str) -> TestResult:
        """Create an error result for a failed test case."""
        return TestResult(
            case_id=tc.case_id,
            app_name=tc.app_name,
            right_type=tc.right_type,
            channel=tc.channel,
            status=TestStatus.ERROR,
            error_message=msg,
            notes=f"测试无法执行: {msg}",
        )
