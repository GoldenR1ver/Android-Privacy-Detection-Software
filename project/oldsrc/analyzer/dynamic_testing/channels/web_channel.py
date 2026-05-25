"""Web channel — Selenium-based web URL accessibility testing."""

from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

from ..models import TestCase, TestResult, TestStatus

logger = logging.getLogger(__name__)


class WebChannel:
    """Handles web channel testing via Selenium (or Appium WebView)."""

    def __init__(self):
        self._selenium_available = None

    def _check_selenium(self) -> bool:
        """Check if selenium is available."""
        if self._selenium_available is None:
            try:
                import selenium  # noqa: F401
                self._selenium_available = True
            except ImportError:
                logger.warning("selenium not installed — web tests will be skipped")
                self._selenium_available = False
        return self._selenium_available

    def execute(self, test_case: TestCase) -> TestResult:
        """Test whether a privacy-related URL is reachable."""
        result = TestResult(
            case_id=test_case.case_id,
            app_name=test_case.app_name,
            right_type=test_case.right_type,
            channel=test_case.channel,
        )

        start_time = time.time()

        # Extract URL from path_text
        url = test_case.path_text.strip()
        if not url.startswith("http"):
            # Try to find a URL in the text
            import re
            match = re.search(r"https?://[^\s,;，；]+", test_case.path_text)
            if match:
                url = match.group(0)
            else:
                result.status = TestStatus.SKIP
                result.error_message = f"path_text 中未提取到有效URL: {test_case.path_text[:100]}"
                result.notes = "需人工确认网址"
                result.duration_sec = time.time() - start_time
                return result

        if not self._check_selenium():
            result.status = TestStatus.SKIP
            result.error_message = "selenium 未安装，跳过网页测试"
            result.notes = f"目标URL: {url}"
            result.duration_sec = time.time() - start_time
            return result

        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options

            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")

            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(30)

            try:
                driver.get(url)
                result.status = TestStatus.PASS
                result.notes = f"URL可访问: {url}"
                result.functionality_score = 4
                result.findability_score = 5
                result.efficiency_score = 4
                result.consistency_score = 5
            except Exception as e:
                result.status = TestStatus.FAIL
                result.error_message = f"URL不可访问: {str(e)[:200]}"
                result.notes = f"目标URL: {url}"
                result.functionality_score = 1
                result.findability_score = 1
                result.efficiency_score = 1
            finally:
                driver.quit()

        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)[:300]

        result.duration_sec = round(time.time() - start_time, 2)
        return result
