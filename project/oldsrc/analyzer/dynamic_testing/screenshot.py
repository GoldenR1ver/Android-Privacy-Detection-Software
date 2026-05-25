"""Screenshot capture for test evidence."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ScreenshotCapture:
    """Captures and saves screenshots during Appium test execution."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, driver, label: str = "") -> Optional[str]:
        """Take a screenshot and save it with the given label.

        Returns the file path, or None on failure.
        """
        if driver is None:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_label = label.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]
        filename = f"{timestamp}_{safe_label}.png" if safe_label else f"{timestamp}.png"
        filepath = self.base_dir / filename

        try:
            driver.save_screenshot(str(filepath))
            logger.debug("Screenshot saved: %s", filepath)
            return str(filepath)
        except Exception as e:
            logger.warning("Failed to save screenshot: %s", e)
            return None
