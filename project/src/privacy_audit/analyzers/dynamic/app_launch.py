"""ADB helpers to launch an app and verify it is in the foreground."""

from __future__ import annotations

import logging
import re
import subprocess
import time
from typing import Any

logger = logging.getLogger("privacy_audit.appium")

# Packages where Appium must not auto-resolve launcher activity at session start.
PACKAGES_SKIP_APPIUM_AUTO_LAUNCH: frozenset[str] = frozenset(
    {
        "com.lvsongguo",  # 小荷健康
    }
)

_FOCUS_PKG_RE = re.compile(r"mCurrentFocus=Window\{[^}]+\s+u\d+\s+([^\s/]+)/")
_RESUMED_PKG_RE = re.compile(r"mResumedActivity:.*?([a-zA-Z0-9_.]+)/")


def _adb_shell(args: list[str], *, timeout: float = 12.0) -> str:
    try:
        proc = subprocess.run(
            ["adb", "shell", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        logger.debug("adb shell %s failed: %s", args[:3], exc)
        return ""
    if proc.returncode != 0:
        logger.debug("adb shell rc=%s stderr=%s", proc.returncode, (proc.stderr or "")[:200])
    return (proc.stdout or "") + (proc.stderr or "")


def adb_foreground_package() -> str:
    """Best-effort current foreground package from dumpsys."""
    for args in (
        ["dumpsys", "window", "displays"],
        ["dumpsys", "activity", "activities"],
    ):
        text = _adb_shell(args)
        if not text:
            continue
        for line in text.splitlines():
            if "mCurrentFocus" in line:
                match = _FOCUS_PKG_RE.search(line)
                if match:
                    return match.group(1).strip()
            if "mResumedActivity" in line:
                match = _RESUMED_PKG_RE.search(line)
                if match:
                    return match.group(1).strip()
    return ""


def adb_launch_package(app_package: str, *, wait_sec: float = 1.8) -> bool:
    """Bring app to foreground via launcher intent (does not require Appium session)."""
    if not app_package:
        return False
    commands = [
        [
            "monkey",
            "-p",
            app_package,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        ],
        [
            "am",
            "start",
            "-a",
            "android.intent.action.MAIN",
            "-c",
            "android.intent.category.LAUNCHER",
            "-p",
            app_package,
        ],
    ]
    for cmd in commands:
        out = _adb_shell(cmd)
        if "Error" in out and "does not exist" in out:
            continue
        time.sleep(wait_sec)
        fg = adb_foreground_package()
        if fg == app_package:
            logger.info("adb_launch ok package=%s (foreground confirmed)", app_package)
            return True
        logger.info(
            "adb_launch tried %s; foreground=%s (expected %s)",
            " ".join(cmd[:4]),
            fg or "(unknown)",
            app_package,
        )
    fg = adb_foreground_package()
    if fg == app_package:
        return True
    logger.warning(
        "adb_launch could not confirm foreground for %s (saw %s)",
        app_package,
        fg or "(unknown)",
    )
    return False


def ensure_target_app_foreground(
    app_package: str,
    *,
    steps: list[str] | None = None,
    driver: Any | None = None,
    max_attempts: int = 3,
) -> bool:
    """
    Ensure the target package is foreground before / during Appium automation.

    Uses adb launch first, then optional Appium activate_app.
    """
    if not app_package:
        return False
    log_steps = steps if steps is not None else []
    for attempt in range(1, max_attempts + 1):
        fg = adb_foreground_package()
        if fg == app_package:
            log_steps.append(f"foreground_ok[{app_package}]=attempt{attempt}")
            return True
        log_steps.append(
            f"foreground_launch_attempt{attempt}:was={fg or 'unknown'}:target={app_package}"
        )
        adb_launch_package(app_package)
        if driver is not None:
            try:
                driver.activate_app(app_package)
                time.sleep(1.0)
            except Exception as exc:
                log_steps.append(f"activate_app_attempt{attempt}=fail({exc})")
        fg = adb_foreground_package()
        current = ""
        if driver is not None:
            try:
                current = str(getattr(driver, "current_package", "") or "")
            except Exception:
                current = ""
        if fg == app_package or current == app_package:
            log_steps.append(f"foreground_ok[{app_package}]=post_activate{attempt}")
            return True
        time.sleep(0.8)
    log_steps.append(f"foreground_fail[{app_package}]")
    return False
