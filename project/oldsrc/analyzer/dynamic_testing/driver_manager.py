"""Appium driver session manager for MuMu emulator."""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Optional

from .config import (
    APPIUM_HOST,
    APPIUM_PORT,
    APPIUM_CAPS,
    IMPLICIT_WAIT_SEC,
    COMMAND_TIMEOUT_SEC,
    MUMU_HOST,
    MUMU_ADB_PORT,
)

logger = logging.getLogger(__name__)


class DriverManager:
    """Manages Appium WebDriver sessions for Android apps on MuMu emulator."""

    def __init__(self, host: str = APPIUM_HOST, port: int = APPIUM_PORT):
        self.host = host
        self.port = port
        self.driver: Optional[object] = None
        self._current_pkg: Optional[str] = None
        self._device_serial: Optional[str] = None

    # ── ADB helpers ──────────────────────────────────────────────────────

    def _adb(self, *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
        """Run an ADB command, automatically selecting the device if needed."""
        cmd = ["adb"]
        if self._device_serial:
            cmd.extend(["-s", self._device_serial])
        cmd.extend(args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def _find_device_serial(self) -> Optional[str]:
        """Find the best device serial to use. Prefers the MuMu TCP address."""
        try:
            result = subprocess.run(
                ["adb", "devices"], capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split("\n")[1:]
            for line in lines:
                if not line.strip().endswith("device"):
                    continue
                serial = line.split()[0].strip()
                # Prefer MuMu address
                if MUMU_HOST in serial:
                    return serial
                # Accept emulator
                if serial.startswith("emulator-"):
                    return serial
                # Accept any device
                return serial
            return None
        except Exception:
            return None

    def ensure_mumu_connected(self) -> bool:
        """Connect ADB to the MuMu emulator if not already connected."""
        mumu_addr = f"{MUMU_HOST}:{MUMU_ADB_PORT}"
        try:
            result = subprocess.run(
                ["adb", "devices"], capture_output=True, text=True, timeout=10
            )
            if mumu_addr in result.stdout:
                logger.info("MuMu emulator already connected: %s", mumu_addr)
                self._device_serial = mumu_addr
                return True

            logger.info("Connecting to MuMu emulator at %s ...", mumu_addr)
            subprocess.run(
                ["adb", "connect", mumu_addr],
                capture_output=True, text=True, timeout=15,
            )
            time.sleep(0.5)
            self._device_serial = mumu_addr
            return True
        except FileNotFoundError:
            logger.error("adb not found in PATH. Please install Android SDK platform-tools.")
            return False
        except Exception as e:
            logger.error("Failed to connect to MuMu: %s", e)
            # Fallback: try to find any connected device
            serial = self._find_device_serial()
            if serial:
                logger.info("Using fallback device: %s", serial)
                self._device_serial = serial
                return True
            return False

    def is_device_ready(self) -> bool:
        """Check if an Android device is connected and ready."""
        try:
            result = subprocess.run(
                ["adb", "devices"], capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split("\n")[1:]
            for line in lines:
                if line.strip().endswith("device"):
                    return True
            return False
        except Exception:
            return False

    # ── Session management ────────────────────────────────────────────────

    def start_session(self, app_pkg: str) -> object:
        """Start an Appium session for the given app package.

        Returns a WebDriver instance, or None on failure.
        """
        if not app_pkg:
            logger.error("Cannot start session: app_pkg is empty")
            return None

        caps = dict(APPIUM_CAPS)
        caps["appium:appPackage"] = app_pkg
        caps["appium:appWaitActivity"] = "*"
        if self._device_serial:
            caps["appium:udid"] = self._device_serial

        try:
            from appium import webdriver
            from appium.options.android import UiAutomator2Options

            options = UiAutomator2Options()
            options.platform_name = "Android"
            options.automation_name = "UiAutomator2"
            options.app_package = caps.get("appium:appPackage", "")
            if caps.get("appium:appWaitActivity"):
                options.app_wait_activity = caps["appium:appWaitActivity"]
            options.no_reset = caps.get("appium:noReset", True)
            options.auto_grant_permissions = caps.get("appium:autoGrantPermissions", True)
            options.new_command_timeout = caps.get("appium:newCommandTimeout", 120)
            if caps.get("appium:udid"):
                options.udid = caps["appium:udid"]

            url = f"http://{self.host}:{self.port}"
            logger.info("Starting Appium session for %s at %s", app_pkg, url)

            self.driver = webdriver.Remote(url, options=options)
            self.driver.implicitly_wait(IMPLICIT_WAIT_SEC)
            self._current_pkg = app_pkg

            logger.info("Session started successfully for %s", app_pkg)
            return self.driver
        except ImportError:
            logger.error(
                "Appium-Python-Client not installed. Run: pip install Appium-Python-Client"
            )
            return None
        except Exception as e:
            logger.error("Failed to start Appium session for %s: %s", app_pkg, e)
            self.driver = None
            self._current_pkg = None
            return None

    def stop_session(self) -> None:
        """Quit the current driver session."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.warning("Error quitting driver: %s", e)
            finally:
                self.driver = None
                self._current_pkg = None

    def restart_session(self, app_pkg: str) -> object:
        """Stop and restart an Appium session for the given app package."""
        self.stop_session()
        time.sleep(0.5)
        return self.start_session(app_pkg)

    # ── APK installation ──────────────────────────────────────────────────

    def is_app_installed(self, app_pkg: str) -> bool:
        """Check if an app is already installed on the device."""
        try:
            result = self._adb("shell", "pm", "list", "packages", app_pkg, timeout=10)
            return app_pkg in result.stdout
        except Exception as e:
            logger.warning("Failed to check if %s is installed: %s", app_pkg, e)
            return False

    def install_apk(self, apk_path: str) -> bool:
        """Install an APK file onto the connected device.

        Returns True if installation succeeded or the app was already installed.
        """
        try:
            logger.info("Installing APK: %s", apk_path)
            result = self._adb("install", "-r", apk_path, timeout=120)
            if "Success" in result.stdout:
                logger.info("APK installed successfully: %s", apk_path)
                return True
            else:
                logger.error("APK install failed: %s — %s", apk_path, result.stdout.strip())
                return False
        except Exception as e:
            logger.error("APK install error: %s", e)
            return False

    def ensure_app_installed(self, app_name: str, app_pkg: str) -> bool:
        """Ensure the app is installed — install from APK_DIR if missing.

        Checks if the app is already on the device. If not, looks for
        an APK file named '{app_name}.apk' in APK_DIR and installs it.
        """
        if self.is_app_installed(app_pkg):
            logger.info("App %s (%s) already installed", app_name, app_pkg)
            return True

        from .config import APK_DIR

        if APK_DIR is None:
            logger.warning(
                "App %s (%s) not installed and APK_DIR is not configured. "
                "Skipping APK install.",
                app_name, app_pkg,
            )
            return False

        apk_path = APK_DIR / f"{app_name}.apk"
        if not apk_path.exists():
            logger.warning(
                "App %s not installed and no APK found at %s. "
                "Please manually install the app or place the APK file there.",
                app_name, apk_path,
            )
            return False

        logger.info("App %s not installed — installing from %s", app_name, apk_path)
        return self.install_apk(str(apk_path))

    # ── Session helpers ───────────────────────────────────────────────────

    def force_stop_app(self, app_pkg: str) -> None:
        """Force-stop the app on the device after testing."""
        try:
            self._adb("shell", "am", "force-stop", app_pkg, timeout=10)
            logger.info("Force-stopped app: %s", app_pkg)
        except Exception as e:
            logger.warning("Failed to force-stop %s: %s", app_pkg, e)

    def is_session_active(self) -> bool:
        """Check if the current driver session is still usable."""
        if self.driver is None:
            return False
        try:
            self.driver.current_activity
            return True
        except Exception:
            return False
