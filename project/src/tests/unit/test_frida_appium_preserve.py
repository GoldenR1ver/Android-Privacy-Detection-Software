"""Frida + Appium integration guards."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from privacy_audit.analyzers.dynamic.appium_runner import _reset_app_for_path


def test_reset_app_skips_terminate_when_preserving_frida() -> None:
    driver = MagicMock()
    steps: list[str] = []
    with patch(
        "privacy_audit.analyzers.dynamic.appium_runner._navigate_back_to_home"
    ) as back_home, patch(
        "privacy_audit.analyzers.dynamic.appium_runner.ensure_target_app_foreground",
        return_value=True,
    ) as foreground:
        _reset_app_for_path(
            driver,
            "com.example.app",
            steps,
            2,
            preserve_frida_session=True,
        )
    driver.terminate_app.assert_not_called()
    back_home.assert_called_once()
    foreground.assert_called_once()
    assert "path_2_reset=frida_preserve_no_terminate" in steps


def test_reset_app_terminates_by_default() -> None:
    driver = MagicMock()
    steps: list[str] = []
    with patch(
        "privacy_audit.analyzers.dynamic.appium_runner.ensure_target_app_foreground",
        return_value=True,
    ):
        _reset_app_for_path(driver, "com.example.app", steps, 2)
    driver.terminate_app.assert_called_once_with("com.example.app")
