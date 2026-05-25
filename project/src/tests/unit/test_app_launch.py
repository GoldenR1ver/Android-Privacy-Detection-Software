"""Unit tests for adb foreground / launch helpers."""

from privacy_audit.analyzers.dynamic.app_launch import (
    PACKAGES_SKIP_APPIUM_AUTO_LAUNCH,
    _FOCUS_PKG_RE,
    _RESUMED_PKG_RE,
)


def test_focus_regex() -> None:
    line = "  mCurrentFocus=Window{abc u0 com.eg.android.AlipayGphone/com.alipay.mobile}"
    match = _FOCUS_PKG_RE.search(line)
    assert match
    assert match.group(1) == "com.eg.android.AlipayGphone"


def test_resumed_regex() -> None:
    line = "    mResumedActivity: ActivityRecord{abc u0 com.wuba/.MainActivity t123}"
    match = _RESUMED_PKG_RE.search(line)
    assert match
    assert match.group(1) == "com.wuba"


def test_xiaohe_skips_appium_autolaunch() -> None:
    assert "com.lvsongguo" in PACKAGES_SKIP_APPIUM_AUTO_LAUNCH
    assert "com.eg.android.AlipayGphone" not in PACKAGES_SKIP_APPIUM_AUTO_LAUNCH
