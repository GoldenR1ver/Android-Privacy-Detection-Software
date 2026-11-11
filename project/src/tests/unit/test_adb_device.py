"""Unit tests for adb device serial resolution."""

from __future__ import annotations

import privacy_audit.utils.adb_device as adb_device


def test_resolve_prefers_android_serial(monkeypatch) -> None:
    monkeypatch.setenv("ANDROID_SERIAL", "127.0.0.1:16384")
    monkeypatch.setattr(adb_device, "list_connected_devices", lambda **_: ["127.0.0.1:7555"])
    assert adb_device.resolve_adb_serial() == "127.0.0.1:16384"


def test_resolve_single_device(monkeypatch) -> None:
    monkeypatch.delenv("ANDROID_SERIAL", raising=False)
    monkeypatch.delenv("ADB_SERIAL", raising=False)
    monkeypatch.setattr(
        adb_device,
        "list_connected_devices",
        lambda **_: ["127.0.0.1:16384"],
    )
    assert adb_device.resolve_adb_serial() == "127.0.0.1:16384"


def test_resolve_multiple_prefers_mumu_16384(monkeypatch) -> None:
    monkeypatch.delenv("ANDROID_SERIAL", raising=False)
    monkeypatch.setattr(
        adb_device,
        "list_connected_devices",
        lambda **_: ["127.0.0.1:7555", "127.0.0.1:16384"],
    )
    assert adb_device.resolve_adb_serial() == "127.0.0.1:16384"


def test_adb_base_cmd_includes_serial(monkeypatch) -> None:
    monkeypatch.setattr(adb_device, "resolve_adb_serial", lambda **_: "127.0.0.1:16384")
    cmd = adb_device.adb_base_cmd()
    assert cmd[-2:] == ["-s", "127.0.0.1:16384"]
    assert cmd[0].endswith("adb") or cmd[0].endswith("adb.EXE")
