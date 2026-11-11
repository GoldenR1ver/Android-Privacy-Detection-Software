"""Resolve adb device serial when multiple emulators are connected."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Any

logger = logging.getLogger("privacy_audit.adb")

# MuMu 12 often exposes both ports; prefer the primary ADB port first.
PREFERRED_SERIALS: tuple[str, ...] = (
    "127.0.0.1:16384",
    "127.0.0.1:7555",
    "emulator-5554",
)


def list_connected_devices(timeout_sec: float = 8.0) -> list[str]:
    """Return adb serials in ``device`` state."""
    adb = shutil.which("adb")
    if not adb:
        return []
    try:
        proc = subprocess.run(
            [adb, "devices"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    devices: list[str] = []
    for line in (proc.stdout or "").splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def resolve_adb_serial(*, explicit: str | None = None) -> str | None:
    """Pick adb -s serial: env override, single device, or known MuMu port."""
    if explicit and explicit.strip():
        return explicit.strip()
    for env_key in ("ANDROID_SERIAL", "ADB_SERIAL"):
        value = os.environ.get(env_key, "").strip()
        if value:
            return value

    devices = list_connected_devices()
    if not devices:
        return None
    if len(devices) == 1:
        return devices[0]

    for preferred in PREFERRED_SERIALS:
        if preferred in devices:
            logger.warning(
                "Multiple adb devices %s; using %s (set ANDROID_SERIAL to override)",
                devices,
                preferred,
            )
            return preferred

    chosen = devices[0]
    logger.warning(
        "Multiple adb devices %s; using %s (set ANDROID_SERIAL to override)",
        devices,
        chosen,
    )
    return chosen


def adb_base_cmd(*, serial: str | None = None) -> list[str]:
    """Build ``adb`` or ``adb -s <serial>`` command prefix."""
    adb = shutil.which("adb") or "adb"
    resolved = resolve_adb_serial(explicit=serial)
    if resolved:
        return [adb, "-s", resolved]
    return [adb]


def adb_run(
    args: list[str],
    *,
    serial: str | None = None,
    timeout: float = 12.0,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run adb with optional device serial."""
    cmd = [*adb_base_cmd(serial=serial), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def adb_device_status(*, serial: str | None = None) -> dict[str, Any]:
    """Summary for logging / preflight."""
    devices = list_connected_devices()
    resolved = resolve_adb_serial(explicit=serial)
    return {
        "devices": devices,
        "resolved_serial": resolved,
        "multiple": len(devices) > 1,
    }
