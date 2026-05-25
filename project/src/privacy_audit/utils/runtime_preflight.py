"""Pre-flight checks for dynamic runs (adb / Appium / Frida)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("privacy_audit.preflight")


def check_adb_devices(timeout_sec: float = 8.0) -> dict[str, Any]:
    """Return adb device list; empty if adb missing or no devices."""
    adb = shutil.which("adb")
    if not adb:
        return {"ok": False, "error": "adb not found in PATH", "devices": []}
    try:
        proc = subprocess.run(
            [adb, "devices"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"adb devices timed out after {timeout_sec}s", "devices": []}
    except OSError as exc:
        return {"ok": False, "error": str(exc), "devices": []}

    devices: list[str] = []
    for line in (proc.stdout or "").splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"adb exit {proc.returncode}"
        return {"ok": False, "error": err, "devices": devices}
    if not devices:
        return {
            "ok": False,
            "error": "no adb device in 'device' state (run: adb connect <emulator>)",
            "devices": [],
        }
    return {"ok": True, "error": "", "devices": devices}


def check_appium_server(appium_server_url: str, timeout_sec: float = 5.0) -> dict[str, Any]:
    """Ping Appium /status endpoint."""
    base = appium_server_url.rstrip("/")
    status_url = f"{base}/status"
    try:
        req = urllib.request.Request(status_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"raw": body[:500]}
        return {"ok": True, "url": status_url, "error": "", "status": payload}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "url": status_url, "error": f"HTTP {exc.code}: {exc.reason}", "status": {}}
    except Exception as exc:
        return {
            "ok": False,
            "url": status_url,
            "error": (
                f"{exc}. Is Appium Server running? "
                f"Start it (e.g. appium -a 127.0.0.1 -p 4723) before stage3-dynamic."
            ),
            "status": {},
        }


def check_frida_usb(timeout_sec: float = 8.0) -> dict[str, Any]:
    """Try to obtain Frida USB device (short timeout)."""
    try:
        import frida  # type: ignore
    except ImportError as exc:
        return {"ok": False, "error": f"frida python package not installed: {exc}", "device_id": ""}

    try:
        device = frida.get_usb_device(timeout=max(1, int(timeout_sec)))
        device_id = getattr(device, "id", "") or str(device)
        return {"ok": True, "error": "", "device_id": device_id}
    except Exception as exc:
        return {
            "ok": False,
            "error": (
                f"{exc}. Ensure frida-server is running on the emulator "
                f"(adb shell /data/local/tmp/frida-server &)."
            ),
            "device_id": "",
        }


def log_preflight_results(
    *,
    adb: dict[str, Any],
    appium: dict[str, Any] | None = None,
    frida: dict[str, Any] | None = None,
) -> bool:
    """Log preflight summaries; return True if all required checks passed."""
    all_ok = True
    if adb.get("ok"):
        logger.info("adb: OK devices=%s", adb.get("devices"))
    else:
        all_ok = False
        logger.error("adb: FAIL %s", adb.get("error"))

    if appium is not None:
        if appium.get("ok"):
            logger.info("appium: OK %s", appium.get("url"))
        else:
            all_ok = False
            logger.error("appium: FAIL %s", appium.get("error"))

    if frida is not None:
        if frida.get("ok"):
            logger.info("frida: OK device_id=%s", frida.get("device_id"))
        else:
            all_ok = False
            logger.error("frida: FAIL %s", frida.get("error"))

    return all_ok


def run_stage3_preflight(
    appium_server_url: str,
    *,
    skip_appium: bool = False,
    skip_frida: bool = False,
    appium_timeout_sec: float = 5.0,
    frida_timeout_sec: float = 8.0,
) -> dict[str, Any]:
    """Run all stage3 preflight checks and return structured result."""
    adb = check_adb_devices()
    appium = None if skip_appium else check_appium_server(appium_server_url, appium_timeout_sec)
    frida = None if skip_frida else check_frida_usb(frida_timeout_sec)
    ok = log_preflight_results(adb=adb, appium=appium, frida=frida)
    return {"ok": ok, "adb": adb, "appium": appium, "frida": frida}
