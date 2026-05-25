"""Capture UI hierarchy and screenshot from Android emulator (Appium or adb)."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from privacy_audit.utils.ui_hierarchy import analyze_sidebar_state, parse_page_source, summarize_clickable_labels

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = _PACKAGE_DIR / "output"


def default_output_dir() -> Path:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_label(label: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff-]+", "_", label.strip())
    return text.strip("_") or "screen"


def _prepare_capture_session(root_dir: Path, label: str) -> tuple[Path, dict[str, Path]]:
    """Create output/{stamp}_{label}/ and return fixed artifact paths inside it."""
    stamp = _utc_stamp()
    safe = _safe_label(label)
    capture_dir = root_dir / f"{stamp}_{safe}"
    capture_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "xml": capture_dir / "hierarchy.xml",
        "png": capture_dir / "screenshot.png",
        "json": capture_dir / "summary.json",
    }
    return capture_dir, paths


def _adb_dump_xml(device: str | None, dest_xml: Path) -> dict[str, Any]:
    adb = "adb"
    remote = "/sdcard/window_dump.xml"
    prefix = [adb]
    if device:
        prefix.extend(["-s", device])
    try:
        subprocess.run(
            [*prefix, "shell", "uiautomator", "dump", remote],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        subprocess.run(
            [*prefix, "pull", remote, str(dest_xml)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        xml_text = dest_xml.read_text(encoding="utf-8", errors="replace")
        return {"ok": True, "error": "", "source": "adb_uiautomator", "xml_chars": len(xml_text)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "source": "adb_uiautomator", "xml_chars": 0}


def _adb_screencap(device: str | None, dest_png: Path) -> dict[str, Any]:
    adb = "adb"
    remote = "/sdcard/screengrab.png"
    prefix = [adb]
    if device:
        prefix.extend(["-s", device])
    try:
        subprocess.run(
            [*prefix, "shell", "screencap", "-p", remote],
            check=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            [*prefix, "pull", remote, str(dest_png)],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return {"ok": True, "error": "", "source": "adb_screencap"}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "source": "adb_screencap"}


def _capture_via_appium(
    appium_server_url: str,
    app_package: str,
    device_name: str,
    dest_xml: Path,
    dest_png: Path,
) -> tuple[str, dict[str, Any]]:
    from privacy_audit.analyzers.dynamic.appium_runner import _configure_driver_timeouts
    from privacy_audit.utils.runtime_preflight import check_appium_server

    meta: dict[str, Any] = {"source": "appium", "ok": False, "error": ""}
    check = check_appium_server(appium_server_url, timeout_sec=5.0)
    if not check.get("ok"):
        meta["error"] = check.get("error", "Appium unreachable")
        return "", meta

    try:
        from appium import webdriver  # type: ignore
        from appium.options.android import UiAutomator2Options  # type: ignore
    except Exception as exc:
        meta["error"] = f"Appium client missing: {exc}"
        return "", meta

    options = UiAutomator2Options()
    options.set_capability("platformName", "Android")
    options.set_capability("automationName", "UiAutomator2")
    options.set_capability("deviceName", device_name)
    if app_package:
        options.set_capability("appium:appPackage", app_package)
        options.set_capability("appium:appWaitActivity", "*")
    options.set_capability("appium:noReset", "true")

    driver = None
    try:
        driver = webdriver.Remote(appium_server_url, options=options)
        _configure_driver_timeouts(driver, 2.0)
        if app_package:
            driver.activate_app(app_package)
            time.sleep(1.0)
        xml_text = driver.page_source or ""
        dest_xml.write_text(xml_text, encoding="utf-8")
        driver.save_screenshot(str(dest_png))
        meta["ok"] = True
        meta["xml_chars"] = len(xml_text)
        return xml_text, meta
    except Exception as exc:
        meta["error"] = str(exc)
        return "", meta
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


def capture_emulator_screen(
    label: str = "capture",
    *,
    output_dir: Path | None = None,
    mode: str = "auto",
    appium_server_url: str = "http://127.0.0.1:4723",
    app_package: str = "",
    device_name: str = "Android",
    adb_device: str | None = None,
) -> dict[str, Any]:
    """
    Save hierarchy XML, PNG screenshot, element summary JSON, and sidebar analysis.

    Each run writes to ``output_dir/{UTCstamp}_{label}/`` (hierarchy.xml, screenshot.png, summary.json).

    mode: auto | appium | adb  (auto tries Appium when app_package set, else adb)
    """
    root_dir = output_dir or default_output_dir()
    root_dir.mkdir(parents=True, exist_ok=True)
    capture_dir, paths = _prepare_capture_session(root_dir, label)

    xml_text = ""
    capture_meta: dict[str, Any] = {}

    use_appium = mode in ("auto", "appium") and (mode == "appium" or bool(app_package))
    if use_appium:
        xml_text, capture_meta = _capture_via_appium(
            appium_server_url, app_package, device_name, paths["xml"], paths["png"]
        )

    if not capture_meta.get("ok"):
        if mode == "appium":
            pass
        else:
            adb_xml = _adb_dump_xml(adb_device, paths["xml"])
            adb_png = _adb_screencap(adb_device, paths["png"])
            capture_meta = {
                "source": "adb",
                "ok": adb_xml.get("ok") and adb_png.get("ok"),
                "xml": adb_xml,
                "png": adb_png,
                "error": adb_xml.get("error") or adb_png.get("error"),
            }
            if paths["xml"].exists():
                xml_text = paths["xml"].read_text(encoding="utf-8", errors="replace")

    sidebar_state = analyze_sidebar_state(xml_text)
    elements = parse_page_source(xml_text)
    report = {
        "label": label,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_dir": str(capture_dir),
        "capture": capture_meta,
        "files": {k: str(v) for k, v in paths.items()},
        "element_count": len(elements),
        "clickable_summary": summarize_clickable_labels(xml_text, max_items=60),
        "sidebar_xml_analysis": {
            "drawer_open": sidebar_state.drawer_open,
            "trigger_present": sidebar_state.trigger_present,
            "menu_entries_visible": sidebar_state.menu_entries_visible,
            "evidence": sidebar_state.evidence,
        },
    }
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ok": bool(capture_meta.get("ok")),
        "label": label,
        "output_dir": str(root_dir),
        "capture_dir": str(capture_dir),
        **{k: str(v) for k, v in paths.items()},
        "report": report,
    }


def capture_with_active_driver(driver: Any, label: str, output_dir: Path | None = None) -> dict[str, Any]:
    """Capture using an existing Appium session (e.g. mid-navigation debug)."""
    root_dir = output_dir or default_output_dir()
    root_dir.mkdir(parents=True, exist_ok=True)
    capture_dir, paths = _prepare_capture_session(root_dir, label)
    xml_text = driver.page_source or ""
    paths["xml"].write_text(xml_text, encoding="utf-8")
    driver.save_screenshot(str(paths["png"]))
    state = analyze_sidebar_state(xml_text)
    report = {
        "label": label,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_dir": str(capture_dir),
        "capture": {"source": "active_driver", "ok": True},
        "files": {k: str(v) for k, v in paths.items()},
        "element_count": len(parse_page_source(xml_text)),
        "clickable_summary": summarize_clickable_labels(xml_text, max_items=60),
        "sidebar_xml_analysis": {
            "drawer_open": state.drawer_open,
            "trigger_present": state.trigger_present,
            "menu_entries_visible": state.menu_entries_visible,
            "evidence": state.evidence,
        },
    }
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "label": label,
        "output_dir": str(root_dir),
        "capture_dir": str(capture_dir),
        **{k: str(v) for k, v in paths.items()},
        "report": report,
    }
