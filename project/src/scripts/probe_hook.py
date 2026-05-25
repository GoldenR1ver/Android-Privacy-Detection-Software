"""
Diagnose Frida attach for Keep / MuMu.

Default: try every Keep-related PID, pick one with working Java + android_id hook.

Usage:
  python scripts/probe_hook.py
  python scripts/probe_hook.py 7811
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import frida

from privacy_audit.analyzers.dynamic.hook_loader import wrap_hook_script
from privacy_audit.analyzers.dynamic.frida_runner import (
    _ordered_attach_pids,
    _verify_java_session,
)

HOOK = (
    Path(__file__).resolve().parents[1]
    / "privacy_audit/analyzers/dynamic/hooks/hook_android_id.js"
).read_text(encoding="utf-8")
PKG = "com.gotokeep.keep"
APP = "Keep"


def _try_hook(device: frida.core.Device, pid: int, label: str) -> bool:
    print(f"\n--- try pid={pid} name={label!r} ---")
    session = None
    try:
        session = device.attach(pid, persist_timeout=30)
        print("  attach: OK")
    except Exception as exc:
        print(f"  attach: FAIL ({exc})")
        return False
    try:
        if label == APP:
            print("  java_ready: skip probe on MuMu parent (hooks wait for Java after load)")
        elif not _verify_java_session(session, timeout_sec=15.0):
            print("  java_ready: FAIL (ART not available in this process)")
            return False
        else:
            print("  java_ready: OK")
        script = session.create_script(wrap_hook_script(HOOK))
        msgs: list = []

        def on_message(message, _data):
            msgs.append(message)
            print("  MSG", message)

        script.on("message", on_message)
        script.load()
        wait_sec = 25 if label == APP else 8
        time.sleep(wait_sec)
        script.unload()
        payloads = [m.get("payload") for m in msgs if isinstance(m.get("payload"), dict)]
        runtime = [
            p
            for p in payloads
            if p.get("api_name") not in ("hook_loaded", "hook_error", "java_ready", "java_timeout")
        ]
        if runtime:
            print("  hook runtime events:", len(runtime))
            return True
        errors = [m for m in msgs if m.get("type") == "error"]
        hook_loaded = [p for p in payloads if p.get("api_name") == "hook_loaded"]
        if errors:
            print("  hook errors:", errors[0].get("description", errors[0]))
            return False
        if hook_loaded and label == APP:
            print("  hook_loaded on Keep parent (OK for stage3; trigger APIs in app UI)")
            return True
        print("  hook loaded but no API fired (open a screen that reads android_id)")
        return bool(hook_loaded)
    finally:
        if session is not None:
            try:
                session.detach()
            except Exception:
                pass


def main() -> int:
    device = frida.get_usb_device(timeout=8)
    if len(sys.argv) > 1:
        pid = int(sys.argv[1])
        name = ""
        for p in device.enumerate_processes():
            if int(p.pid) == pid:
                name = getattr(p, "name", "") or ""
                break
        ok = _try_hook(device, pid, name)
        return 0 if ok else 1

    print("Keep-related processes (frida-ps):")
    for p in device.enumerate_processes():
        n = (getattr(p, "name", "") or "")
        if "gotokeep" in n.lower() or n == APP:
            print(f"  pid={p.pid} name={n}")

    targets = _ordered_attach_pids(device, PKG, APP)
    if not targets:
        print("\nNo process found. Launch Keep on the emulator first:")
        print("  adb shell monkey -p com.gotokeep.keep -c android.intent.category.LAUNCHER 1")
        return 1

    print("\nRecommended attach order:", targets)
    print(
        "\nNote: pid named com.gotokeep.keep often fails with "
        "'refused to load frida-agent' on MuMu; use the 'Keep' parent PID instead.\n"
    )

    for pid, label in targets:
        if _try_hook(device, pid, label):
            print(f"\nSUCCESS: use pid={pid} ({label}) for stage3 / hooks.")
            return 0

    print(
        "\nAll attach attempts failed. Checklist:\n"
        "  1. frida-server running as root: adb shell ps -A | grep frida\n"
        "  2. PC frida version matches server (frida --version vs device)\n"
        "  3. Keep in foreground, logged in\n"
        "  4. Restart: adb shell su -c 'killall frida-server; /data/local/tmp/frida-server &'\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
