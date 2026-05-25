import time
from pathlib import Path

import frida

hook = (
    Path(__file__).resolve().parents[1]
    / "privacy_audit/analyzers/dynamic/hooks/hook_android_id.js"
).read_text(encoding="utf-8")
pkg = "com.gotokeep.keep"
d = frida.get_usb_device(timeout=8)
pid = d.spawn([pkg])
print("spawned", pid)
d.resume(pid)
for wait in (3, 6, 10):
    time.sleep(wait if wait == 3 else wait - 3)
    print(f"--- after ~{wait}s ---")
    for p in d.enumerate_processes():
        n = getattr(p, "name", "") or ""
        if "gotokeep" in n.lower() or n == "Keep":
            print(" proc", p.pid, n)
    target_pid = None
    for p in d.enumerate_processes():
        if (getattr(p, "name", "") or "") == pkg:
            target_pid = int(p.pid)
            break
    if not target_pid:
        continue
    try:
        session = d.attach(target_pid)
        msgs = []

        def on_message(message, _data):
            msgs.append(message)
            print("MSG", message)

        script = session.create_script(hook)
        script.on("message", on_message)
        script.load()
        time.sleep(4)
        session.detach()
        print("attach", target_pid, "msgs", len(msgs))
        break
    except Exception as exc:
        print("attach fail", target_pid, exc)
