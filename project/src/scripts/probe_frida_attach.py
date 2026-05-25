"""Probe which Keep PID accepts Frida + Java.perform."""
import time

import frida

d = frida.get_usb_device(timeout=8)
targets: list[tuple[int, str]] = []
for p in d.enumerate_processes():
    n = (getattr(p, "name", "") or "")
    if "gotokeep" in n.lower() or n == "Keep":
        targets.append((int(p.pid), n))
print("targets:", targets)
script_src = """
Java.perform(function () {
  send({ api_name: "ping", pii_type: "probe" });
});
"""
print("--- spawn test ---")
try:
    pid = d.spawn(["com.gotokeep.keep"])
    print("spawned pid", pid)
    session = d.attach(pid)
    script = session.create_script(script_src)
    msgs = []
    script.on("message", lambda m, _d: msgs.append(m))
    script.load()
    d.resume(pid)
    time.sleep(3)
    script.unload()
    session.detach()
    print("spawn messages:", msgs[:5])
except Exception as exc:
    print("spawn FAIL", exc)

for pid, name in targets:
    try:
        session = d.attach(pid)
        print("OK attach", pid, name)
        script = session.create_script(script_src)
        msgs: list = []

        def on_message(message, _data):
            msgs.append(message)

        script.on("message", on_message)
        script.load()
        time.sleep(1.5)
        script.unload()
        session.detach()
        print("  messages:", msgs[:3])
    except Exception as exc:
        print("FAIL attach", pid, name, exc)
