"""
Diagnose Frida attach/spawn for MuMu / multi-process apps.

Matches stage3 FridaCaptureSession: attach *and* cold spawn (frida -U -f).

Usage (from project/src):
  python scripts/probe_hook.py --profile diaoyu --mode both
  python scripts/probe_hook.py --package com.lchr.diaoyu --app 钓鱼人 --mode spawn
  python scripts/probe_hook.py --profile jiuyan --skip-java-probe
  python scripts/probe_hook.py --profile keep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import frida

from privacy_audit.analyzers.dynamic.hook_loader import wrap_hook_script
from privacy_audit.analyzers.dynamic.frida_runner import (
    _accept_java_probe_state,
    _ordered_attach_pids,
    _resolve_attach_candidates,
    _verify_java_session,
    load_app_frida_config,
    resolve_code_prefixes,
    spawn_package_cold,
)
from privacy_audit.ingestion.evidence_loader import _slug_for_app

HOOK = (
    Path(__file__).resolve().parents[1]
    / "privacy_audit/analyzers/dynamic/hooks/hook_android_id.js"
).read_text(encoding="utf-8")

SCRIPT_PING = 'send({api_name:"script_loaded",pii_type:"frida",hook_file:"probe"});'

# Poll Java.available — reports java_bridge_ok or java_never_ready (not silent failure).
JAVA_WAIT_BASELINE = r"""
send({ api_name: "script_loaded", pii_type: "frida", hook_file: "java_wait" });
(function () {
  var attempts = 0;
  function tick() {
    attempts += 1;
    if (typeof Java !== "undefined" && Java.available) {
      Java.perform(function () {
        send({
          api_name: "java_bridge_ok",
          pii_type: "frida",
          hook_file: "java_wait",
          args_summary: "attempts=" + attempts
        });
      });
      return;
    }
    if (attempts < 120) {
      setTimeout(tick, 250);
    } else {
      send({
        api_name: "java_never_ready",
        pii_type: "frida",
        hook_file: "java_wait",
        args_summary: "attempts=" + attempts
      });
    }
  }
  tick();
})();
"""

PROFILES: dict[str, dict[str, Any]] = {
    "jiuyan": {
        "package": "com.jiucaigongshe",
        "app_name": "韭研公社",
        "skip_java_probe_on_display": False,
        "accept_java_timeout_on_display": True,
        "include_push_subprocess": True,
    },
    "diaoyu": {
        "package": "com.lchr.diaoyu",
        "app_name": "钓鱼人",
        "skip_java_probe_on_display": False,
        "accept_java_timeout_on_display": True,
        "include_push_subprocess": False,
    },
    "keep": {
        "package": "com.gotokeep.keep",
        "app_name": "Keep",
        "skip_java_probe_on_display": True,
        "accept_java_timeout_on_display": True,
        "include_push_subprocess": False,
    },
}


def _pidof(package: str) -> list[int]:
    try:
        proc = subprocess.run(
            ["adb", "shell", "pidof", package],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        print(f"  pidof error: {exc}")
        return []
    return [int(tok) for tok in (proc.stdout or "").split() if tok.isdigit()]


def _related_match_keys(package: str, app_name: str) -> set[str]:
    keys = {package.lower(), app_name.lower()}
    if "jiucai" in package.lower():
        keys.add("pushcore")
    if "gotokeep" in package.lower():
        keys.add("gotokeep")
    return keys


def _process_matches(keys: set[str], name: str, ident: str) -> bool:
    blob = f"{name} {ident}".lower()
    return any(k and k in blob for k in keys)


def _list_related_processes(device: frida.core.Device, package: str, app_name: str) -> None:
    keys = _related_match_keys(package, app_name)
    print(f"\nfrida-ps (related to {package!r} / {app_name!r}):")
    for proc in device.enumerate_processes():
        name = (getattr(proc, "name", "") or "").strip()
        ident = (getattr(proc, "identifier", "") or "").strip()
        if _process_matches(keys, name, ident):
            print(f"  pid={proc.pid} name={name!r} identifier={ident!r}")


def _collect_targets(
    device: frida.core.Device,
    package: str,
    app_name: str,
    *,
    include_push_subprocess: bool,
) -> list[tuple[int, str]]:
    seen: set[int] = set()
    ordered: list[tuple[int, str]] = []

    def add(pid: int, label: str) -> None:
        if pid and pid not in seen:
            ordered.append((pid, label))
            seen.add(pid)

    for pid in _pidof(package):
        add(pid, f"pidof:{pid}")

    for pid, label in _ordered_attach_pids(device, package, app_name):
        add(pid, label)

    for _score, pid, label in _resolve_attach_candidates(device, package, app_name):
        add(pid, label)

    if include_push_subprocess:
        for proc in device.enumerate_processes():
            name = (getattr(proc, "name", "") or "").strip()
            ident = (getattr(proc, "identifier", "") or "").strip()
            pid = int(getattr(proc, "pid", 0) or 0)
            if not pid:
                continue
            if package in name or package in ident:
                if ":push" in name.lower() or "pushcore" in name.lower():
                    add(pid, name or ident)

    return ordered


def _resolve_accept_java_timeout_on_display(
    app_name: str,
    profile_default: bool,
) -> bool:
    packages_file = _SRC / "input" / "app_packages.json"
    try:
        _pkg, _pfx, opts = load_app_frida_config(app_name, packages_file)
        if "frida_accept_java_timeout_on_display" in opts:
            return bool(opts["frida_accept_java_timeout_on_display"])
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return profile_default


def _hook_source(*, raw_baseline: bool, java_wait_only: bool) -> str:
    if java_wait_only or raw_baseline:
        return JAVA_WAIT_BASELINE
    return SCRIPT_PING + "\n" + wrap_hook_script(HOOK)


def _print_hook_diagnostics(msgs: list[dict[str, Any]]) -> None:
    payloads = [m.get("payload") for m in msgs if isinstance(m.get("payload"), dict)]
    apis = [str(p.get("api_name", "")) for p in payloads if p.get("api_name")]
    errors = [m for m in msgs if m.get("type") == "error"]
    print(f"  diagnostics: payloads={len(payloads)} api_names={apis or '(none)'}")
    if errors:
        print(f"  diagnostics: script_errors={len(errors)}")
    if not payloads and not errors:
        print(
            "  diagnostics: zero messages — Java bridge may not have started "
            "(try --spawn-order resume-load or --raw-baseline)"
        )


_NON_RUNTIME_APIS = frozenset(
    {
        "hook_loaded",
        "hook_error",
        "java_ready",
        "java_timeout",
        "java_active",
        "papi_ready",
        "raw_java_ok",
        "raw_class_ok",
        "script_loaded",
        "java_bridge_ok",
        "java_never_ready",
    }
)


def _probe_summary_from_msgs(
    msgs: list[dict[str, Any]],
    *,
    app_name: str,
    package: str,
    mode: str,
    pid: int | None,
    label: str,
    success: bool,
) -> dict[str, Any]:
    payloads = [m.get("payload") for m in msgs if isinstance(m.get("payload"), dict)]
    runtime = [
        p
        for p in payloads
        if str(p.get("api_name", "")) and str(p.get("api_name", "")) not in _NON_RUNTIME_APIS
    ]
    apis = [str(p.get("api_name", "")) for p in payloads if p.get("api_name")]

    java_bridge = "unknown"
    if any(p.get("api_name") == "java_bridge_ok" for p in payloads):
        java_bridge = "ok"
    elif any(p.get("api_name") == "java_never_ready" for p in payloads):
        java_bridge = "never_ready"
    elif any(p.get("api_name") == "java_timeout" for p in payloads):
        java_bridge = "timeout"
    elif any(p.get("api_name") == "java_ready" for p in payloads):
        java_bridge = "ok"
    elif any(p.get("api_name") == "java_active" for p in payloads):
        java_bridge = "partial"

    return {
        "schema_version": 1,
        "app_name": app_name,
        "package": package,
        "slug": _slug_for_app(app_name),
        "probe_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "pid": pid,
        "label": label,
        "java_bridge": java_bridge,
        "success": success,
        "runtime_event_count": len(runtime),
        "api_names": apis[:40],
        "preferred_dynamic_tracks": (
            ["static_tracker", "mitm", "ui"]
            if java_bridge == "never_ready"
            else ["frida_java", "frida_network", "ui"]
        ),
    }


def _write_probe_record(record: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = str(record.get("slug") or record.get("app_name", "app"))
    path = output_dir / f"{slug}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _update_frida_tier_java_bridge(
    app_name: str,
    java_bridge: str,
    tier_file: Path,
) -> None:
    if not tier_file.exists() or java_bridge == "unknown":
        return
    data = json.loads(tier_file.read_text(encoding="utf-8"))
    apps = data.setdefault("apps", {})
    entry = dict(apps.get(app_name) or {})
    entry["java_bridge"] = java_bridge
    if java_bridge == "never_ready":
        entry.setdefault("tier", "network_static_only")
        entry["enable_frida_consent"] = False
        entry["preferred_dynamic_tracks"] = ["static_tracker", "mitm", "ui"]
    elif java_bridge == "ok":
        entry.setdefault("preferred_dynamic_tracks", ["frida_java", "frida_network", "ui"])
    apps[app_name] = entry
    tier_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _evaluate_hook_result(
    msgs: list[dict[str, Any]],
    *,
    skip_probe: bool,
) -> bool:
    payloads = [m.get("payload") for m in msgs if isinstance(m.get("payload"), dict)]
    runtime = [
        p
        for p in payloads
        if str(p.get("api_name", "")) and str(p.get("api_name", "")) not in _NON_RUNTIME_APIS
    ]
    java_active = [p for p in payloads if p.get("api_name") == "java_active"]
    errors = [m for m in msgs if m.get("type") == "error"]
    hook_loaded = [p for p in payloads if p.get("api_name") == "hook_loaded"]
    java_bridge = [p for p in payloads if p.get("api_name") == "java_bridge_ok"]
    java_never = [p for p in payloads if p.get("api_name") == "java_never_ready"]
    script_loaded = [p for p in payloads if p.get("api_name") == "script_loaded"]

    if runtime:
        print(f"  SUCCESS runtime_events={len(runtime)}")
        for p in runtime[:3]:
            print(f"    - {p.get('api_name')} {p.get('args_summary', '')[:80]}")
        return True
    if java_bridge:
        print(
            f"  SUCCESS Java bridge OK "
            f"({java_bridge[0].get('args_summary', '')})"
        )
        return True
    if java_never:
        print(
            f"  FAIL Java bridge never ready "
            f"({java_never[0].get('args_summary', '')})"
        )
        _print_hook_diagnostics(msgs)
        return False
    if script_loaded and not java_active and not hook_loaded:
        print("  script_loaded OK but no Java/hook events — bridge or timing issue")
    if errors:
        err_text = str(errors[0].get("description", errors[0]))
        print("  hook error:", err_text)
        return False
    if java_active:
        print("  java_active seen but no android_id API yet (partial success)")
        return True
    if hook_loaded and skip_probe:
        print("  hook_loaded on display parent only (OK for Keep-style MuMu parent)")
        return True
    if hook_loaded:
        print("  hook_loaded but no runtime API (try more UI in app)")
        _print_hook_diagnostics(msgs)
        return False
    print("  no hook_loaded / no runtime")
    _print_hook_diagnostics(msgs)
    return False


def _process_alive(package: str, pid: int) -> bool:
    return pid in _pidof(package)


def _run_hook_on_session(
    session: frida.core.Session,
    device: frida.core.Device,
    *,
    hook_wait_sec: int,
    raw_baseline: bool,
    java_wait_only: bool,
    skip_probe: bool,
    resume: int | None = None,
    resume_before_load: bool = False,
    post_resume_boot_sec: float = 3.0,
    package: str = "",
) -> tuple[bool, list[dict[str, Any]]]:
    script = session.create_script(
        _hook_source(raw_baseline=raw_baseline, java_wait_only=java_wait_only)
    )
    msgs: list[dict[str, Any]] = []

    def on_message(message: dict, _data: bytes | None) -> None:
        msgs.append(message)
        payload = message.get("payload")
        if message.get("type") == "error":
            print("  ERR", message.get("description", message))
        elif isinstance(payload, dict):
            api = str(payload.get("api_name", ""))
            if api in (
                "script_loaded",
                "java_active",
                "hook_loaded",
                "hook_error",
                "java_bridge_ok",
                "java_never_ready",
            ):
                print(f"  DIAG {api}: {str(payload.get('args_summary', ''))[:80]}")
            elif api and api not in ("",):
                print("  MSG", payload)

    script.on("message", on_message)
    if resume_before_load and resume is not None:
        print(f"  resume pid={resume} before script load (frida --no-pause style)")
        device.resume(resume)
        if post_resume_boot_sec > 0:
            print(f"  art_boot_wait: {post_resume_boot_sec:.0f}s")
            time.sleep(post_resume_boot_sec)
        script.load()
    else:
        script.load()
        if resume is not None:
            print(f"  resume pid={resume} after script load (stage3 spawn order)")
            device.resume(resume)
    print(f"  hook_wait: {hook_wait_sec}s (browse app / pull to refresh)")
    time.sleep(hook_wait_sec)
    if resume is not None and package:
        alive = _process_alive(package, resume)
        print(f"  process_alive after wait: pid={resume} -> {alive}")
    try:
        script.unload()
    except Exception:
        pass
    ok = _evaluate_hook_result(msgs, skip_probe=skip_probe)
    return ok, msgs


def _maybe_run_java_probe(
    session: frida.core.Session,
    *,
    package: str,
    app_name: str,
    label: str,
    pid: int,
    skip_java_probe: bool,
    skip_java_probe_on_display: bool,
    accept_java_timeout_on_display: bool,
) -> bool:
    """Return True if hook install should proceed."""
    if skip_java_probe:
        print("  java_probe: skipped (--skip-java-probe / classmates-style)")
        return True
    skip_probe = (
        skip_java_probe_on_display
        and label == app_name
        and label != package
    )
    if skip_probe:
        print("  java_probe: skipped (display-name parent profile)")
        return True

    prefixes = resolve_code_prefixes(package)
    state = _verify_java_session(session, prefixes)
    accepted = _accept_java_probe_state(
        state,
        label,
        app_name,
        package,
        accept_java_timeout_on_display=accept_java_timeout_on_display,
    )
    print(
        f"  java_probe: {state} accepted={accepted} "
        f"accept_java_timeout_on_display={accept_java_timeout_on_display} "
        f"prefixes={prefixes[:4]}"
    )
    if accepted:
        return True
    if state == "java_timeout":
        main_pids = _pidof(package)
        if main_pids and pid == main_pids[0]:
            print(
                "  java_probe: inconclusive on main PID — proceeding to hook install "
                "(same as dynamic capture attach fallback)"
            )
            return True
    if state in {"not_ready", "java_timeout"}:
        print("  java_probe: FAIL — probe rejected before hook install")
        return False
    return True


def _try_attach_hook(
    device: frida.core.Device,
    pid: int,
    label: str,
    *,
    package: str,
    app_name: str,
    skip_java_probe_on_display: bool,
    accept_java_timeout_on_display: bool,
    hook_wait_sec: int,
    skip_java_probe: bool,
    raw_baseline: bool,
    java_wait_only: bool = False,
    probe_mode: str = "attach",
    probe_output_dir: Path | None = None,
    frida_tier_file: Path | None = None,
    update_tier: bool = True,
) -> bool:
    print(f"\n--- attach pid={pid} label={label!r} ---")
    session = None
    try:
        session = device.attach(pid, persist_timeout=30)
        print("  attach: OK")
    except Exception as exc:
        print(f"  attach: FAIL ({exc})")
        return False
    try:
        skip_probe = (
            skip_java_probe_on_display
            and label == app_name
            and label != package
        )
        if not _maybe_run_java_probe(
            session,
            package=package,
            app_name=app_name,
            label=label,
            pid=pid,
            skip_java_probe=skip_java_probe,
            skip_java_probe_on_display=skip_java_probe_on_display,
            accept_java_timeout_on_display=accept_java_timeout_on_display,
        ):
            return False
        ok, msgs = _run_hook_on_session(
            session,
            device,
            hook_wait_sec=hook_wait_sec,
            raw_baseline=raw_baseline,
            java_wait_only=java_wait_only,
            skip_probe=skip_probe,
            package=package,
        )
        if probe_output_dir is not None:
            record = _probe_summary_from_msgs(
                msgs,
                app_name=app_name,
                package=package,
                mode=probe_mode,
                pid=pid,
                label=label,
                success=ok,
            )
            out_path = _write_probe_record(record, probe_output_dir)
            print(f"  probe_record: {out_path} java_bridge={record['java_bridge']}")
            if update_tier and frida_tier_file is not None:
                _update_frida_tier_java_bridge(app_name, record["java_bridge"], frida_tier_file)
        return ok
    finally:
        if session is not None:
            try:
                session.detach()
            except Exception:
                pass


def _spawn_session_once(
    device: frida.core.Device,
    package: str,
) -> tuple[frida.core.Session, int]:
    pid = spawn_package_cold(device, package)
    session = device.attach(pid)
    return session, pid


def _try_spawn_hook(
    device: frida.core.Device,
    *,
    package: str,
    hook_wait_sec: int,
    raw_baseline: bool,
    spawn_order: str,
    probe_output_dir: Path | None = None,
    app_name: str = "",
    frida_tier_file: Path | None = None,
    update_tier: bool = True,
) -> bool:
    """Cold spawn hook probe. ``paused-load`` = dynamic capture order; ``resume-load`` = frida --no-pause."""
    orders = {
        "paused-load": ("load hooks while paused, then resume", False),
        "resume-load": ("resume first, then load hooks", True),
    }
    if spawn_order == "auto":
        sequence = ("paused-load", "resume-load")
    else:
        sequence = (spawn_order,)

    print(f"\n=== spawn mode (frida -U -f {package}) ===")
    for idx, order in enumerate(sequence):
        label, resume_before_load = orders[order]
        session = None
        try:
            if idx > 0:
                print(f"\n  --- retry spawn order: {order} ({label}) ---")
            session, pid = _spawn_session_once(device, package)
            print(f"  spawn+attach: OK pid={pid} order={order}")
            print("  java_probe: skipped (spawn path installs hooks, not prefix probe)")
            ok, msgs = _run_hook_on_session(
                session,
                device,
                hook_wait_sec=hook_wait_sec,
                raw_baseline=raw_baseline,
                java_wait_only=False,
                skip_probe=False,
                resume=pid,
                resume_before_load=resume_before_load,
                package=package,
            )
            if probe_output_dir is not None and app_name:
                record = _probe_summary_from_msgs(
                    msgs,
                    app_name=app_name,
                    package=package,
                    mode="spawn",
                    pid=pid,
                    label=order,
                    success=ok,
                )
                out_path = _write_probe_record(record, probe_output_dir)
                print(f"  probe_record: {out_path} java_bridge={record['java_bridge']}")
                if update_tier and frida_tier_file is not None:
                    _update_frida_tier_java_bridge(app_name, record["java_bridge"], frida_tier_file)
            if ok:
                print(f"\n>>> USE spawn + order={order!r} for capture-dynamic <<<")
                return True
        except Exception as exc:
            print(f"  spawn: FAIL ({exc})")
        finally:
            if session is not None:
                try:
                    session.detach()
                except Exception:
                    pass
    return False


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Frida attach/spawn + Java/hook probe")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="jiuyan",
        help="Built-in app profile (default: jiuyan)",
    )
    parser.add_argument("--package", default="", help="Override Android package")
    parser.add_argument("--app", default="", help="Override display app name")
    parser.add_argument("--pid", type=int, default=0, help="Probe a single PID only (attach)")
    parser.add_argument(
        "--mode",
        choices=("attach", "spawn", "both"),
        default="both",
        help="attach=running process; spawn=cold start (frida -U -f); both=spawn then attach",
    )
    parser.add_argument(
        "--skip-java-probe",
        action="store_true",
        help="Skip Java prefix probe; load hooks directly (classmates / frida -l style)",
    )
    parser.add_argument(
        "--raw-baseline",
        action="store_true",
        help="Use Java.available wait baseline instead of wrapped android_id hook",
    )
    parser.add_argument(
        "--java-wait-only",
        action="store_true",
        help="Only test script ping + Java.available polling (no android_id hook)",
    )
    parser.add_argument(
        "--spawn-order",
        choices=("auto", "paused-load", "resume-load"),
        default="auto",
        help="spawn: load-before-resume (stage3) or resume-before-load (frida --no-pause)",
    )
    parser.add_argument("--hook-wait-sec", type=int, default=25, help="Seconds after hook load")
    parser.add_argument(
        "--probe-output-dir",
        default="output/frida_probe",
        help="Write probe summary JSON (R1.5), default output/frida_probe/<slug>.json",
    )
    parser.add_argument(
        "--no-update-tier",
        action="store_true",
        help="Do not patch input/frida_tier.json java_bridge from probe result",
    )
    parser.add_argument(
        "--frida-tier-file",
        default="input/frida_tier.json",
        help="Frida tier config to update with java_bridge (default input/frida_tier.json)",
    )
    args = parser.parse_args()

    probe_output_dir = Path(args.probe_output_dir)
    frida_tier_file = Path(args.frida_tier_file)
    update_tier = not args.no_update_tier

    prof = PROFILES[args.profile]
    package = (args.package or prof["package"]).strip()
    app_name = (args.app or prof["app_name"]).strip()
    profile_matches = package == prof["package"] and app_name == prof["app_name"]
    profile_default_accept = bool(prof.get("accept_java_timeout_on_display", True))
    skip_display = bool(prof.get("skip_java_probe_on_display", False))
    include_push = bool(prof.get("include_push_subprocess", False))
    accept_java_timeout = _resolve_accept_java_timeout_on_display(
        app_name,
        profile_default_accept if profile_matches else True,
    )

    device = frida.get_usb_device(timeout=8)
    print(
        f"Probe profile={args.profile} mode={args.mode} package={package} app={app_name}"
    )

    _list_related_processes(device, package, app_name)
    pids = _pidof(package)
    print(f"\nadb pidof {package}: {pids or '(none)'}")

    if args.pid:
        live = {int(proc.pid) for proc in device.enumerate_processes()}
        if args.pid not in live:
            print(f"\nPID {args.pid} is not running (app likely restarted since last pidof).")
            if pids:
                print(
                    f"  Current main process: pid={pids[0]}\n"
                    f"  Retry: python scripts/probe_hook.py --pid {pids[0]} --hook-wait-sec {args.hook_wait_sec}"
                )
            else:
                print(f"  Launch {app_name} first, or run without --pid for auto-discovery.")
            return 1
        name = ""
        for proc in device.enumerate_processes():
            if int(proc.pid) == args.pid:
                name = (getattr(proc, "name", "") or "").strip()
                break
        ok = _try_attach_hook(
            device,
            args.pid,
            name or str(args.pid),
            package=package,
            app_name=app_name,
            skip_java_probe_on_display=skip_display,
            accept_java_timeout_on_display=accept_java_timeout,
            hook_wait_sec=args.hook_wait_sec,
            skip_java_probe=args.skip_java_probe,
            raw_baseline=args.raw_baseline or args.java_wait_only,
            java_wait_only=args.java_wait_only,
            probe_output_dir=probe_output_dir,
            frida_tier_file=frida_tier_file,
            update_tier=update_tier,
        )
        return 0 if ok else 1

    if args.mode in {"spawn", "both"}:
        if _try_spawn_hook(
            device,
            package=package,
            hook_wait_sec=args.hook_wait_sec,
            raw_baseline=args.raw_baseline or args.java_wait_only,
            spawn_order=args.spawn_order,
            probe_output_dir=probe_output_dir,
            app_name=app_name,
            frida_tier_file=frida_tier_file,
            update_tier=update_tier,
        ):
            return 0
        print("\n=== spawn failed; fallback to attach on running process ===")

    targets = _collect_targets(
        device, package, app_name, include_push_subprocess=include_push
    )
    if not targets:
        print(f"\nNo process found. Launch {app_name} first:")
        print(f"  adb shell monkey -p {package} -c android.intent.category.LAUNCHER 1")
        return 1

    print("\nAttach probe order:")
    for pid, label in targets:
        print(f"  pid={pid} label={label!r}")

    for pid, label in targets:
        if _try_attach_hook(
            device,
            pid,
            label,
            package=package,
            app_name=app_name,
            skip_java_probe_on_display=skip_display,
            accept_java_timeout_on_display=accept_java_timeout,
            hook_wait_sec=args.hook_wait_sec,
            skip_java_probe=args.skip_java_probe or args.mode == "spawn",
            raw_baseline=args.raw_baseline or args.java_wait_only,
            java_wait_only=args.java_wait_only,
            probe_output_dir=probe_output_dir,
            frida_tier_file=frida_tier_file,
            update_tier=update_tier,
        ):
            print(f"\n>>> USE pid={pid} ({label!r}) for capture-dynamic --frida-mode attach <<<")
            return 0

    main_pid = pids[0] if pids else targets[0][0]
    print(
        "\nAll probe attempts failed.\n"
        "  Framework note: attach-only + strict java_probe often false-negatives apps that\n"
        "  classmates hook via cold spawn (frida -U -f) or skip the prefix probe.\n"
        "\nRetry (closest to classmates):\n"
        f"  python scripts/probe_hook.py --package {package} --app {app_name} "
        f"--mode spawn --hook-wait-sec 30\n"
        f"  python scripts/probe_hook.py --package {package} --app {app_name} "
        f"--skip-java-probe --pid {main_pid}\n"
        "\nDynamic capture (tier default spawn):\n"
        f"  python -m privacy_audit.cli -v capture-dynamic --app-name {app_name} "
        f"--app-package {package} --real-run --enable-frida --frida-mode spawn ...\n"
        "\nEnvironment:\n"
        "  1. frida-server running as root\n"
        "  2. PC frida version matches device (frida --version)\n"
        f"  3. {app_name} logged in; compare probe --profile keep on same device\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

