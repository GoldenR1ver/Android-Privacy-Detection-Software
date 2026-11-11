#!/usr/bin/env python3
"""
Standalone Frida hook runner for fridaSample experiments.

Target default: 韭研公社 (com.jiucaigongshe)

Usage (from project/src):
  python fridaSample/run_hook.py
  python fridaSample/run_hook.py --mode spawn --wait-sec 30
  python fridaSample/run_hook.py --hooks hook_sdk_umeng.js hook_network_okhttp.js --use-project-hooks
  python fridaSample/run_hook.py --list-hooks
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SAMPLE_ROOT = Path(__file__).resolve().parent
_SRC = _SAMPLE_ROOT.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import frida

from privacy_audit.analyzers.dynamic.frida_runner import (
    _ordered_attach_pids,
    _resolve_attach_candidates,
    spawn_package_cold,
)
from privacy_audit.analyzers.dynamic.hook_loader import (
    is_frida_runtime_api,
    wrap_hook_script,
)

TARGET_FILE = _SAMPLE_ROOT / "target" / "jiuyangongshe.json"
SAMPLE_HOOKS_DIR = _SAMPLE_ROOT / "hooks"
PROJECT_HOOKS_DIR = (
    _SRC / "privacy_audit" / "analyzers" / "dynamic" / "hooks"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_target() -> dict[str, Any]:
    return json.loads(TARGET_FILE.read_text(encoding="utf-8"))


def _list_hook_files(use_project: bool) -> list[Path]:
    files = sorted(SAMPLE_HOOKS_DIR.glob("hook_*.js"))
    if use_project:
        seen = {p.name for p in files}
        for p in sorted(PROJECT_HOOKS_DIR.glob("hook_*.js")):
            if p.name not in seen and p.name != "hook_papi.js":
                files.append(p)
    return files


def _resolve_hook_paths(names: list[str], use_project: bool) -> list[Path]:
    resolved: list[Path] = []
    for name in names:
        candidate = SAMPLE_HOOKS_DIR / name
        if candidate.exists():
            resolved.append(candidate)
            continue
        if use_project:
            alt = PROJECT_HOOKS_DIR / name
            if alt.exists():
                resolved.append(alt)
                continue
        raise FileNotFoundError(f"Hook not found: {name}")
    return resolved


def _build_script_source(hook_paths: list[Path]) -> str:
    chunks = ['send({api_name:"script_loaded",pii_type:"frida",hook_file:"run_hook.py"});']
    for path in hook_paths:
        raw = path.read_text(encoding="utf-8")
        wrapped = wrap_hook_script(raw, hook_file=path.name)
        chunks.append(f"// ---- {path.name} ----\n{wrapped}")
    return "\n\n".join(chunks)


def _pidof(package: str) -> list[int]:
    import subprocess

    try:
        proc = subprocess.run(
            ["adb", "shell", "pidof", package],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return []
    return [int(tok) for tok in (proc.stdout or "").split() if tok.isdigit()]


def _collect_attach_targets(
    device: frida.core.Device,
    package: str,
    app_name: str,
    *,
    include_push: bool,
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

    if include_push:
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


def _run_session(
    session: frida.core.Session,
    device: frida.core.Device,
    *,
    script_source: str,
    wait_sec: int,
    package: str,
    resume_pid: int | None = None,
    resume_before_load: bool = False,
) -> list[dict[str, Any]]:
    script = session.create_script(script_source)
    events: list[dict[str, Any]] = []

    def on_message(message: dict, _data: bytes | None) -> None:
        events.append(message)
        payload = message.get("payload")
        if message.get("type") == "error":
            print("  ERR", message.get("description", message))
        elif isinstance(payload, dict):
            api = str(payload.get("api_name", ""))
            summary = str(payload.get("args_summary", ""))[:100]
            hook = str(payload.get("hook_file", ""))
            if api in ("script_loaded", "hook_loaded", "java_active", "hook_error"):
                print(f"  [{hook}] {api}: {summary}")
            elif api:
                print(f"  HIT {api}: {summary}")

    script.on("message", on_message)

    if resume_before_load and resume_pid is not None:
        device.resume(resume_pid)
        time.sleep(3.0)
        script.load()
    else:
        script.load()
        if resume_pid is not None:
            device.resume(resume_pid)

    print(f"  collecting for {wait_sec}s — operate the app to trigger APIs …")
    time.sleep(wait_sec)

    try:
        script.unload()
    except Exception:
        pass
    return events


def _summarize_events(
    events: list[dict[str, Any]],
    *,
    target: dict[str, Any],
    mode: str,
    pid: int | None,
    label: str,
    hook_files: list[str],
) -> dict[str, Any]:
    payloads = [e.get("payload") for e in events if isinstance(e.get("payload"), dict)]
    runtime = [
        p
        for p in payloads
        if isinstance(p, dict) and is_frida_runtime_api(str(p.get("api_name", "")))
    ]
    return {
        "schema_version": 1,
        "runner": "fridaSample.run_hook",
        "app_name": target.get("app_name"),
        "package": target.get("package"),
        "mode": mode,
        "pid": pid,
        "label": label,
        "hook_files": hook_files,
        "captured_at_utc": _utc_now(),
        "event_count": len(payloads),
        "runtime_event_count": len(runtime),
        "events": [
            {
                "timestamp_utc": _utc_now(),
                **{k: p.get(k) for k in ("pii_type", "hook_file", "api_name", "args_summary", "stack_summary")},
            }
            for p in payloads
            if isinstance(p, dict)
        ],
        "runtime_hits": [
            {
                "api_name": p.get("api_name"),
                "pii_type": p.get("pii_type"),
                "hook_file": p.get("hook_file"),
                "args_summary": p.get("args_summary"),
            }
            for p in runtime
        ],
    }


def _write_output(record: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = str(record.get("package", "app")).replace(".", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{slug}_{ts}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    target = _load_target()
    package = str(target["package"])
    app_name = str(target["app_name"])
    default_hooks = list(target.get("default_hooks") or [])

    parser = argparse.ArgumentParser(description="fridaSample hook runner (韭研公社)")
    parser.add_argument(
        "--mode",
        choices=("attach", "spawn", "both"),
        default="attach",
        help="attach=运行中进程; spawn=冷启动; both=先 spawn 再 attach",
    )
    parser.add_argument(
        "--hooks",
        nargs="*",
        default=default_hooks,
        help="Hook 脚本文件名（默认读取 target/jiuyangongshe.json default_hooks）",
    )
    parser.add_argument(
        "--use-project-hooks",
        action="store_true",
        help="同时允许从 privacy_audit/.../hooks 加载脚本",
    )
    parser.add_argument(
        "--with-project-pii",
        action="store_true",
        help="在 default hooks 之外追加 project_hooks（android_id / okhttp 等）",
    )
    parser.add_argument("--wait-sec", type=int, default=25, help="注入后采集秒数")
    parser.add_argument(
        "--output-dir",
        default="output/frida_sample",
        help="事件 JSON 输出目录",
    )
    parser.add_argument("--list-hooks", action="store_true", help="列出可用 hook 脚本")
    parser.add_argument(
        "--spawn-order",
        choices=("paused-load", "resume-load"),
        default="paused-load",
        help="spawn 时先 load 再 resume，或先 resume 再 load",
    )
    args = parser.parse_args()

    if args.list_hooks:
        print("Sample hooks:")
        for p in sorted(SAMPLE_HOOKS_DIR.glob("hook_*.js")):
            print(f"  {p.name}")
        print("\nProject hooks (privacy_audit):")
        for p in sorted(PROJECT_HOOKS_DIR.glob("hook_*.js")):
            print(f"  {p.name}")
        return 0

    hook_names = list(args.hooks)
    if args.with_project_pii:
        args.use_project_hooks = True
        for name in target.get("project_hooks") or []:
            if name not in hook_names:
                hook_names.append(name)

    hook_paths = _resolve_hook_paths(hook_names, args.use_project_hooks)
    script_source = _build_script_source(hook_paths)
    output_dir = Path(args.output_dir)

    print(f"Target: {app_name} ({package})")
    print(f"Hooks: {', '.join(p.name for p in hook_paths)}")

    device = frida.get_usb_device(timeout=8)

    def try_spawn() -> tuple[bool, list[dict[str, Any]], int | None, str]:
        session = None
        try:
            pid = spawn_package_cold(device, package)
            session = device.attach(pid)
            resume_before = args.spawn_order == "resume-load"
            print(f"\n=== spawn pid={pid} order={args.spawn_order} ===")
            events = _run_session(
                session,
                device,
                script_source=script_source,
                wait_sec=args.wait_sec,
                package=package,
                resume_pid=pid,
                resume_before_load=resume_before,
            )
            runtime = sum(
                1
                for e in events
                if isinstance(e.get("payload"), dict)
                and is_frida_runtime_api(str(e["payload"].get("api_name", "")))
            )
            ok = runtime > 0 or any(
                isinstance(e.get("payload"), dict)
                and e["payload"].get("api_name") in ("java_bridge_ok", "java_active", "hook_loaded")
                for e in events
            )
            return ok, events, pid, args.spawn_order
        except Exception as exc:
            print(f"  spawn failed: {exc}")
            return False, [], None, "spawn_failed"
        finally:
            if session is not None:
                try:
                    session.detach()
                except Exception:
                    pass

    if args.mode in {"spawn", "both"}:
        ok, events, pid, label = try_spawn()
        if ok or args.mode == "spawn":
            record = _summarize_events(
                events,
                target=target,
                mode="spawn",
                pid=pid,
                label=label,
                hook_files=[p.name for p in hook_paths],
            )
            out = _write_output(record, output_dir)
            print(f"\nSaved: {out}")
            print(f"Runtime hits: {record['runtime_event_count']}")
            return 0 if ok else 1
        print("\nspawn 未成功，fallback 到 attach …")

    include_push = bool(target.get("include_push_subprocess", True))
    targets = _collect_attach_targets(
        device, package, app_name, include_push=include_push
    )
    if not targets:
        print(f"\n未找到进程。请先启动 {app_name}:")
        print(f"  adb shell monkey -p {package} -c android.intent.category.LAUNCHER 1")
        return 1

    print("\nAttach candidates:")
    for pid, label in targets:
        print(f"  pid={pid} label={label!r}")

    for pid, label in targets:
        session = None
        try:
            print(f"\n=== attach pid={pid} label={label!r} ===")
            session = device.attach(pid, persist_timeout=30)
            events = _run_session(
                session,
                device,
                script_source=script_source,
                wait_sec=args.wait_sec,
                package=package,
            )
            record = _summarize_events(
                events,
                target=target,
                mode="attach",
                pid=pid,
                label=label,
                hook_files=[p.name for p in hook_paths],
            )
            out = _write_output(record, output_dir)
            print(f"\nSaved: {out}")
            print(f"Runtime hits: {record['runtime_event_count']}")
            if record["runtime_event_count"] > 0:
                return 0
        except Exception as exc:
            print(f"  attach failed: {exc}")
        finally:
            if session is not None:
                try:
                    session.detach()
                except Exception:
                    pass

    print(
        "\n全部 attach 尝试均未捕获 runtime 事件。\n"
        "建议：\n"
        f"  python fridaSample/run_hook.py --mode spawn --wait-sec 30\n"
        f"  python scripts/probe_hook.py --profile jiuyan\n"
        "  确认 frida-server 已 root 启动且 PC 端 frida 版本一致"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
