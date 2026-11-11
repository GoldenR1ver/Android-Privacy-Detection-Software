#!/usr/bin/env python3
"""
Consent audit UI experiment driver (before / after / compare / all).

Consent audit uses mitm-primary capture (--dynamic-track consent): UI + mitmproxy network
evidence for before/after switch comparison. Frida is supplementary only when
--enable-frida is passed explicitly.

Workflow:
  1) phase=before  — App in full/consent-ON state, run Dynamic capture for each app
  2) MANUAL        — In emulator, disable switch (基本版 / 隐私撤回 / etc.) per app policy
  3) phase=after   — Same Dynamic capture command into consent_after/
  4) phase=compare — UI before/after JSON diff -> output/consent_ui/

Usage (from project/src):
  python scripts/run_consent_ui_batch.py --phase before
  # ... manual toggle on device ...
  python scripts/run_consent_ui_batch.py --phase after
  python scripts/run_consent_ui_batch.py --phase compare

  python scripts/run_consent_ui_batch.py --phase all --app 小荷健康
    # single-app: before -> pause -> after -> compare

  python scripts/run_consent_ui_batch.py --phase before --app 韭研公社 --frida-mode spawn -v
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_APPS = _SRC / "input" / "apps_full_batch_15.txt"
PACKAGES = _SRC / "input" / "app_packages.json"
TIER_FILE = _SRC / "input" / "frida_tier.json"

DEFAULT_FRIDA_GRACE_SEC = 30
DEFAULT_FRIDA_TAIL_SEC = 60


def _load_apps(path: Path, only: str) -> list[str]:
    if only.strip():
        return [a.strip() for a in only.split(",") if a.strip()]
    apps: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            apps.append(line)
    return apps


def _load_package(app: str) -> str:
    from privacy_audit.analyzers.dynamic.frida_runner import parse_app_package_entry

    data = json.loads(PACKAGES.read_text(encoding="utf-8"))
    if app not in data:
        raise KeyError(f"{app} not in {PACKAGES}")
    pkg, _, _opts = parse_app_package_entry(data[app])
    return pkg


def _adb_force_stop(package: str, *, adb_serial: str | None = None) -> None:
    from privacy_audit.utils.adb_device import adb_run

    adb_run(["shell", "am", "force-stop", package], serial=adb_serial)


def _prepare_app_foreground(
    app: str,
    package: str,
    *,
    frida_mode: str,
    adb_serial: str | None = None,
) -> None:
    """Reduce batch drift: stop target app, relaunch, brief settle.

    Skipped in spawn mode — Dynamic capture cold-spawns the package instead.
    """
    if frida_mode == "spawn":
        _adb_force_stop(package, adb_serial=adb_serial)
        time.sleep(0.5)
        return
    _adb_force_stop(package, adb_serial=adb_serial)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from privacy_audit.analyzers.dynamic.app_launch import adb_launch_package;"
            f"adb_launch_package({package!r})",
        ],
        cwd=_SRC,
        check=False,
    )
    time.sleep(2.0)


def _run_dynamic_capture(
    app: str,
    phase: str,
    *,
    verbose: bool,
    enable_frida: bool | None,
    disable_frida: bool,
    frida_mode: str,
    grace_sec: int,
    tail_sec: int,
    tier_file: Path,
    mitm_adb_proxy_host: str,
    adb_serial: str | None,
) -> None:
    from privacy_audit.analyzers.dynamic.frida_runner import (
        resolve_frida_capture_seconds,
        resolve_frida_launch_mode,
        should_enable_frida_for_consent,
    )

    pkg = _load_package(app)
    use_frida = should_enable_frida_for_consent(
        app,
        tier_file,
        cli_enable=enable_frida,
        cli_disable=disable_frida,
    )
    launch_mode = resolve_frida_launch_mode(app, tier_file, frida_mode)
    grace, tail = resolve_frida_capture_seconds(
        app,
        tier_file,
        grace_default=grace_sec,
        tail_default=tail_sec,
    )
    _prepare_app_foreground(app, pkg, frida_mode=launch_mode, adb_serial=adb_serial)
    out_root = _SRC / "output" / f"consent_{phase}"
    cmd = [
        sys.executable,
        "-m",
        "privacy_audit.cli",
        "capture-dynamic",
        "--app-name",
        app,
        "--app-package",
        pkg,
        "--policy-text-dir",
        "input/pii_text",
        "--output-dir",
        str(out_root.relative_to(_SRC)).replace("\\", "/"),
        "--real-run",
        "--skip-preflight",
        "--frida-mode",
        launch_mode,
        "--frida-tier-file",
        str(tier_file.relative_to(_SRC)).replace("\\", "/"),
        "--frida-grace-sec",
        str(grace),
        "--frida-tail-sec",
        str(tail),
        "--dynamic-track",
        "consent",
        "--probe-dir",
        "output/frida_probe",
        "--mitm-adb-proxy-host",
        mitm_adb_proxy_host,
    ]
    if adb_serial:
        cmd.extend(["--adb-serial", adb_serial])
    if use_frida:
        cmd.append("--enable-frida")
    if verbose:
        cmd.insert(3, "-v")
    print(
        f"  [{app}] track=consent(mitm-primary) frida={'ON' if use_frida else 'OFF'} "
        f"mode={launch_mode} grace={grace}s tail={tail}s",
        flush=True,
    )
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=_SRC, check=True)


def _sync_flat(phase: str) -> None:
    """Copy dynamic_evidence.json -> input/consent_{before|after}/<App>.json for consent CLI."""
    src_root = _SRC / "output" / f"consent_{phase}"
    dst_root = _SRC / "input" / f"consent_{phase}"
    dst_root.mkdir(parents=True, exist_ok=True)
    for app_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        src = app_dir / "dynamic_evidence.json"
        if src.exists():
            dst = dst_root / f"{app_dir.name}.json"
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _run_compare(apps_file: Path) -> None:
    cmd = [
        sys.executable,
        str(_SRC / "scripts" / "compare_consent_ui.py"),
        "--batch",
        "--apps-file",
        str(apps_file.relative_to(_SRC)).replace("\\", "/"),
        "--before-dir",
        "output/consent_before",
        "--after-dir",
        "output/consent_after",
        "--output-dir",
        "output/consent_ui",
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=_SRC, check=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Consent audit UI batch (before/after/compare)")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["before", "after", "compare", "sync", "all"],
        help="before|after=stage3; compare=UI diff; sync=copy to input/lab4_2_*; all=one app interactive",
    )
    parser.add_argument(
        "--apps-file",
        default=str(DEFAULT_APPS.relative_to(_SRC)).replace("\\", "/"),
        metavar="PATH",
        help="App list file (default: input/apps_full_batch_15.txt)",
    )
    parser.add_argument(
        "--app",
        "--apps",
        default="",
        dest="app",
        metavar="NAME",
        help="Single app or comma-separated list (overrides --apps-file)",
    )
    parser.add_argument(
        "--frida-tier-file",
        default=str(TIER_FILE.relative_to(_SRC)).replace("\\", "/"),
        help="Frida tier map (default: input/frida_tier.json)",
    )
    parser.add_argument(
        "--enable-frida",
        action="store_true",
        help="Force Frida on for every app in this batch",
    )
    parser.add_argument(
        "--no-frida",
        action="store_true",
        help="Force UI-only (never pass --enable-frida)",
    )
    parser.add_argument(
        "--frida-mode",
        default="auto",
        choices=["auto", "attach", "spawn"],
        help="Frida launch mode (default: auto from frida_tier.json)",
    )
    parser.add_argument(
        "--frida-grace-sec",
        type=int,
        default=DEFAULT_FRIDA_GRACE_SEC,
        help=f"Seconds before Appium starts (default: {DEFAULT_FRIDA_GRACE_SEC})",
    )
    parser.add_argument(
        "--frida-tail-sec",
        type=int,
        default=DEFAULT_FRIDA_TAIL_SEC,
        help=f"Seconds after Appium ends (default: {DEFAULT_FRIDA_TAIL_SEC})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--mitm-adb-proxy-host",
        default=os.environ.get("MITM_ADB_PROXY_HOST", "10.0.2.2"),
        help="Host IP for adb http_proxy during mitm (MuMu: ipconfig, e.g. 172.27.154.110)",
    )
    parser.add_argument(
        "--adb-serial",
        default=os.environ.get("ANDROID_SERIAL", ""),
        help="adb -s serial when multiple devices (default: auto-pick MuMu 16384)",
    )
    args = parser.parse_args()

    tier_file = Path(args.frida_tier_file)
    if not tier_file.is_absolute():
        tier_file = _SRC / tier_file

    apps_path = Path(args.apps_file)
    if not apps_path.is_absolute():
        apps_path = _SRC / apps_path
    apps = _load_apps(apps_path, args.app)

    capture_kwargs = {
        "verbose": args.verbose,
        "enable_frida": True if args.enable_frida else None,
        "disable_frida": bool(args.no_frida),
        "frida_mode": args.frida_mode,
        "grace_sec": args.frida_grace_sec,
        "tail_sec": args.frida_tail_sec,
        "tier_file": tier_file,
        "mitm_adb_proxy_host": args.mitm_adb_proxy_host,
        "adb_serial": args.adb_serial.strip() or None,
    }

    if args.phase == "compare":
        _run_compare(apps_path)
        return 0

    if args.phase == "sync":
        _sync_flat("before")
        _sync_flat("after")
        print("Synced to input/consent_before and input/consent_after", flush=True)
        return 0

    if args.phase == "all":
        if len(apps) != 1:
            print("--phase all requires exactly one --app", flush=True)
            return 1
        app = apps[0]
        print(f"\n=== BEFORE ({app}) — keep full/consent-ON ===\n", flush=True)
        _run_dynamic_capture(app, "before", **capture_kwargs)
        _sync_flat("before")
        input("\n>>> Toggle OFF in emulator, then press Enter to run AFTER dynamic capture...\n")
        print(f"\n=== AFTER ({app}) ===\n", flush=True)
        _run_dynamic_capture(app, "after", **capture_kwargs)
        _sync_flat("after")
        _run_compare(apps_path)
        return 0

    for idx, app in enumerate(apps):
        print(f"\n=== DYNAMIC_CAPTURE {args.phase.upper()} — {app} ({idx + 1}/{len(apps)}) ===\n", flush=True)
        _run_dynamic_capture(app, args.phase, **capture_kwargs)
        print(f"  [{app}] dynamic verification done (app auto-closed)", flush=True)
        if idx + 1 < len(apps):
            time.sleep(1.0)
    _sync_flat(args.phase)
    if args.phase == "after":
        print(
            "\nNext: python scripts/run_consent_ui_batch.py --phase compare\n",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

