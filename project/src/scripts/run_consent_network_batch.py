#!/usr/bin/env python3
"""
Consent audit mitm-primary batch: before → manual toggle → after → compare → consent reports.

每个 App 跑完 Dynamic capture 动态验证后会自动 adb force-stop 关闭应用，再进入下一个。

在 project/src 下执行：

  # 15 个 App 全流程（推荐）
  python scripts/run_consent_network_batch.py --phase all --apps-file input/apps_full_batch_15.txt -v

  # 全部已配置 App（app_packages.json + pii_text）
  python scripts/run_consent_network_batch.py --phase all --apps-file auto -v

  # 仅 BEFORE
  python scripts/run_consent_network_batch.py --phase before --apps-file input/apps_full_batch_15.txt -v

  # 手动关闭各 App 隐私开关后跑 AFTER
  python scripts/run_consent_network_batch.py --phase after --apps-file input/apps_full_batch_15.txt -v

  # 对比 + Consent audit 正式报告
  python scripts/run_consent_network_batch.py --phase compare,consent --apps-file input/apps_full_batch_15.txt

环境变量：
  MITM_ADB_PROXY_HOST=172.27.154.110
  ANDROID_SERIAL=127.0.0.1:16384
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
DEFAULT_APPS = _SRC / "input" / "apps_full_batch_15.txt"
PACKAGES = _SRC / "input" / "app_packages.json"
PII_DIR = _SRC / "input" / "pii_text"


def _load_packages() -> dict[str, str]:
    from privacy_audit.analyzers.dynamic.frida_runner import parse_app_package_entry

    raw = json.loads(PACKAGES.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for name, entry in raw.items():
        pkg, _, _ = parse_app_package_entry(entry)
        if pkg:
            out[str(name)] = pkg
    return out


def _discover_apps() -> list[str]:
    packages = _load_packages()
    return sorted(name for name in packages if (PII_DIR / f"{name}.txt").exists())


def _load_apps(path: Path | None, only: str, *, apps_file: str) -> list[str]:
    if only.strip():
        return [a.strip() for a in only.split(",") if a.strip()]
    if apps_file.strip().lower() == "auto":
        return _discover_apps()
    if path is None or not path.exists():
        raise FileNotFoundError(f"apps file not found: {path}")
    apps: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            apps.append(line)
    return apps


def _parse_phases(raw: str) -> list[str]:
    if raw.strip().lower() in {"all", "*"}:
        return ["before", "after", "compare", "consent"]
    return [p.strip() for p in raw.split(",") if p.strip()]


def _run_consent_capture(phase: str, apps: list[str], args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(_SRC / "scripts" / "run_consent_ui_batch.py"),
        "--phase",
        phase,
        "--app",
        ",".join(apps),
        "--mitm-adb-proxy-host",
        args.mitm_adb_proxy_host,
    ]
    if args.adb_serial:
        cmd.extend(["--adb-serial", args.adb_serial])
    if args.verbose:
        cmd.append("-v")
    if args.enable_frida:
        cmd.append("--enable-frida")
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=_SRC, check=True)


def _run_consent_audit(apps: list[str], *, verbose: bool) -> None:
    for app in apps:
        before = _SRC / "input" / "consent_before" / f"{app}.json"
        after = _SRC / "input" / "consent_after" / f"{app}.json"
        if not before.exists() or not after.exists():
            print(f"[skip consent] {app}: missing before/after input", flush=True)
            continue
        cmd = [
            sys.executable,
            "-m",
            "privacy_audit.cli",
            "audit-consent",
            "--app-name",
            app,
            "--before-off-file",
            str(before.relative_to(_SRC)).replace("\\", "/"),
            "--after-off-file",
            str(after.relative_to(_SRC)).replace("\\", "/"),
            "--output-dir",
            "output/consent",
        ]
        if verbose:
            cmd.insert(3, "-v")
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=_SRC, check=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Consent audit mitm-primary batch")
    parser.add_argument(
        "--phase",
        default="all",
        help="before | after | compare | consent | all | comma-separated (default: all)",
    )
    parser.add_argument("--app", "--apps", default="", dest="app")
    parser.add_argument(
        "--apps-file",
        default=str(DEFAULT_APPS.relative_to(_SRC)).replace("\\", "/"),
        help="App list file, or 'auto' to discover from app_packages + pii_text",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--mitm-adb-proxy-host",
        default=os.environ.get("MITM_ADB_PROXY_HOST", "172.27.154.110"),
    )
    parser.add_argument(
        "--adb-serial",
        default=os.environ.get("ANDROID_SERIAL", "127.0.0.1:16384"),
    )
    parser.add_argument(
        "--enable-frida",
        action="store_true",
        help="Also enable Frida (supplementary; default mitm-only)",
    )
    parser.add_argument(
        "--pause-between-apps",
        action="store_true",
        help="Wait for Enter between apps in before/after phases",
    )
    args = parser.parse_args()

    apps_path: Path | None = None
    if args.apps_file.strip().lower() != "auto":
        apps_path = Path(args.apps_file)
        if not apps_path.is_absolute():
            apps_path = _SRC / apps_path
    apps = _load_apps(apps_path, args.app, apps_file=args.apps_file)
    phases = _parse_phases(args.phase)

    summary: dict = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "phases": phases,
        "apps": apps,
        "mitm_primary": True,
        "auto_close_app_after_capture": True,
    }
    print(f"Consent audit mitm batch ({len(apps)} apps): {', '.join(apps)}", flush=True)

    if "before" in phases:
        for idx, app in enumerate(apps):
            print(f"\n=== BEFORE {idx + 1}/{len(apps)}: {app} ===\n", flush=True)
            _run_consent_capture("before", [app], args)
            if args.pause_between_apps and idx + 1 < len(apps):
                input(">>> 登录下一 App 后按 Enter...\n")

    if "after" in phases:
        if "before" in phases:
            input(
                "\n>>> 请在模拟器中逐个关闭各 App 隐私开关/撤回同意，完成后按 Enter 运行 AFTER...\n"
            )
        for idx, app in enumerate(apps):
            print(f"\n=== AFTER {idx + 1}/{len(apps)}: {app} ===\n", flush=True)
            _run_consent_capture("after", [app], args)
            if args.pause_between_apps and idx + 1 < len(apps):
                input(">>> 切换下一 App 后按 Enter...\n")

    if "compare" in phases:
        apps_file_arg = args.apps_file
        if apps_path is not None:
            apps_file_arg = str(apps_path.relative_to(_SRC)).replace("\\", "/")
        elif args.app.strip():
            tmp = _SRC / "output" / "_consent_network_apps_tmp.txt"
            tmp.write_text("\n".join(apps) + "\n", encoding="utf-8")
            apps_file_arg = str(tmp.relative_to(_SRC)).replace("\\", "/")
        cmd = [
            sys.executable,
            str(_SRC / "scripts" / "run_consent_ui_batch.py"),
            "--phase",
            "compare",
            "--apps-file",
            apps_file_arg,
        ]
        if args.verbose:
            cmd.append("-v")
        subprocess.run(cmd, cwd=_SRC, check=True)
        subprocess.run(
            [sys.executable, str(_SRC / "scripts" / "run_consent_ui_batch.py"), "--phase", "sync"],
            cwd=_SRC,
            check=True,
        )

    if "consent" in phases:
        _run_consent_audit(apps, verbose=args.verbose)

    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    out = _SRC / "output" / "consent_network_batch_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
