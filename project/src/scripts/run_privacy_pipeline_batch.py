#!/usr/bin/env python3
"""
Privacy audit full pipeline batch (Static → Dynamic → Disclosure → Consent → Excel).

在 project/src 下执行：

  # 查看步骤
  python scripts/run_privacy_pipeline_batch.py --help

  # 单 App 全流程（推荐先验证）
  python scripts/run_privacy_pipeline_batch.py --apps 钓鱼人 --phase all --real-run

  # 仅 Stage3（逐 App 真机，自动选 Frida/mitm/UI）
  python scripts/run_privacy_pipeline_batch.py --phase dynamic --real-run --apps Keep,小红书

  # 15 App 静态 + 动态（Stage3 需逐 App 登录后跑）
  python scripts/run_privacy_pipeline_batch.py --phase static,dynamic,disclosure --real-run

  # Lab4-2：先 before，手动关隐私开关，再 after + compare
  python scripts/run_privacy_pipeline_batch.py --phase consent-before --apps 钓鱼人 --real-run
  python scripts/run_privacy_pipeline_batch.py --phase consent-after,consent-compare,consent --apps 钓鱼人

环境变量（可选）：
  MITM_ADB_PROXY_HOST=172.27.154.110
  ANDROID_SERIAL=127.0.0.1:16384

前置：MuMu/adb 单设备、Appium:4723、mitmdump、系统 CA、input/pii_text/<App>.txt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_PACKAGES = _SRC / "input" / "app_packages.json"
DEFAULT_APPS_FILE = _SRC / "input" / "apps_full_batch_15.txt"
PII_DIR = _SRC / "input" / "pii_text"
STATIC_OUT = _SRC / "output" / "static"
DYNAMIC_OUT = _SRC / "output" / "dynamic"
TIER_FILE = _SRC / "input" / "frida_tier.json"
PROBE_DIR = _SRC / "output" / "frida_probe"

APP_SLUGS: dict[str, str] = {
    "Keep": "keep",
    "小红书": "xiaohongshu",
    "支付宝": "alipay",
    "LOFTER": "lofter",
    "智联招聘": "zhilianzhaopin",
    "世纪佳缘": "shijiijiayuan",
    "好大夫医生版": "haodaifuyishengban",
    "东方财富": "dongfangcaifu",
    "小荷健康": "xiaohejiankang",
    "58同城": "58tongcheng",
    "钓鱼人": "diaoyuren",
    "韭研公社": "jiuyangongshe",
    "快看漫画": "kuaikanmanhua",
    "杉果游戏": "shanguoyouxi",
    "小黑盒": "xiaoheihe",
    "咪噜游戏": "miluyouxi",
}

PHASES = [
    "preflight",
    "probe",
    "static",
    "dynamic",
    "disclosure",
    "consent-before",
    "consent-after",
    "consent-compare",
    "consent",
    "excel",
    "matrix",
]


def _load_packages(path: Path) -> dict[str, str]:
    from privacy_audit.analyzers.dynamic.frida_runner import parse_app_package_entry

    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for name, entry in raw.items():
        pkg, _, _ = parse_app_package_entry(entry)
        if pkg:
            out[str(name)] = pkg
    return out


def _load_apps_list(path: Path) -> list[str]:
    apps: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            apps.append(line)
    return apps


def _discover_apps(packages: dict[str, str]) -> list[str]:
    """Apps with package mapping and policy text."""
    apps = [name for name in packages if (PII_DIR / f"{name}.txt").exists()]
    return sorted(apps)


LEGACY_STATIC_OUT = _SRC / "output" / "stage2_static_apk"


def _resolve_static_file(app_name: str) -> Path | None:
    slug = APP_SLUGS.get(app_name, app_name.lower().replace(" ", "_"))
    for root in (STATIC_OUT, LEGACY_STATIC_OUT):
        for rel in (
            root / slug / "static_evidence.json",
            root / app_name / "static_evidence.json",
        ):
            if rel.exists():
                return rel
    return None


def _run_cli(cmd: list[str], *, verbose: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    full = [sys.executable, "-m", "privacy_audit.cli"]
    if verbose:
        full.append("-v")
    full.extend(cmd)
    print("+", " ".join(full), flush=True)
    return subprocess.run(full, cwd=_SRC, check=check)


def _run_script(rel: str, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(_SRC / "scripts" / rel), *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=_SRC, check=check)


def _adb_preflight(adb_serial: str | None) -> dict[str, Any]:
    from privacy_audit.utils.adb_device import adb_device_status, list_connected_devices

    devices = list_connected_devices()
    status = adb_device_status(serial=adb_serial)
    ok = len(devices) >= 1 and status.get("resolved_serial")
    if len(devices) > 1 and not adb_serial and not os.environ.get("ANDROID_SERIAL"):
        print(
            "[warn] 多台 adb 设备已连接，将自动选用 MuMu 16384；"
            "建议 adb disconnect 多余端口或设置 ANDROID_SERIAL",
            flush=True,
        )
    return {"ok": ok, **status}


def _run_probe(apps: list[str], packages: dict[str, str], *, verbose: bool) -> None:
    probe_script = _SRC / "scripts" / "probe_hook.py"
    if not probe_script.exists():
        print("[skip] probe_hook.py not found", flush=True)
        return
    for app in apps:
        pkg = packages.get(app, "")
        if not pkg:
            continue
        slug = APP_SLUGS.get(app, app)
        out = PROBE_DIR / f"{slug.lower()}.json"
        if out.exists():
            print(f"[skip probe] {app} -> {out.name} exists", flush=True)
            continue
        cmd = [
            sys.executable,
            str(probe_script),
            "--app-name",
            app,
            "--app-package",
            pkg,
        ]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=_SRC, check=False)


def _run_static_scan(*, fast: bool, verbose: bool) -> None:
    apk_dir = None
    for name in ("apk", "apks"):
        candidate = _SRC / "input" / name
        if candidate.is_dir() and any(candidate.glob("*.apk")):
            apk_dir = candidate
            break
    if apk_dir is None:
        print("[skip static] no APK under input/apk or input/apks", flush=True)
        return
    cmd = [
        "scan-static",
        "--apk-dir",
        str(apk_dir.relative_to(_SRC)).replace("\\", "/"),
        "--output-dir",
        "output/static",
    ]
    if fast:
        cmd.append("--fast")
    _run_cli(cmd, verbose=verbose)


def _run_dynamic_capture(
    app: str,
    package: str,
    *,
    real_run: bool,
    verbose: bool,
    mitm_host: str,
    adb_serial: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    cmd = [
        "capture-dynamic",
        "--app-name",
        app,
        "--app-package",
        package,
        "--policy-text-dir",
        "input/pii_text",
        "--output-dir",
        str(output_dir.relative_to(_SRC)).replace("\\", "/"),
        "--dynamic-track",
        "auto",
        "--probe-dir",
        "output/frida_probe",
        "--frida-tier-file",
        "input/frida_tier.json",
        "--mitm-adb-proxy-host",
        mitm_host,
        "--skip-preflight",
    ]
    if adb_serial:
        cmd.extend(["--adb-serial", adb_serial])
    static_path = _resolve_static_file(app)
    if static_path:
        cmd.extend(
            [
                "--static-evidence-file",
                str(static_path.relative_to(_SRC)).replace("\\", "/"),
            ]
        )
    if real_run:
        cmd.append("--real-run")
    _run_cli(cmd, verbose=verbose)
    dyn = output_dir / app / "dynamic_evidence.json"
    row: dict[str, Any] = {"app_name": app, "dynamic_evidence": str(dyn) if dyn.exists() else ""}
    if dyn.exists():
        data = json.loads(dyn.read_text(encoding="utf-8"))
        net = data.get("network_result") or {}
        row["mitm_flow_count"] = net.get("flow_count", 0)
        row["tracker_hit_count"] = (net.get("summary") or {}).get("tracker_hit_count", 0)
        row["mitm_blocked"] = net.get("mitm_blocked")
    return row


def _run_disclosure_audit(app: str, *, verbose: bool, dynamic_dir: Path) -> None:
    policy = PII_DIR / f"{app}.txt"
    if not policy.exists():
        print(f"[skip disclosure] no policy for {app}", flush=True)
        return
    _run_cli(
        [
            "audit-disclosure",
            "--app-name",
            app,
            "--policy-file",
            str(policy.relative_to(_SRC)).replace("\\", "/"),
            "--evidence-dir",
            str(dynamic_dir.relative_to(_SRC)).replace("\\", "/"),
            "--static-evidence-dir",
            "output/static",
            "--output-dir",
            "output/disclosure",
        ],
        verbose=verbose,
    )


def _run_consent_capture_phase(
    phase: str,
    apps: list[str],
    *,
    verbose: bool,
    mitm_host: str,
    adb_serial: str | None,
) -> None:
    args = [
        "--phase",
        phase,
        "--app",
        ",".join(apps),
        "--mitm-adb-proxy-host",
        mitm_host,
    ]
    if adb_serial:
        args.extend(["--adb-serial", adb_serial])
    if verbose:
        args.append("-v")
    _run_script("run_consent_ui_batch.py", args)


def _run_consent_compare(apps_file: Path, *, verbose: bool) -> None:
    args = [
        "--phase",
        "compare",
        "--apps-file",
        str(apps_file.relative_to(_SRC)).replace("\\", "/"),
    ]
    if verbose:
        args.append("-v")
    _run_script("run_consent_ui_batch.py", args)


def _run_consent_audit(app: str, *, verbose: bool) -> None:
    before = _SRC / "input" / "consent_before" / f"{app}.json"
    after = _SRC / "input" / "consent_after" / f"{app}.json"
    if not before.exists() or not after.exists():
        print(f"[skip consent] missing before/after for {app}", flush=True)
        return
    _run_cli(
        [
            "audit-consent",
            "--app-name",
            app,
            "--before-off-file",
            str(before.relative_to(_SRC)).replace("\\", "/"),
            "--after-off-file",
            str(after.relative_to(_SRC)).replace("\\", "/"),
            "--output-dir",
            "output/consent",
        ],
        verbose=verbose,
    )


def _run_excel(apps: list[str], *, verbose: bool, dynamic_dir: Path) -> None:
    for app in apps:
        _run_cli(
            [
                "export-excel",
                "--app-name",
                app,
                "--static-evidence-dir",
                "output/static",
                "--dynamic-evidence-dir",
                str(dynamic_dir.relative_to(_SRC)).replace("\\", "/"),
            ],
            verbose=verbose,
            check=False,
        )


def _parse_phases(raw: str) -> list[str]:
    if raw.strip().lower() in ("all", "*"):
        return list(PHASES)
    chosen = [p.strip() for p in raw.split(",") if p.strip()]
    for p in chosen:
        if p not in PHASES:
            raise SystemExit(f"Unknown phase: {p}. Choose from: {', '.join(PHASES)}")
    return chosen


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Privacy audit pipeline batch (static/dynamic + rights audits)")
    parser.add_argument(
        "--phase",
        default="all",
        help=f"Comma-separated phases or 'all': {', '.join(PHASES)}",
    )
    parser.add_argument("--apps", default="", help="Comma-separated app names (default: apps-file list)")
    parser.add_argument(
        "--apps-file",
        default=str(DEFAULT_APPS_FILE.relative_to(_SRC)).replace("\\", "/"),
        help="App list when --apps empty; use 'auto' for all apps with pii_text+package",
    )
    parser.add_argument("--packages-file", default="input/app_packages.json")
    parser.add_argument("--real-run", action="store_true", help="Dynamic capture / consent use real Appium+mitm")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--mitm-adb-proxy-host",
        default=os.environ.get("MITM_ADB_PROXY_HOST", "172.27.154.110"),
    )
    parser.add_argument(
        "--adb-serial",
        default=os.environ.get("ANDROID_SERIAL", "127.0.0.1:16384"),
        help="adb -s serial (default 127.0.0.1:16384 or ANDROID_SERIAL)",
    )
    parser.add_argument(
        "--dynamic-output-dir",
        default="output/dynamic",
        help="Dynamic output root (default output/dynamic)",
    )
    parser.add_argument("--fast-static", action="store_true", default=True)
    parser.add_argument("--no-fast-static", action="store_true")
    parser.add_argument("--run-probe", action="store_true", help="Force probe phase even if not in --phase")
    parser.add_argument(
        "--pause-between-apps",
        action="store_true",
        help="After each dynamic capture app, wait for Enter (switch login / foreground)",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    packages_path = _SRC / args.packages_file
    packages = _load_packages(packages_path)

    if args.apps.strip():
        apps = [a.strip() for a in args.apps.split(",") if a.strip()]
    elif args.apps_file.strip().lower() == "auto":
        apps = _discover_apps(packages)
    else:
        apps_path = _SRC / args.apps_file
        apps = _load_apps_list(apps_path)

    phases = _parse_phases(args.phase)
    if args.run_probe and "probe" not in phases:
        phases.insert(0, "probe")

    dynamic_dir = _SRC / args.dynamic_output_dir
    adb_serial = args.adb_serial.strip() or None
    fast_static = args.fast_static and not args.no_fast_static

    # temp apps file for consent compare when using explicit --apps
    apps_file_for_compare = _SRC / "input" / "_pipeline_batch_apps.txt"
    apps_file_for_compare.write_text("\n".join(apps) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "phases": phases,
        "apps": apps,
        "dynamic_reports": [],
    }
    print(f"Pipeline apps ({len(apps)}): {', '.join(apps)}", flush=True)
    print(f"Phases: {', '.join(phases)}", flush=True)

    if "preflight" in phases and not args.skip_preflight:
        pf = _adb_preflight(adb_serial)
        print(f"adb preflight: {pf}", flush=True)
        if not pf.get("ok"):
            print("[error] adb 无可用设备，请先 adb connect", flush=True)
            return 1

    if "probe" in phases:
        print("\n=== PROBE ===\n", flush=True)
        _run_probe(apps, packages, verbose=args.verbose)

    if "static" in phases:
        print("\n=== STATIC ===\n", flush=True)
        _run_static_scan(fast=fast_static, verbose=args.verbose)

    if "dynamic" in phases:
        print("\n=== DYNAMIC CAPTURE ===\n", flush=True)
        for idx, app in enumerate(apps):
            pkg = packages.get(app, "")
            if not pkg:
                print(f"[skip dynamic] {app}: no package in app_packages.json", flush=True)
                continue
            if not (PII_DIR / f"{app}.txt").exists():
                print(f"[skip dynamic] {app}: no input/pii_text/{app}.txt", flush=True)
                continue
            print(f"\n--- Dynamic capture {idx + 1}/{len(apps)}: {app} ---\n", flush=True)
            row = _run_dynamic_capture(
                app,
                pkg,
                real_run=args.real_run,
                verbose=args.verbose,
                mitm_host=args.mitm_adb_proxy_host,
                adb_serial=adb_serial,
                output_dir=dynamic_dir,
            )
            summary["dynamic_reports"].append(row)
            if args.pause_between_apps and idx + 1 < len(apps):
                input(f"\n>>> [{app}] 完成。切换下一 App 登录后按 Enter 继续...\n")

    if "disclosure" in phases:
        print("\n=== DISCLOSURE AUDIT ===\n", flush=True)
        for app in apps:
            _run_disclosure_audit(app, verbose=args.verbose, dynamic_dir=dynamic_dir)

    if "consent-before" in phases:
        print("\n=== CONSENT BEFORE ===\n", flush=True)
        for app in apps:
            _run_consent_capture_phase(
                "before",
                [app],
                verbose=args.verbose,
                mitm_host=args.mitm_adb_proxy_host,
                adb_serial=adb_serial,
            )
            if args.pause_between_apps:
                input(f"\n>>> [{app}] BEFORE 完成。若批量请切换 App；按 Enter 继续 AFTER 前请先关隐私开关...\n")

    if "consent-after" in phases:
        if "consent-before" in phases and args.real_run:
            input(
                "\n>>> 请在模拟器中逐个关闭隐私开关/撤回同意，完成后按 Enter 运行 AFTER...\n"
            )
        print("\n=== CONSENT AFTER ===\n", flush=True)
        for app in apps:
            _run_consent_capture_phase(
                "after",
                [app],
                verbose=args.verbose,
                mitm_host=args.mitm_adb_proxy_host,
                adb_serial=adb_serial,
            )

    if "consent-compare" in phases:
        print("\n=== CONSENT COMPARE (UI diff) ===\n", flush=True)
        _run_consent_compare(apps_file_for_compare, verbose=args.verbose)
        _run_script("run_consent_ui_batch.py", ["--phase", "sync"])

    if "consent" in phases:
        print("\n=== CONSENT REPORT ===\n", flush=True)
        for app in apps:
            _run_consent_audit(app, verbose=args.verbose)

    if "excel" in phases:
        print("\n=== EXPORT EXCEL ===\n", flush=True)
        _run_excel(apps, verbose=args.verbose, dynamic_dir=dynamic_dir)

    if "matrix" in phases:
        print("\n=== TRACK MATRIX ===\n", flush=True)
        _run_script("generate_app_track_matrix.py", [])

    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    out = _SRC / "output" / "pipeline_batch_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPipeline summary -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
