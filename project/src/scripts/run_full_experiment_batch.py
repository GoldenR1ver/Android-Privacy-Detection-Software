#!/usr/bin/env python3
"""
全量实验批处理：证据路径建立 + Static scan/3 + Disclosure audit/2/3。

在 project/src 目录下执行：

  python scripts/run_full_experiment_batch.py --help
  python scripts/run_full_experiment_batch.py
  python scripts/run_full_experiment_batch.py --real-run-dynamic -v
  python scripts/run_full_experiment_batch.py --from-step build-evidence

前置条件（见 README）：
  - Python 3.10+，依赖已安装
  - 政策文本：input/pii_text/<App>.txt
  - 静态扫描（可选）：input/apks/<App>.apk 或 input/apk/<App>.apk
  - Dynamic capture 真机：MuMu/adb、Appium(4723)、frida-server；各 App 已登录并保持前台
  - oldsrc 数据：../oldsrc/analyzer/output/run_20260427_001723（用于 legacy 证据导入）

Consent audit 说明：理想情况需「开关关闭前/后」各一份 dynamic_evidence。
  本脚本会将 stage3_dynamic_real 同步到 consent_before，将 stage3_oldsrc_derived
  同步到 consent_after（若存在）；若仅有单份真机结果，consent audit 结论仅供参考。
Access audit 说明：dynamic 来自 stage3 真机/样本/derived；copy 来自 oldsrc access_result.json。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
PROJECT_ROOT = SRC_ROOT.parent
DEFAULT_APPS_FILE = SRC_ROOT / "input" / "apps_full_batch_15.txt"
DEFAULT_PACKAGES_FILE = SRC_ROOT / "input" / "app_packages.json"
LEGACY_RUN = PROJECT_ROOT / "oldsrc" / "analyzer" / "output" / "run_20260427_001723"
LEGACY_MD = PROJECT_ROOT / "oldsrc" / "analyzer" / "output" / "run_20260427_001723z"

JOB_ID_PREFIX = "full_batch_15"

# 与 evidence_loader._slug_for_app 一致
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
}

STEPS_ORDER = [
    "kb-build",
    "static",
    "dynamic",
    "import-legacy",
    "build-evidence",
    "audit-disclosure",
    "prepare-audit-inputs",
    "audit-consent",
    "audit-access",
    "status",
]


def _utc_job_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_apps(path: Path) -> list[str]:
    apps: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        apps.append(line)
    return apps


def _load_packages(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    from privacy_audit.analyzers.dynamic.frida_runner import parse_app_package_entry

    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for name, entry in data.items():
        package, _prefixes, _opts = parse_app_package_entry(entry)
        out[str(name)] = package
    return out


def _run(cmd: list[str], *, verbose: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    full = [sys.executable, "-m", "privacy_audit.cli"]
    if verbose and "-v" not in cmd and "--verbose" not in cmd:
        full.append("-v")
    full.extend(cmd)
    print("+", " ".join(full), flush=True)
    return subprocess.run(full, cwd=SRC_ROOT, check=check)


def _resolve_apk_dir() -> Path | None:
    for name in ("apks", "apk"):
        candidate = SRC_ROOT / "input" / name
        if candidate.is_dir() and any(candidate.glob("*.apk")):
            return candidate
    return None


def _resolve_static_file(app_name: str, static_root: Path) -> Path | None:
    slug = APP_SLUGS.get(app_name, app_name.lower().replace(" ", "_"))
    candidates = [
        static_root / app_name / "static_evidence.json",
        static_root / slug / "static_evidence.json",
        static_root / app_name.lower() / "static_evidence.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_dynamic_source(app_name: str, prefer_real: bool = True) -> Path | None:
    roots: list[Path] = [
        SRC_ROOT / "output" / "consent_before" / app_name / "dynamic_evidence.json",
    ]
    if prefer_real:
        roots.append(SRC_ROOT / "output" / "stage3_dynamic_real" / app_name / "dynamic_evidence.json")
    roots.extend(
        [
            SRC_ROOT / "output" / "stage3_dynamic_samples" / app_name / "dynamic_evidence.json",
            SRC_ROOT / "output" / "stage3_oldsrc_derived" / app_name / "dynamic_evidence.json",
        ]
    )
    for path in roots:
        if path.exists():
            return path
    return None


def _copy_json(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def prepare_lab4_inputs(apps: list[str], *, verbose: bool = False) -> dict[str, Any]:
    """将 stage3 / oldsrc 产物同步到 Consent/access 批处理输入目录。"""
    lab42_before = SRC_ROOT / "input" / "consent_before"
    lab42_after = SRC_ROOT / "input" / "consent_after"
    lab43_dynamic = SRC_ROOT / "input" / "access_runtime"
    lab43_copy = SRC_ROOT / "input" / "access_copy"
    for d in (lab42_before, lab42_after, lab43_dynamic, lab43_copy):
        d.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"apps": {}, "legacy_run": str(LEGACY_RUN)}
    for app in apps:
        app_report: dict[str, str] = {}
        real_dyn = SRC_ROOT / "output" / "stage3_dynamic_real" / app / "dynamic_evidence.json"
        oldsrc_dyn = SRC_ROOT / "output" / "stage3_oldsrc_derived" / app / "dynamic_evidence.json"
        sample_dyn = SRC_ROOT / "output" / "stage3_dynamic_samples" / app / "dynamic_evidence.json"

        dyn_for_3 = _resolve_dynamic_source(app)
        if dyn_for_3 and _copy_json(dyn_for_3, lab43_dynamic / f"{app}.json"):
            app_report["access_runtime"] = str(dyn_for_3)

        access = LEGACY_RUN / "dynamic_testing" / "per_app" / app / "access_result.json"
        if _copy_json(access, lab43_copy / f"{app}.json"):
            app_report["access_copy"] = str(access)

        if _copy_json(real_dyn, lab42_before / f"{app}.json"):
            app_report["consent_before"] = str(real_dyn)
        elif _copy_json(sample_dyn, lab42_before / f"{app}.json"):
            app_report["consent_before"] = str(sample_dyn)

        if _copy_json(oldsrc_dyn, lab42_after / f"{app}.json"):
            app_report["consent_after"] = str(oldsrc_dyn)
        elif lab42_before.joinpath(f"{app}.json").exists() and not lab42_after.joinpath(f"{app}.json").exists():
            shutil.copy2(lab42_before / f"{app}.json", lab42_after / f"{app}.json")
            app_report["consent_after"] = "(same as before; replace after manual switch-off run)"

        report["apps"][app] = app_report
        if verbose:
            print(f"  [{app}] {app_report}", flush=True)

    out = SRC_ROOT / "output" / "lab4_inputs_prepare_summary.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["summary_path"] = str(out)
    return report


def _step_index(name: str) -> int:
    try:
        return STEPS_ORDER.index(name)
    except ValueError:
        raise SystemExit(f"Unknown step: {name}. Choose from: {', '.join(STEPS_ORDER)}")


def _should_run(step: str, from_step: str | None, only_step: str | None) -> bool:
    if only_step:
        return step == only_step
    if from_step is None:
        return True
    return _step_index(step) >= _step_index(from_step)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="全量实验批处理（证据 + Static scan/3 + Disclosure audit/2/3）")
    parser.add_argument(
        "--apps-file",
        default=str(DEFAULT_APPS_FILE.relative_to(SRC_ROOT)).replace("\\", "/"),
        help="App 列表文件（默认 input/apps_full_batch_15.txt）",
    )
    parser.add_argument(
        "--packages-file",
        default=str(DEFAULT_PACKAGES_FILE.relative_to(SRC_ROOT)).replace("\\", "/"),
        help="App 名 -> Android 包名 JSON（默认 input/app_packages.json）",
    )
    parser.add_argument(
        "--job-id",
        default="",
        help="Batch job id 前缀；默认 full_batch_15_<UTC时间>",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="CLI DEBUG 日志")
    parser.add_argument(
        "--real-run-dynamic",
        action="store_true",
        help="Dynamic capture 使用 --real-run（需 Appium + Frida + 各 App 前台登录）",
    )
    parser.add_argument(
        "--skip-dynamic",
        action="store_true",
        help="跳过 Dynamic capture（仅用 oldsrc/样本动态证据）",
    )
    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="跳过 Static scan APK 扫描",
    )
    parser.add_argument(
        "--fast-static",
        action="store_true",
        default=True,
        help="Static scan 使用 --fast（默认开启）",
    )
    parser.add_argument(
        "--no-fast-static",
        action="store_true",
        help="Static scan 完整 Androguard 分析（较慢）",
    )
    parser.add_argument(
        "--from-step",
        choices=STEPS_ORDER,
        help=f"从某步开始执行（含该步）：{', '.join(STEPS_ORDER)}",
    )
    parser.add_argument(
        "--only-step",
        choices=STEPS_ORDER,
        help="只执行单步",
    )
    parser.add_argument(
        "--apps",
        default="",
        help="覆盖列表：逗号分隔 App 名",
    )
    parser.add_argument(
        "--dynamic-app",
        default="",
        help="仅对单个 App 跑 Dynamic capture（用于逐 App 真机轮换）",
    )
    parser.add_argument(
        "--no-knowledge-base",
        action="store_true",
        help="证据包与 Disclosure audit batch 关闭法律知识库增强",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=2,
        help="batch 失败重试次数",
    )
    args = parser.parse_args()

    def _resolve_under_src(rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return (SRC_ROOT / p).resolve()

    if args.apps.strip():
        apps = [a.strip() for a in args.apps.split(",") if a.strip()]
    else:
        apps_path = _resolve_under_src(args.apps_file)
        apps = _load_apps(apps_path)

    packages_path = _resolve_under_src(args.packages_file)
    packages = _load_packages(packages_path)

    job_suffix = _utc_job_suffix()
    job_id = args.job_id or f"{JOB_ID_PREFIX}_{job_suffix}"
    job_disclosure = f"{job_id}_lab4_1"
    job_consent = f"{job_id}_lab4_2"
    job_access = f"{job_id}_lab4_3"

    static_out = SRC_ROOT / "output" / "stage2_static_apk"
    stage3_out = SRC_ROOT / "output" / "stage3_dynamic_real"
    evidence_out = SRC_ROOT / "output" / "lab4_1_evidence"
    disclosure_out = SRC_ROOT / "output" / "lab4_1"
    consent_out = SRC_ROOT / "output" / "lab4_2"
    access_out = SRC_ROOT / "output" / "lab4_3"

    if (SRC_ROOT / "output" / "stage2_static_apk").is_dir():
        static_evidence_dir = "output/static"
    else:
        static_evidence_dir = "output/static"

    apps_file_resolved = _resolve_under_src(args.apps_file)
    try:
        apps_rel = str(apps_file_resolved.relative_to(SRC_ROOT)).replace("\\", "/")
    except ValueError:
        apps_rel = str(apps_file_resolved)

    print(f"Apps ({len(apps)}): {', '.join(apps)}", flush=True)
    print(f"Job id base: {job_id}", flush=True)
    print(f"Working directory: {SRC_ROOT}", flush=True)

    kb_flags: list[str] = []
    if args.no_knowledge_base:
        kb_flags = ["--no-knowledge-base"]

    def run_step(step: str) -> None:
        if not _should_run(step, args.from_step, args.only_step):
            return

        if step == "kb-build":
            _run(["kb-build"], verbose=args.verbose)

        elif step == "static":
            if args.skip_static:
                print("[skip] stage2", flush=True)
                return
            apk_dir = _resolve_apk_dir()
            if apk_dir is None:
                print(
                    "[warn] 未找到 input/apks 或 input/apk 下的 *.apk，跳过 scan-static；"
                    "将依赖 import-legacy 与 stage2_static_samples。",
                    flush=True,
                )
                return
            cmd = [
                "scan-static",
                "--apk-dir",
                str(apk_dir.relative_to(SRC_ROOT)).replace("\\", "/"),
                "--output-dir",
                "output/static",
            ]
            if args.fast_static and not args.no_fast_static:
                cmd.append("--fast")
            _run(cmd, verbose=args.verbose)

        elif step == "dynamic":
            if args.skip_dynamic:
                print("[skip] stage3", flush=True)
                return
            dynamic_apps = [args.dynamic_app] if args.dynamic_app else apps
            missing_pkg: list[str] = []
            for app in dynamic_apps:
                pkg = packages.get(app, "")
                if not pkg:
                    missing_pkg.append(app)
                    continue
                static_file = _resolve_static_file(app, static_out)
                if static_file is None:
                    static_file = _resolve_static_file(
                        app, SRC_ROOT / "output" / "stage2_static_samples"
                    )
                cmd = [
                    "capture-dynamic",
                    "--app-name",
                    app,
                    "--app-package",
                    pkg,
                    "--policy-text-dir",
                    "input/pii_text",
                    "--output-dir",
                    "output/dynamic",
                    "--frida-mode",
                    "auto",
                    "--dynamic-track",
                    "auto",
                    "--skip-preflight",
                    "--mitm-adb-proxy-host",
                    os.environ.get("MITM_ADB_PROXY_HOST", "172.27.154.110"),
                ]
                adb_serial = os.environ.get("ANDROID_SERIAL", "127.0.0.1:16384").strip()
                if adb_serial:
                    cmd.extend(["--adb-serial", adb_serial])
                if static_file:
                    cmd.extend(
                        [
                            "--static-evidence-file",
                            str(static_file.relative_to(SRC_ROOT)).replace("\\", "/"),
                        ]
                    )
                if args.real_run_dynamic:
                    cmd.append("--real-run")
                # Tier-1 apps still get Frida via --dynamic-track auto + frida_tier.json
                else:
                    print(
                        f"[dry-run] capture-dynamic {app}（无 --real-run-dynamic）；"
                        "仅模拟，不产生真机 Frida 事件。",
                        flush=True,
                    )
                _run(cmd, verbose=args.verbose)
                print(
                    f"\n>>> Dynamic capture 完成: {app}。"
                    " 若需下一 App，请在模拟器中切换登录后重新运行 "
                    f'--only-step stage3 --dynamic-app "<下一App>"\n',
                    flush=True,
                )
            if missing_pkg:
                print(
                    "[warn] 以下 App 未在 app_packages.json 中配置包名，已跳过 Dynamic capture: "
                    + ", ".join(missing_pkg),
                    flush=True,
                )

        elif step == "import-legacy":
            if not LEGACY_RUN.is_dir():
                print(f"[warn] oldsrc 不存在，跳过 import-legacy: {LEGACY_RUN}", flush=True)
                return
            cmd = [
                "import-legacy",
                "--apps-file",
                apps_rel,
                "--static-out-dir",
                "output/static/legacy",
                "--dynamic-out-dir",
                "output/dynamic/legacy",
            ]
            _run(cmd, verbose=args.verbose)

        elif step == "build-evidence":
            cmd = [
                "build-evidence",
                "--apps-file",
                apps_rel,
                "--output-dir",
                "output/evidence",
                "--dynamic-evidence-dir",
                "output",
                "--static-evidence-dir",
                static_evidence_dir,
                *kb_flags,
            ]
            if LEGACY_RUN.is_dir():
                cmd.extend(["--legacy-run-dir", str(LEGACY_RUN)])
            if LEGACY_MD.is_dir():
                cmd.extend(["--legacy-md-run-dir", str(LEGACY_MD)])
            _run(cmd, verbose=args.verbose)

        elif step == "audit-disclosure":
            cmd = [
                "batch",
                "--apps-file",
                apps_rel,
                "--mode",
                "disclosure",
                "--policy-dir",
                "input/pii_text",
                "--evidence-dir",
                "output/evidence",
                "--static-evidence-dir",
                static_evidence_dir,
                "--output-dir",
                "output/disclosure",
                "--job-id",
                job_disclosure,
                "--retry-count",
                str(args.retry_count),
                "--force-rerun",
                *kb_flags,
            ]
            if LEGACY_RUN.is_dir():
                cmd.extend(["--legacy-run-dir", str(LEGACY_RUN)])
            if LEGACY_MD.is_dir():
                cmd.extend(["--legacy-md-run-dir", str(LEGACY_MD)])
            _run(cmd, verbose=args.verbose)

        elif step == "prepare-audit-inputs":
            summary = prepare_lab4_inputs(apps, verbose=args.verbose)
            print(f"Consent/access 输入已同步: {summary.get('summary_path')}", flush=True)

        elif step == "audit-consent":
            prepare_lab4_inputs(apps, verbose=False)
            cmd = [
                "batch",
                "--apps-file",
                apps_rel,
                "--mode",
                "consent",
                "--before-dir",
                "input/consent_before",
                "--after-dir",
                "input/consent_after",
                "--output-dir",
                "output/consent",
                "--job-id",
                job_consent,
                "--retry-count",
                str(args.retry_count),
                "--force-rerun",
            ]
            _run(cmd, verbose=args.verbose)

        elif step == "audit-access":
            prepare_lab4_inputs(apps, verbose=False)
            cmd = [
                "batch",
                "--apps-file",
                apps_rel,
                "--mode",
                "access",
                "--policy-dir",
                "input/pii_text",
                "--dynamic-dir",
                "input/access_runtime",
                "--copy-dir",
                "input/access_copy",
                "--output-dir",
                "output/access",
                "--job-id",
                job_access,
                "--retry-count",
                str(args.retry_count),
                "--force-rerun",
            ]
            _run(cmd, verbose=args.verbose)

        elif step == "status":
            for mode, out_dir, jid in (
                ("disclosure", disclosure_out, job_disclosure),
                ("consent", consent_out, job_consent),
                ("access", access_out, job_access),
            ):
                result = _run(
                    ["batch-status", "--output-dir", str(out_dir.relative_to(SRC_ROOT)).replace("\\", "/"), "--job-id", jid, "--mode", mode],
                    verbose=False,
                    check=False,
                )
                if result.returncode != 0:
                    print(f"[warn] batch-status {mode} job={jid} failed", flush=True)

    for step in STEPS_ORDER:
        run_step(step)

    print("\n=== 完成 ===", flush=True)
    print(f"证据包: {evidence_out}", flush=True)
    print(f"Disclosure audit: {disclosure_out} (job {job_disclosure})", flush=True)
    print(f"Consent audit: {consent_out} (job {job_consent})", flush=True)
    print(f"Access audit: {access_out} (job {job_access})", flush=True)
    print(f"Dynamic capture 真机: {stage3_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
