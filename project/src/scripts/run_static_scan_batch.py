#!/usr/bin/env python3
"""Run full Disclosure audit pipeline for APK batch and print experiment statistics."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
JOB_ID = "apk_batch_001"
APPS_FILE = SRC_ROOT / "input" / "apps_apk_batch.txt"
LEGACY_RUN = SRC_ROOT.parent / "oldsrc" / "analyzer" / "output" / "run_20260427_001723"
LEGACY_MD = SRC_ROOT.parent / "oldsrc" / "analyzer" / "output" / "run_20260427_001723z"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=SRC_ROOT, check=True)


def _load_apps() -> list[str]:
    apps: list[str] = []
    for line in APPS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        apps.append(line)
    return apps


def _summarize_batch(summary_path: Path) -> dict:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    per_app: list[dict] = []
    totals = Counter()
    for report in data.get("reports", []):
        app = report.get("app_name", "")
        report_path = SRC_ROOT / report.get("report_path", "").replace("\\", "/")
        if not report_path.exists():
            report_path = SRC_ROOT / "output" / "disclosure" / f"{app}_disclosure_report.json"
        detail = {}
        if report_path.exists():
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            summary = rep.get("summary", {})
            status_counts = summary.get("status_counts", {})
            for k, v in status_counts.items():
                totals[k] += int(v)
            detail = {
                "declared_count": summary.get("declared_count", 0),
                "observed_count": summary.get("observed_count", 0),
                "status_counts": status_counts,
            }
        per_app.append(
            {
                "app_name": app,
                "batch_status": report.get("status"),
                **detail,
            }
        )
    return {
        "job_id": data.get("manifest", {}).get("job_id", JOB_ID),
        "total_apps": data.get("summary", {}).get("total_apps"),
        "success_count": data.get("summary", {}).get("success_count"),
        "failed_count": data.get("summary", {}).get("failed_count"),
        "aggregate_verdict_counts": dict(totals),
        "per_app": per_app,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    apps = _load_apps()
    print(f"Apps ({len(apps)}): {', '.join(apps)}")

    _run([sys.executable, "-m", "privacy_audit.cli", "kb-build"])
    _run(
        [
            sys.executable,
            "-m",
            "privacy_audit.cli",
            "scan-static",
            "--apk-dir",
            "input/apk",
            "--output-dir",
            "output/static",
            "--fast",
        ]
    )
    import_cmd = [
        sys.executable,
        "-m",
        "privacy_audit.cli",
        "import-legacy",
        "--apps-file",
        str(APPS_FILE.relative_to(SRC_ROOT)).replace("\\", "/"),
        "--static-out-dir",
        "output/static/legacy",
    ]
    if LEGACY_RUN.is_dir():
        import_cmd.extend(["--legacy-run-dir", str(LEGACY_RUN)])
    _run(import_cmd)

    _run(
        [
            sys.executable,
            "-m",
            "privacy_audit.cli",
            "build-evidence",
            "--apps-file",
            str(APPS_FILE.relative_to(SRC_ROOT)).replace("\\", "/"),
            "--static-evidence-dir",
            "output/static",
            "--legacy-run-dir",
            str(LEGACY_RUN),
            "--legacy-md-run-dir",
            str(LEGACY_MD),
            "--output-dir",
            "output/evidence",
        ]
    )

    batch_cmd = [
        sys.executable,
        "-m",
        "privacy_audit.cli",
        "batch",
        "--apps-file",
        str(APPS_FILE.relative_to(SRC_ROOT)).replace("\\", "/"),
        "--mode",
        "disclosure",
        "--policy-dir",
        "input/pii_text",
        "--evidence-dir",
        "output/evidence",
        "--static-evidence-dir",
        "output/static",
        "--output-dir",
        "output/disclosure",
        "--job-id",
        JOB_ID,
        "--retry-count",
        "2",
        "--force-rerun",
    ]
    if LEGACY_RUN.is_dir():
        batch_cmd.extend(["--legacy-run-dir", str(LEGACY_RUN)])
    if LEGACY_MD.is_dir():
        batch_cmd.extend(["--legacy-md-run-dir", str(LEGACY_MD)])
    _run(batch_cmd)

    summary_path = SRC_ROOT / "output" / "disclosure" / f"batch_disclosure_{JOB_ID}_summary.json"
    stats = _summarize_batch(summary_path)
    stats_path = SRC_ROOT / "output" / "disclosure" / f"experiment_stats_{JOB_ID}.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nStats written: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
