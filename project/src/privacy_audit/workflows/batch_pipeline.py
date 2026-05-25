"""Stage 7 batch orchestration: retry, resume, logging, task state."""

from __future__ import annotations

import json
import shutil
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from privacy_audit.reporting.report_exporter import export_json_report
from privacy_audit.utils.logger import setup_file_logger
from privacy_audit.utils.retry import run_with_retry


STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_job_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _infer_report_status(report: dict[str, Any]) -> str:
    explicit = str(report.get("status", "")).lower()
    if explicit in {"pass", "completed", "consistent"}:
        return STATUS_SUCCESS
    if explicit in {"todo", "pending"}:
        return STATUS_PENDING
    if explicit in {"review_needed", "failed", "error"}:
        return STATUS_FAILED
    if report.get("error"):
        return STATUS_FAILED
    if report.get("report_path") or report.get("verdicts") or report.get("comparisons"):
        return STATUS_SUCCESS
    return STATUS_SUCCESS


@dataclass
class BatchJobConfig:
    """Configuration for a batch run."""

    apps: list[str]
    mode: str
    output_dir: Path
    job_id: str = ""
    resume: bool = False
    force_rerun: bool = False
    retry_count: int = 2
    retry_delay_sec: float = 3.0
    retry_backoff: float = 2.0
    extra: dict[str, Any] = field(default_factory=dict)


class BatchStateManager:
    """Persist and query per-app task state for resume."""

    def __init__(self, job_dir: Path, config: BatchJobConfig) -> None:
        self.job_dir = job_dir
        self.config = config
        self.manifest_path = job_dir / "manifest.json"
        self._manifest = self._load_or_create()

    def _load_or_create(self) -> dict[str, Any]:
        if self.manifest_path.exists() and self.config.resume and not self.config.force_rerun:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            data["apps"] = list(self.config.apps)
            data["mode"] = self.config.mode
            data["updated_at_utc"] = _utc_now_iso()
            return data

        tasks: dict[str, Any] = {}
        for app in self.config.apps:
            tasks[app] = {
                "status": STATUS_PENDING,
                "attempts": 0,
                "last_error": "",
                "report_path": "",
                "started_at_utc": "",
                "finished_at_utc": "",
            }
        return {
            "schema_version": 1,
            "job_id": self.config.job_id,
            "mode": self.config.mode,
            "created_at_utc": _utc_now_iso(),
            "updated_at_utc": _utc_now_iso(),
            "apps": list(self.config.apps),
            "config": {
                "resume": self.config.resume,
                "force_rerun": self.config.force_rerun,
                "retry_count": self.config.retry_count,
                "retry_delay_sec": self.config.retry_delay_sec,
            },
            "tasks": tasks,
        }

    def save(self) -> None:
        self._manifest["updated_at_utc"] = _utc_now_iso()
        export_json_report(self._manifest, self.manifest_path)

    def should_skip(self, app_name: str) -> bool:
        if self.config.force_rerun:
            return False
        if not self.config.resume:
            return False
        task = self._manifest.get("tasks", {}).get(app_name, {})
        return task.get("status") == STATUS_SUCCESS

    def mark_running(self, app_name: str) -> None:
        task = self._manifest["tasks"][app_name]
        task["status"] = STATUS_RUNNING
        task["started_at_utc"] = _utc_now_iso()
        task["last_error"] = ""
        self.save()

    def mark_skipped(self, app_name: str, reason: str) -> None:
        task = self._manifest["tasks"][app_name]
        task["status"] = STATUS_SKIPPED
        task["finished_at_utc"] = _utc_now_iso()
        task["last_error"] = reason
        self.save()

    def mark_success(self, app_name: str, report: dict[str, Any]) -> None:
        task = self._manifest["tasks"][app_name]
        task["status"] = STATUS_SUCCESS
        task["finished_at_utc"] = _utc_now_iso()
        task["report_path"] = str(report.get("report_path", ""))
        task["last_error"] = ""
        self.save()

    def mark_failed(self, app_name: str, error: str, attempts: int) -> None:
        task = self._manifest["tasks"][app_name]
        task["status"] = STATUS_FAILED
        task["attempts"] = attempts
        task["last_error"] = error
        task["finished_at_utc"] = _utc_now_iso()
        self.save()

    def increment_attempt(self, app_name: str) -> int:
        task = self._manifest["tasks"][app_name]
        task["attempts"] = int(task.get("attempts", 0)) + 1
        self.save()
        return int(task["attempts"])

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest


def _archive_report(job_dir: Path, app_name: str, report: dict[str, Any]) -> str | None:
    report_path = report.get("report_path")
    if not report_path:
        return None
    src = Path(str(report_path))
    if not src.exists():
        return None
    archive_dir = job_dir / "reports" / app_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    dst = archive_dir / src.name
    shutil.copy2(src, dst)
    return str(dst)


def run_batch_job(
    config: BatchJobConfig,
    process_app: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Run batch with retry, resume, per-app logs, and manifest state."""
    job_id = config.job_id or _default_job_id()
    config.job_id = job_id
    job_dir = config.output_dir / "batch_jobs" / job_id
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    batch_logger = setup_file_logger("privacy_audit.batch", logs_dir / "batch.log")
    state = BatchStateManager(job_dir, config)
    reports: list[dict[str, Any]] = []
    task_results: list[dict[str, Any]] = []

    batch_logger.info(
        "Starting batch job_id=%s mode=%s apps=%d resume=%s force_rerun=%s",
        job_id,
        config.mode,
        len(config.apps),
        config.resume,
        config.force_rerun,
    )

    for app_name in config.apps:
        if state.should_skip(app_name):
            batch_logger.info("Skipping %s (already success, resume mode)", app_name)
            state.mark_skipped(app_name, "resume_skip_success")
            task_results.append(
                {
                    "app_name": app_name,
                    "status": STATUS_SKIPPED,
                    "message": "skipped_resume",
                }
            )
            continue

        app_logger = setup_file_logger(
            f"privacy_audit.batch.{app_name}",
            logs_dir / f"{app_name}.log",
        )
        state.mark_running(app_name)
        app_logger.info("Processing app=%s mode=%s", app_name, config.mode)

        attempts_used = 0
        last_error = ""

        def _run_once() -> dict[str, Any]:
            nonlocal attempts_used
            attempts_used = state.increment_attempt(app_name)
            app_logger.info("Attempt %d for %s", attempts_used, app_name)
            return process_app(app_name)

        def _on_retry(attempt: int, exc: Exception) -> None:
            app_logger.warning("Retry %d for %s failed: %s", attempt, app_name, exc)

        try:
            report = run_with_retry(
                _run_once,
                retries=max(0, config.retry_count),
                delay_sec=config.retry_delay_sec,
                backoff=config.retry_backoff,
                on_retry=_on_retry,
            )
            report_status = _infer_report_status(report)
            if report_status == STATUS_FAILED:
                err_msg = str(report.get("error", "report_status_failed"))
                state.mark_failed(app_name, err_msg, attempts_used)
                task_results.append(
                    {
                        "app_name": app_name,
                        "status": STATUS_FAILED,
                        "attempts": attempts_used,
                        "error": err_msg,
                    }
                )
                batch_logger.error("App %s failed (report status): %s", app_name, err_msg)
            else:
                archived = _archive_report(job_dir, app_name, report)
                if archived:
                    report["archived_report_path"] = archived
                state.mark_success(app_name, report)
                reports.append(report)
                task_results.append(
                    {
                        "app_name": app_name,
                        "status": STATUS_SUCCESS,
                        "attempts": attempts_used,
                        "report_path": report.get("report_path", ""),
                        "archived_report_path": archived or "",
                    }
                )
                app_logger.info("Success for %s report=%s", app_name, report.get("report_path", ""))
                batch_logger.info("Success for %s", app_name)
        except Exception as exc:  # pragma: no cover - runtime dependent
            last_error = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            app_logger.error("Failed for %s after %d attempts: %s\n%s", app_name, attempts_used, last_error, tb)
            state.mark_failed(app_name, last_error, attempts_used)
            task_results.append(
                {
                    "app_name": app_name,
                    "status": STATUS_FAILED,
                    "attempts": attempts_used,
                    "error": last_error,
                }
            )
            batch_logger.error("Failed for %s: %s", app_name, last_error)

    summary = _build_summary(config, state.manifest, task_results, reports)
    summary["job_id"] = job_id
    summary["job_dir"] = str(job_dir)
    summary["manifest_path"] = str(state.manifest_path)
    summary["logs_dir"] = str(logs_dir)
    summary["reports"] = reports
    summary["task_results"] = task_results

    summary_path = job_dir / "summary.json"
    export_json_report(summary, summary_path)
    summary["report_path"] = str(summary_path)

    # Also write a flat copy for quick access at output root.
    flat_summary = config.output_dir / f"batch_{config.mode}_{job_id}_summary.json"
    export_json_report(summary, flat_summary)
    summary["flat_summary_path"] = str(flat_summary)

    batch_logger.info(
        "Batch finished: success=%d failed=%d skipped=%d",
        summary["summary"]["success_count"],
        summary["summary"]["failed_count"],
        summary["summary"]["skipped_count"],
    )
    return summary


def _build_summary(
    config: BatchJobConfig,
    manifest: dict[str, Any],
    task_results: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    success_count = sum(1 for t in task_results if t.get("status") == STATUS_SUCCESS)
    failed_count = sum(1 for t in task_results if t.get("status") == STATUS_FAILED)
    skipped_count = sum(1 for t in task_results if t.get("status") == STATUS_SKIPPED)
    pending_count = sum(
        1 for t in manifest.get("tasks", {}).values() if t.get("status") == STATUS_PENDING
    )

    overall = "completed"
    if failed_count > 0 and success_count > 0:
        overall = "partial"
    elif failed_count > 0 and success_count == 0:
        overall = "failed"

    return {
        "schema_version": 1,
        "stage": "stage7_batch",
        "mode": config.mode,
        "apps": config.apps,
        "status": overall,
        "generated_at_utc": _utc_now_iso(),
        "summary": {
            "total_apps": len(config.apps),
            "success_count": success_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "pending_count": pending_count,
            "reports_count": len(reports),
        },
        "manifest": manifest,
    }


def run_batch(apps: list[str], mode: str = "lab4-1", reports: list[dict[str, Any]] | None = None) -> dict:
    """Backward-compatible summary builder (legacy callers)."""
    reports = reports or []
    pass_count = 0
    review_count = 0
    todo_count = 0
    for report in reports:
        status = str(report.get("status", "")).lower()
        if status in {"pass", "consistent", "completed"}:
            pass_count += 1
        elif status in {"todo", "pending"}:
            todo_count += 1
        else:
            review_count += 1
    return {
        "apps": apps,
        "mode": mode,
        "status": "completed",
        "total_apps": len(apps),
        "reports_count": len(reports),
        "pass_count": pass_count,
        "review_count": review_count,
        "todo_count": todo_count,
    }


def load_batch_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load a batch manifest for status inspection."""
    if not manifest_path.exists():
        return {"error": f"manifest not found: {manifest_path}"}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def find_latest_job_dir(output_dir: Path, mode: str | None = None) -> Path | None:
    """Find the most recently updated batch job directory."""
    root = output_dir / "batch_jobs"
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if mode:
        filtered: list[Path] = []
        for path in candidates:
            manifest = path / "manifest.json"
            if manifest.exists():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if data.get("mode") == mode:
                    filtered.append(path)
        candidates = filtered or candidates
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
