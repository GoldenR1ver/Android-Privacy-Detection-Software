"""Stage 3 pipeline: dynamic evidence (Appium + Frida)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from privacy_audit.analyzers.dynamic.appium_runner import run_ui_flow
from privacy_audit.analyzers.dynamic.frida_runner import run_frida_hooks
from privacy_audit.analyzers.dynamic.frida_sync import FridaCaptureSession
from privacy_audit.analyzers.dynamic.app_launch import adb_launch_package
from privacy_audit.analyzers.dynamic.llm_path_assistant import (
    build_evidence_explanation,
    build_path_suggestions,
)
from privacy_audit.analyzers.policy.policy_parser import load_personal_info_taxonomy
from privacy_audit.reporting.report_exporter import export_json_report
from privacy_audit.utils.progress_log import log_step
from privacy_audit.utils.runtime_preflight import run_stage3_preflight

logger = logging.getLogger("privacy_audit.stage3")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_candidate_pii(stage2_evidence_file: Path | None) -> list[str]:
    if stage2_evidence_file is None or not stage2_evidence_file.exists():
        return []
    data = json.loads(stage2_evidence_file.read_text(encoding="utf-8"))
    observed = data.get("observed_pii", [])
    return sorted(set(observed))


def _read_text_if_exists(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _build_dynamic_evidence_items(
    app_name: str,
    ui_result: dict[str, Any],
    frida_result: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for shot in ui_result.get("screenshots", []):
        items.append(
            {
                "app_name": app_name,
                "source": "dynamic.ui",
                "signal": "screenshot",
                "value": shot,
                "timestamp_utc": ui_result.get("timestamp_utc", _utc_now_iso()),
                "evidence_fields": ["ui_result.screenshots"],
            }
        )
    for event in frida_result.get("events", []):
        items.append(
            {
                "app_name": app_name,
                "source": "dynamic.frida",
                "signal": event.get("api_name", ""),
                "value": event.get("args_summary", ""),
                "stack_summary": event.get("stack_summary", ""),
                "timestamp_utc": event.get("timestamp_utc", _utc_now_iso()),
                "pii_type": event.get("pii_type", "unknown"),
                "hook_file": event.get("hook_file", ""),
                "evidence_fields": ["frida_result.events"],
            }
        )
    return items


def run_stage3_dynamic(
    app_name: str,
    app_package: str,
    taxonomy_file: Path,
    pii_text_file: Path | None,
    output_dir: Path,
    stage2_evidence_file: Path | None = None,
    appium_server_url: str = "http://127.0.0.1:4723",
    dry_run: bool = True,
    duration_sec: int = 8,
    frida_launch_mode: str = "attach",
    frida_grace_sec: int = 5,
    frida_tail_sec: int = 3,
    appium_no_reset: bool = True,
    skip_preflight: bool = False,
    appium_connect_timeout_sec: float = 5.0,
) -> dict[str, Any]:
    """Run dynamic evidence pipeline for one app."""
    logger.info(
        "Stage3 start app=%s package=%s dry_run=%s output_dir=%s",
        app_name,
        app_package,
        dry_run,
        output_dir,
    )
    preflight: dict[str, Any] | None = None
    if not dry_run and not skip_preflight:
        with log_step(logger, "Preflight (adb / Appium / Frida)"):
            preflight = run_stage3_preflight(appium_server_url)
            if not preflight.get("ok"):
                logger.warning(
                    "Preflight reported issues; continuing anyway. "
                    "Fix adb/Appium/frida-server or pass --skip-preflight to silence."
                )

    taxonomy = load_personal_info_taxonomy(taxonomy_file)
    policy_text = _read_text_if_exists(pii_text_file)
    candidate_pii = _load_candidate_pii(stage2_evidence_file)
    if not candidate_pii:
        candidate_pii = list(taxonomy.keys())[:6]
    logger.info("candidate_pii=%s policy_chars=%d", candidate_pii, len(policy_text))

    app_output = output_dir / app_name
    screenshot_dir = app_output / "screenshots"
    hooks_dir = Path(__file__).resolve().parents[1] / "analyzers" / "dynamic" / "hooks"

    capture_meta: dict[str, Any] = {
        "sync_mode": not dry_run,
        "frida_grace_sec": frida_grace_sec,
        "frida_tail_sec": frida_tail_sec,
    }

    if dry_run:
        with log_step(logger, "Appium UI flow (dry run)"):
            ui_result = run_ui_flow(
                app_name=app_name,
                policy_text=policy_text,
                appium_server_url=appium_server_url,
                app_package=app_package,
                screenshot_dir=screenshot_dir,
                dry_run=True,
                no_reset=appium_no_reset,
                connect_timeout_sec=appium_connect_timeout_sec,
            )
        with log_step(logger, "Frida hooks (dry run)"):
            frida_result = run_frida_hooks(
                app_package=app_package,
                candidate_pii=candidate_pii,
                taxonomy=taxonomy,
                hooks_dir=hooks_dir,
                duration_sec=duration_sec,
                dry_run=True,
                launch_mode=frida_launch_mode,
                app_name=app_name,
            )
    else:
        frida_session = FridaCaptureSession(
            app_package=app_package,
            candidate_pii=candidate_pii,
            taxonomy=taxonomy,
            hooks_dir=hooks_dir,
            launch_mode=frida_launch_mode,
            app_name=app_name,
        )
        with log_step(logger, "Synchronized Frida + Appium capture"):
            if not dry_run and app_package:
                logger.info("Pre-launch target app via adb before Frida/Appium: %s", app_package)
                adb_launch_package(app_package)
            if not frida_session.start_capture():
                frida_result = frida_session.stop_capture()
                ui_result = run_ui_flow(
                    app_name=app_name,
                    policy_text=policy_text,
                    appium_server_url=appium_server_url,
                    app_package=app_package,
                    screenshot_dir=screenshot_dir,
                    dry_run=False,
                    no_reset=appium_no_reset,
                    connect_timeout_sec=appium_connect_timeout_sec,
                )
            else:
                frida_session.wait_manual_grace(frida_grace_sec)
                logger.info("Starting Appium while Frida is recording...")
                ui_result = run_ui_flow(
                    app_name=app_name,
                    policy_text=policy_text,
                    appium_server_url=appium_server_url,
                    app_package=app_package,
                    screenshot_dir=screenshot_dir,
                    dry_run=False,
                    no_reset=appium_no_reset,
                    connect_timeout_sec=appium_connect_timeout_sec,
                )
                frida_session.wait_tail(frida_tail_sec)
                frida_result = frida_session.stop_capture()
        capture_meta["frida_runtime_events"] = frida_result.get("runtime_event_count", 0)

    suggestions = build_path_suggestions(
        policy_text=policy_text,
        candidate_pii=candidate_pii,
        ui_steps=ui_result.get("steps", []),
    )
    explanation = build_evidence_explanation(
        app_name=app_name,
        candidate_pii=candidate_pii,
        ui_result=ui_result,
        frida_result=frida_result,
    )
    evidence_items = _build_dynamic_evidence_items(app_name, ui_result, frida_result)

    result = {
        "schema_version": 1,
        "stage": "stage3_dynamic",
        "app_name": app_name,
        "app_package": app_package,
        "dry_run": dry_run,
        "generated_at_utc": _utc_now_iso(),
        "candidate_pii": candidate_pii,
        "ui_result": ui_result,
        "frida_result": frida_result,
        "path_suggestions": suggestions,
        "explanation": explanation,
        "evidence_items": evidence_items,
        "capture_meta": capture_meta,
        "summary": {
            "ui_steps": len(ui_result.get("steps", [])),
            "screenshots": len(ui_result.get("screenshots", [])),
            "keyword_checks": len(ui_result.get("keyword_checks", [])),
            "keyword_match_count": sum(
                1 for row in ui_result.get("keyword_checks", []) if row.get("matched")
            ),
            "registered_hook_types": len(frida_result.get("registered_hooks", {})),
            "frida_events": len(frida_result.get("events", [])),
            "frida_runtime_events": frida_result.get("runtime_event_count", 0),
            "candidate_pii_count": len(candidate_pii),
        },
    }
    out_file = app_output / "dynamic_evidence.json"
    export_json_report(result, out_file)
    result["report_path"] = str(out_file)
    if preflight is not None:
        result["preflight"] = preflight
    logger.info(
        "Stage3 complete report=%s ui_error=%s frida_error=%s frida_events=%d",
        out_file,
        ui_result.get("error") or "(none)",
        frida_result.get("error") or "(none)",
        len(frida_result.get("events", [])),
    )
    return result


def _default_taxonomy_file() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "personal_info.yaml"


def _resolve_app_packages_file(path: str | None) -> Path:
    if path:
        return Path(path)
    for candidate in (
        Path("input/app_packages.json"),
        Path(__file__).resolve().parents[2] / "input" / "app_packages.json",
    ):
        if candidate.exists():
            return candidate
    return Path("input/app_packages.json")


def _load_app_package(app_name: str, packages_file: Path) -> str:
    if not packages_file.exists():
        raise FileNotFoundError(
            f"App package map not found: {packages_file}. "
            "Pass --app-package or create input/app_packages.json"
        )
    data = json.loads(packages_file.read_text(encoding="utf-8"))
    if app_name not in data:
        known = ", ".join(sorted(data.keys())[:12])
        raise KeyError(f"Unknown app {app_name!r} in {packages_file}. Known: {known}...")
    return str(data[app_name])


def _parse_dry_run_flag(dry_run_opt: str | None, real_run: bool) -> bool:
    """Return True for simulation (no Appium/Frida on device)."""
    if real_run:
        return False
    if dry_run_opt is None:
        return True
    if dry_run_opt.lower() in {"false", "0", "no", "off"}:
        return False
    if dry_run_opt.lower() in {"true", "1", "yes", "on"}:
        return True
    return True


def main(argv: list[str] | None = None) -> int:
    """
    Convenience CLI for `python -m privacy_audit.workflows.stage3_dynamic_pipeline`.

    Prefer the full entrypoint:
      python -m privacy_audit.cli -v stage3-dynamic --app-name ... --app-package ... --real-run
    """
    import argparse
    import sys

    from privacy_audit.utils.logger import configure_cli_logging

    parser = argparse.ArgumentParser(
        description="Stage3 dynamic evidence (Appium + Frida). "
        "Wrapper around run_stage3_dynamic; use privacy_audit.cli for full options.",
    )
    parser.add_argument(
        "--app",
        "--app-name",
        dest="app_name",
        required=True,
        help="App display name (Chinese OK), e.g. Alipay app name in app_packages.json",
    )
    parser.add_argument("--app-package", default="", help="Android package (default: input/app_packages.json)")
    parser.add_argument(
        "--app-packages-file",
        default="",
        help="JSON map of app name -> package (default: input/app_packages.json)",
    )
    parser.add_argument(
        "--pii-text-dir",
        default="input/pii_text",
        help="Policy text directory",
    )
    parser.add_argument(
        "--output-dir",
        default="output/stage3_dynamic_real",
        help="Output directory",
    )
    parser.add_argument(
        "--taxonomy-file",
        default=str(_default_taxonomy_file()),
        help="personal_info.yaml path",
    )
    parser.add_argument("--appium-server-url", default="http://127.0.0.1:4723")
    parser.add_argument(
        "--real-run",
        action="store_true",
        help="Run real Appium/Frida on device (same as cli stage3-dynamic --real-run)",
    )
    parser.add_argument(
        "--dry-run",
        nargs="?",
        const="true",
        default=None,
        metavar="BOOL",
        help="Simulation only (default). Use --dry-run false or --real-run for device run.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logs to stdout")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--frida-mode", default="attach", choices=["attach", "spawn"])
    parser.add_argument("--frida-grace-sec", type=int, default=5)
    parser.add_argument("--frida-tail-sec", type=int, default=3)
    args = parser.parse_args(argv)

    dry_run = _parse_dry_run_flag(args.dry_run, args.real_run)
    output_dir = Path(args.output_dir)
    log_file = output_dir / args.app_name / "stage3_run.log"
    configure_cli_logging(verbose=args.verbose, log_file=log_file)

    packages_file = _resolve_app_packages_file(args.app_packages_file or None)
    app_package = args.app_package.strip() or _load_app_package(args.app_name, packages_file)

    pii_text_dir = Path(args.pii_text_dir)
    pii_text_file: Path | None = None
    candidate = pii_text_dir / f"{args.app_name}.txt"
    if candidate.exists():
        pii_text_file = candidate

    logger.info(
        "stage3_dynamic_pipeline CLI app=%s package=%s dry_run=%s log=%s",
        args.app_name,
        app_package,
        dry_run,
        log_file,
    )
    if dry_run:
        logger.warning(
            "dry_run=True: no Appium/Frida on emulator. Use --real-run or --dry-run false."
        )

    result = run_stage3_dynamic(
        app_name=args.app_name,
        app_package=app_package,
        taxonomy_file=Path(args.taxonomy_file),
        pii_text_file=pii_text_file,
        output_dir=output_dir,
        appium_server_url=args.appium_server_url,
        dry_run=dry_run,
        frida_launch_mode=args.frida_mode,
        frida_grace_sec=args.frida_grace_sec,
        frida_tail_sec=args.frida_tail_sec,
        skip_preflight=args.skip_preflight,
    )
    print(json.dumps({"report_path": result.get("report_path"), "dry_run": dry_run}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
