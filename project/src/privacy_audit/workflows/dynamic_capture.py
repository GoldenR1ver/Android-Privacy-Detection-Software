"""Dynamic capture pipeline: runtime evidence from Appium + Frida + mitm."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from privacy_audit.analyzers.dynamic.appium_runner import run_ui_flow
from privacy_audit.analyzers.dynamic.frida_runner import (
    get_app_frida_tier_entry,
    load_app_frida_config,
    parse_app_package_entry,
    resolve_frida_launch_mode,
    run_frida_hooks,
)
from privacy_audit.analyzers.dynamic.frida_sync import FridaCaptureSession
from privacy_audit.analyzers.dynamic.app_launch import (
    adb_close_after_dynamic_verification,
    adb_launch_package,
)
from privacy_audit.analyzers.dynamic.llm_path_assistant import (
    build_evidence_explanation,
    build_path_suggestions,
)
from privacy_audit.analyzers.network.mitm_runner import MitmCaptureSession
from privacy_audit.analyzers.network.traffic_checker import build_network_evidence
from privacy_audit.analyzers.policy.policy_parser import load_personal_info_taxonomy
from privacy_audit.reporting.report_exporter import export_json_report
from privacy_audit.utils.progress_log import log_step
from privacy_audit.utils.runtime_preflight import run_capture_preflight
from privacy_audit.workflows.capture_track import resolve_capture_tracks

logger = logging.getLogger("privacy_audit.dynamic_capture")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_candidate_pii(static_evidence_file: Path | None) -> list[str]:
    if static_evidence_file is None or not static_evidence_file.exists():
        return []
    data = json.loads(static_evidence_file.read_text(encoding="utf-8"))
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
    network_result: dict[str, Any] | None = None,
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
    network_result = network_result or {}
    for hit in network_result.get("tracker_hits") or []:
        if not isinstance(hit, dict):
            continue
        items.append(
            {
                "app_name": app_name,
                "source": "dynamic.network",
                "signal": "tracker_host",
                "value": hit.get("host", ""),
                "pii_type": "sdk_info",
                "timestamp_utc": hit.get("timestamp_utc", _utc_now_iso()),
                "context": hit.get("tracker_company", ""),
                "evidence_fields": ["network_result.tracker_hits"],
            }
        )
    for hit in network_result.get("pii_key_hits") or []:
        if not isinstance(hit, dict):
            continue
        items.append(
            {
                "app_name": app_name,
                "source": "dynamic.network",
                "signal": "pii_network_key",
                "value": hit.get("network_key", ""),
                "pii_type": hit.get("pii_type", "unknown"),
                "timestamp_utc": hit.get("timestamp_utc", _utc_now_iso()),
                "evidence_fields": ["network_result.pii_key_hits"],
            }
        )
    return items


def run_dynamic_capture(
    app_name: str,
    app_package: str,
    taxonomy_file: Path,
    policy_text_file: Path | None,
    output_dir: Path,
    static_evidence_file: Path | None = None,
    appium_server_url: str = "http://127.0.0.1:4723",
    dry_run: bool = True,
    duration_sec: int = 8,
    frida_launch_mode: str = "attach",
    frida_grace_sec: int = 5,
    frida_tail_sec: int = 3,
    code_prefixes: list[str] | None = None,
    appium_no_reset: bool = True,
    skip_preflight: bool = False,
    appium_connect_timeout_sec: float = 5.0,
    navigation_bridges_file: Path | None = None,
    enable_llm_nav: bool | None = None,
    llm_model: str = "deepseek-chat",
    llm_base_url: str = "https://api.deepseek.com",
    tap_attempt_timeout_sec: float = 40.0,
    fallback_budget_sec: float = 3.0,
    llm_nav_timeout_sec: float = 20.0,
    enable_frida: bool = False,
    enable_mitm: bool = False,
    dynamic_track: str = "auto",
    frida_tier_file: Path | None = None,
    spawn_fallback_attach: bool | None = None,
    frida_options: dict[str, Any] | None = None,
    probe_dir: Path | None = None,
    tracker_blocklist_dir: Path | None = None,
    mitm_port: int = 8080,
    mitm_adb_proxy_host: str = "10.0.2.2",
    mitm_flows_file: Path | None = None,
    adb_serial: str | None = None,
) -> dict[str, Any]:
    """Run dynamic evidence pipeline for one app."""
    track_plan = resolve_capture_tracks(
        app_name,
        dynamic_track=dynamic_track,
        enable_frida=enable_frida,
        enable_mitm=enable_mitm,
        probe_dir=probe_dir,
        frida_tier_file=frida_tier_file,
    )
    enable_frida = bool(track_plan.get("enable_frida"))
    enable_mitm = bool(track_plan.get("enable_mitm"))

    frida_launch_mode = resolve_frida_launch_mode(
        app_name,
        frida_tier_file,
        frida_launch_mode,
    )
    tier_entry: dict[str, Any] = {}
    if frida_tier_file is not None and frida_tier_file.exists():
        tier_entry = get_app_frida_tier_entry(app_name, frida_tier_file)
    if spawn_fallback_attach is None:
        spawn_fallback_attach = bool(tier_entry.get("spawn_fallback_attach", True))

    logger.info(
        "Dynamic capture start app=%s package=%s dry_run=%s frida_mode=%s output_dir=%s",
        app_name,
        app_package,
        dry_run,
        frida_launch_mode,
        output_dir,
    )
    preflight: dict[str, Any] | None = None
    if not dry_run and not skip_preflight:
        with log_step(logger, "Preflight (adb / Appium / Frida)"):
            preflight = run_capture_preflight(appium_server_url)
            if not preflight.get("ok"):
                logger.warning(
                    "Preflight reported issues; continuing anyway. "
                    "Fix adb/Appium/frida-server or pass --skip-preflight to silence."
                )

    taxonomy = load_personal_info_taxonomy(taxonomy_file)
    policy_text = _read_text_if_exists(policy_text_file)
    candidate_pii = _load_candidate_pii(static_evidence_file)
    if not candidate_pii:
        # No static evidence: load every PII type that has a hook or api_targets
        # (so contacts/clipboard/sdk_info also get instrumented, not just the
        # first 6 keys). Hookless types are skipped by build_hook_registry anyway.
        candidate_pii = [
            pii
            for pii, meta in taxonomy.items()
            if isinstance(meta, dict) and (meta.get("hooks") or meta.get("api_targets"))
        ]
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
        "frida_launch_mode": frida_launch_mode,
        "frida_code_prefixes": code_prefixes or [],
        "frida_tier": tier_entry.get("tier", ""),
        "spawn_fallback_attach": spawn_fallback_attach,
        "dynamic_track": track_plan,
        "enable_mitm": enable_mitm,
    }

    mitm_session: MitmCaptureSession | None = None
    mitm_result: dict[str, Any] = {}
    network_result: dict[str, Any] = {}
    mitm_flows_out = app_output / "mitm_flows.json"

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
                navigation_bridges_file=navigation_bridges_file,
                enable_llm_nav=enable_llm_nav,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                tap_attempt_timeout_sec=tap_attempt_timeout_sec,
                fallback_budget_sec=fallback_budget_sec,
                llm_nav_timeout_sec=llm_nav_timeout_sec,
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
                code_prefixes=code_prefixes,
            )
        if enable_mitm:
            mitm_session = MitmCaptureSession(
                mitm_flows_out,
                listen_port=mitm_port,
                adb_proxy_host=mitm_adb_proxy_host,
                dry_run=True,
                replay_file=mitm_flows_file,
                set_device_proxy=False,
                adb_serial=adb_serial,
            )
            mitm_session.start()
            mitm_result = mitm_session.stop()
            network_result = build_network_evidence(
                mitm_result,
                taxonomy_file=taxonomy_file,
                blocklist_dir=tracker_blocklist_dir,
            )
    elif not enable_frida:
        logger.info(
            "Frida disabled (dynamic_track=%s mitm=%s)",
            track_plan.get("dynamic_track_resolved"),
            enable_mitm,
        )
        capture_meta["frida_enabled"] = False
        if enable_mitm:
            mitm_session = MitmCaptureSession(
                mitm_flows_out,
                listen_port=mitm_port,
                adb_proxy_host=mitm_adb_proxy_host,
                dry_run=False,
                replay_file=mitm_flows_file,
                adb_serial=adb_serial,
            )
            if not mitm_session.start():
                mitm_result = {
                    "adapter": "mitm_runner",
                    "error": "mitm_start_failed",
                    "flows": [],
                    "flow_count": 0,
                }
        if app_package:
            adb_launch_package(app_package)
        with log_step(logger, "Appium UI flow (Frida off)"):
            ui_result = run_ui_flow(
                app_name=app_name,
                policy_text=policy_text,
                appium_server_url=appium_server_url,
                app_package=app_package,
                screenshot_dir=screenshot_dir,
                dry_run=False,
                no_reset=appium_no_reset,
                connect_timeout_sec=appium_connect_timeout_sec,
                navigation_bridges_file=navigation_bridges_file,
                enable_llm_nav=enable_llm_nav,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                tap_attempt_timeout_sec=tap_attempt_timeout_sec,
                fallback_budget_sec=fallback_budget_sec,
                llm_nav_timeout_sec=llm_nav_timeout_sec,
            )
        if mitm_session is not None:
            with log_step(logger, "mitmproxy stop + network evidence"):
                mitm_result = mitm_session.stop()
                network_result = build_network_evidence(
                    mitm_result,
                    taxonomy_file=taxonomy_file,
                    blocklist_dir=tracker_blocklist_dir,
                )
        with log_step(logger, "Frida hooks (skipped)"):
            frida_result = run_frida_hooks(
                app_package=app_package,
                candidate_pii=candidate_pii,
                taxonomy=taxonomy,
                hooks_dir=hooks_dir,
                duration_sec=duration_sec,
                dry_run=True,
                launch_mode=frida_launch_mode,
                app_name=app_name,
                code_prefixes=code_prefixes,
            )
            frida_result = {
                **frida_result,
                "skipped": True,
                "error": "frida_disabled",
            }
        capture_meta["frida_runtime_events"] = 0
        capture_meta["frida_java_active_count"] = 0
        capture_meta["mitm_flow_count"] = mitm_result.get("flow_count", 0)
        capture_meta["tracker_hit_count"] = (network_result.get("summary") or {}).get(
            "tracker_hit_count", 0
        )
    else:
        capture_meta["frida_enabled"] = True
        frida_session = FridaCaptureSession(
            app_package=app_package,
            candidate_pii=candidate_pii,
            taxonomy=taxonomy,
            hooks_dir=hooks_dir,
            launch_mode=frida_launch_mode,
            app_name=app_name,
            code_prefixes=code_prefixes,
            spawn_fallback_attach=bool(spawn_fallback_attach),
            frida_options=frida_options or {},
        )
        if enable_mitm:
            mitm_session = MitmCaptureSession(
                mitm_flows_out,
                listen_port=mitm_port,
                adb_proxy_host=mitm_adb_proxy_host,
                dry_run=False,
                replay_file=mitm_flows_file,
                adb_serial=adb_serial,
            )
            mitm_session.start()
        with log_step(logger, "Synchronized Frida + Appium capture"):
            if not dry_run and app_package and frida_launch_mode != "spawn":
                logger.info("Pre-launch target app via adb before Frida/Appium: %s", app_package)
                adb_launch_package(app_package)
            elif frida_launch_mode == "spawn":
                logger.info(
                    "Spawn mode: skipping adb pre-launch (Frida will cold-spawn %s)",
                    app_package,
                )
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
                    navigation_bridges_file=navigation_bridges_file,
                    enable_llm_nav=enable_llm_nav,
                    llm_model=llm_model,
                    llm_base_url=llm_base_url,
                    tap_attempt_timeout_sec=tap_attempt_timeout_sec,
                    fallback_budget_sec=fallback_budget_sec,
                    llm_nav_timeout_sec=llm_nav_timeout_sec,
                    preserve_frida_session=True,
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
                    navigation_bridges_file=navigation_bridges_file,
                    enable_llm_nav=enable_llm_nav,
                    llm_model=llm_model,
                    llm_base_url=llm_base_url,
                    tap_attempt_timeout_sec=tap_attempt_timeout_sec,
                    fallback_budget_sec=fallback_budget_sec,
                    llm_nav_timeout_sec=llm_nav_timeout_sec,
                    preserve_frida_session=True,
                )
                frida_session.wait_tail(frida_tail_sec)
                frida_result = frida_session.stop_capture()
        capture_meta["frida_runtime_events"] = frida_result.get("runtime_event_count", 0)
        capture_meta["frida_java_active_count"] = frida_result.get("java_active_count", 0)
        if mitm_session is not None:
            mitm_result = mitm_session.stop()
            network_result = build_network_evidence(
                mitm_result,
                taxonomy_file=taxonomy_file,
                blocklist_dir=tracker_blocklist_dir,
            )
            capture_meta["mitm_flow_count"] = mitm_result.get("flow_count", 0)
            capture_meta["tracker_hit_count"] = (network_result.get("summary") or {}).get(
                "tracker_hit_count", 0
            )

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
    evidence_items = _build_dynamic_evidence_items(
        app_name, ui_result, frida_result, network_result
    )

    if not dry_run and app_package:
        with log_step(logger, "Close app after dynamic verification"):
            capture_meta["post_run_app_closed"] = adb_close_after_dynamic_verification(
                app_package,
                adb_serial=adb_serial,
            )

    schema_version = 2 if network_result else 1
    result = {
        "schema_version": schema_version,
        "pipeline": "dynamic_capture",
        "app_name": app_name,
        "app_package": app_package,
        "dry_run": dry_run,
        "generated_at_utc": _utc_now_iso(),
        "candidate_pii": candidate_pii,
        "ui_result": ui_result,
        "frida_result": frida_result,
        "mitm_result": mitm_result,
        "network_result": network_result,
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
            "frida_java_active_count": frida_result.get("java_active_count", 0),
            "candidate_pii_count": len(candidate_pii),
            "mitm_flow_count": mitm_result.get("flow_count", 0),
            "tracker_hit_count": (network_result.get("summary") or {}).get(
                "tracker_hit_count", 0
            ),
            "pii_key_hit_count": (network_result.get("summary") or {}).get(
                "pii_key_hit_count", 0
            ),
        },
    }
    out_file = app_output / "dynamic_evidence.json"
    export_json_report(result, out_file)
    if network_result:
        export_json_report(network_result, app_output / "network_evidence.json")
    result["report_path"] = str(out_file)
    if preflight is not None:
        result["preflight"] = preflight
    logger.info(
        "Dynamic capture complete report=%s ui_error=%s frida_error=%s frida_events=%d",
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
    package, _prefixes, _opts = load_app_frida_config(app_name, packages_file)
    return package


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
    Convenience CLI for `python -m privacy_audit.workflows.dynamic_capture`.

    Prefer the full entrypoint:
      python -m privacy_audit.cli -v capture-dynamic --app-name ... --app-package ... --real-run
    """
    import argparse
    import sys

    from privacy_audit.utils.logger import configure_cli_logging

    parser = argparse.ArgumentParser(
        description="Dynamic capture (Appium + Frida + mitm). "
        "Wrapper around run_dynamic_capture; use privacy_audit.cli for full options.",
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
        "--policy-text-dir",
        default="input/pii_text",
        help="Policy text directory",
    )
    parser.add_argument(
        "--output-dir",
        default="output/dynamic_capture",
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
        help="Run real Appium/Frida on device (same as cli capture-dynamic --real-run)",
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
    parser.add_argument(
        "--frida-mode",
        default="auto",
        choices=["auto", "attach", "spawn"],
    )
    parser.add_argument(
        "--frida-tier-file",
        default="input/frida_tier.json",
        help="Per-app Frida tier / spawn preferences",
    )
    parser.add_argument("--frida-grace-sec", type=int, default=5)
    parser.add_argument("--frida-tail-sec", type=int, default=3)
    args = parser.parse_args(argv)

    dry_run = _parse_dry_run_flag(args.dry_run, args.real_run)
    output_dir = Path(args.output_dir)
    log_file = output_dir / args.app_name / "capture_run.log"
    configure_cli_logging(verbose=args.verbose, log_file=log_file)

    packages_file = _resolve_app_packages_file(args.app_packages_file or None)
    frida_options: dict[str, Any] = {}
    if args.app_package.strip():
        app_package = args.app_package.strip()
        _pkg, code_prefixes, frida_options = parse_app_package_entry(
            {"package": app_package}
        )
    else:
        app_package, code_prefixes, frida_options = load_app_frida_config(
            args.app_name, packages_file
        )

    pii_text_dir = Path(args.pii_text_dir)
    pii_text_file: Path | None = None
    candidate = pii_text_dir / f"{args.app_name}.txt"
    if candidate.exists():
        pii_text_file = candidate

    logger.info(
        "dynamic_capture CLI app=%s package=%s dry_run=%s log=%s",
        args.app_name,
        app_package,
        dry_run,
        log_file,
    )
    if dry_run:
        logger.warning(
            "dry_run=True: no Appium/Frida on emulator. Use --real-run or --dry-run false."
        )

    tier_file = Path(args.frida_tier_file)
    if not tier_file.is_absolute():
        tier_file = Path(__file__).resolve().parents[2] / tier_file

    result = run_dynamic_capture(
        app_name=args.app_name,
        app_package=app_package,
        taxonomy_file=Path(args.taxonomy_file),
        policy_text_file=policy_text_file,
        output_dir=output_dir,
        appium_server_url=args.appium_server_url,
        dry_run=dry_run,
        frida_launch_mode=resolve_frida_launch_mode(args.app_name, tier_file, args.frida_mode),
        frida_grace_sec=args.frida_grace_sec,
        frida_tail_sec=args.frida_tail_sec,
        code_prefixes=code_prefixes,
        skip_preflight=args.skip_preflight,
        frida_tier_file=tier_file,
        frida_options=frida_options,
    )
    print(json.dumps({"report_path": result.get("report_path"), "dry_run": dry_run}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



