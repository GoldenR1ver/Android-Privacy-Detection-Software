"""CLI entrypoint for privacy audit pipelines."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from privacy_audit.utils.logger import configure_cli_logging

from privacy_audit.ingestion.legacy_bridge import import_claims_from_review_bundle
from privacy_audit.ingestion.unknown_label_refiner import refine_unknown_labels
from privacy_audit.reporting.report_exporter import export_json_report
from privacy_audit.workflows.batch_pipeline import (
    BatchJobConfig,
    find_latest_job_dir,
    load_batch_manifest,
    run_batch_job,
)
from privacy_audit.ingestion.evidence_loader import (
    build_and_export_evidence_bundles,
    list_legacy_dynamic_apps,
)
from privacy_audit.ingestion.oldsrc_evidence_importer import import_oldsrc_pipeline
from privacy_audit.workflows.lab4_1_pipeline import run_lab4_1
from privacy_audit.workflows.lab4_2_pipeline import run_lab4_2
from privacy_audit.workflows.lab4_3_pipeline import run_lab4_3
from privacy_audit.workflows.stage3_dynamic_pipeline import run_stage3_dynamic
from privacy_audit.workflows.stage2_static_pipeline import (
    run_stage2_for_apk,
    run_stage2_for_directory,
)


def _default_taxonomy_file() -> Path:
    return Path(__file__).resolve().parent / "config" / "personal_info.yaml"


def _default_rules_file() -> Path:
    return Path(__file__).resolve().parent / "rules" / "decision_rules.yaml"


def _default_legacy_mapping_file() -> Path:
    return Path(__file__).resolve().parent / "config" / "legacy_label_mapping.yaml"


def _default_legacy_run_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "oldsrc" / "analyzer" / "output" / "run_20260427_001723"


def _default_legacy_md_run_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "oldsrc" / "analyzer" / "output" / "run_20260427_001723z"


def _path_arg(value: str | None) -> Path | None:
    return Path(value) if value else None


def _cmd_lab4_1(args: argparse.Namespace) -> dict[str, Any]:
    log = logging.getLogger("privacy_audit.cli")
    log.info(
        "lab4-1 app=%s policy=%s evidence=%s",
        args.app_name,
        args.policy_file,
        args.evidence_file or args.evidence_dir or "(merge)",
    )
    return run_lab4_1(
        app_name=args.app_name,
        policy_file=Path(args.policy_file),
        output_dir=Path(args.output_dir),
        taxonomy_file=Path(args.taxonomy_file),
        rules_file=Path(args.rules_file),
        evidence_file=_path_arg(args.evidence_file),
        evidence_dir=_path_arg(args.evidence_dir),
        static_evidence_dir=_path_arg(args.static_evidence_dir),
        legacy_run_dir=_path_arg(args.legacy_run_dir),
        legacy_md_run_dir=_path_arg(args.legacy_md_run_dir),
    )


def _cmd_build_evidence_bundles(args: argparse.Namespace) -> dict[str, Any]:
    logging.getLogger("privacy_audit.cli").info("build-evidence-bundles output=%s", args.output_dir)
    if args.apps_file:
        apps = _resolve_batch_apps(argparse.Namespace(apps_file=args.apps_file, apps=""))
    else:
        apps = [name.strip() for name in args.apps.split(",") if name.strip()]
    legacy_run = _path_arg(args.legacy_run_dir) or _default_legacy_run_dir()
    if not apps:
        apps = list_legacy_dynamic_apps(legacy_run)
    dynamic_dir = _path_arg(args.dynamic_evidence_dir) or Path("output")
    return build_and_export_evidence_bundles(
        app_names=apps,
        output_dir=Path(args.output_dir),
        taxonomy_file=Path(args.taxonomy_file),
        evidence_dir=dynamic_dir,
        static_dir=_path_arg(args.static_evidence_dir) or Path("output/stage2_static_samples"),
        legacy_run_dir=legacy_run,
        legacy_md_run_dir=_path_arg(args.legacy_md_run_dir) or _default_legacy_md_run_dir(),
        use_knowledge_base=not bool(getattr(args, "no_knowledge_base", False)),
        kb_min_confidence=float(getattr(args, "kb_min_confidence", 0.72)),
    )


def _cmd_kb_build(args: argparse.Namespace) -> dict[str, Any]:
    from privacy_knowledgebase.process_input import generate_outputs

    kb_dir = Path(__file__).resolve().parents[1] / "privacy_knowledgebase"
    input_path = Path(args.input) if args.input else kb_dir / "input.md"
    output_dir = Path(args.output_dir) if args.output_dir else kb_dir / "knowledge_base"

    pdf_ingest: dict[str, Any] | None = None
    if getattr(args, "ingest_pdfs", False):
        from privacy_knowledgebase.legal_pdf_ingest import ingest_reference_pdfs

        ref_dir = Path(args.reference_dir) if getattr(args, "reference_dir", None) else None
        pdf_ingest = ingest_reference_pdfs(
            reference_dir=ref_dir,
            input_md=input_path,
            dry_run=bool(getattr(args, "dry_run", False)),
        )

    outputs = generate_outputs(input_path.resolve(), output_dir.resolve())
    from privacy_audit.ingestion.output_mirror import mirror_knowledge_base

    mirrored = mirror_knowledge_base(output_dir.resolve())
    result: dict[str, Any] = {"status": "ok", "outputs": {name: str(path) for name, path in outputs.items()}}
    if pdf_ingest is not None:
        result["pdf_ingest"] = pdf_ingest
    if mirrored is not None:
        result["output_mirror"] = str(mirrored)
    return result


def _cmd_compare_batch_verdicts(args: argparse.Namespace) -> dict[str, Any]:
    from privacy_audit.workflows.verdict_compare import compare_batch_verdicts, export_verdict_compare

    apps = _resolve_batch_apps(args)
    before_dir = Path(args.before_reports_dir)
    after_dir = Path(args.after_reports_dir)
    data = compare_batch_verdicts(
        before_dir=before_dir,
        after_dir=after_dir,
        apps=apps,
        before_label=args.before_label,
        after_label=args.after_label,
    )
    out_path = Path(args.output_file) if args.output_file else Path(
        args.output_dir
    ) / f"verdict_compare_{args.before_label}_vs_{args.after_label}.json"
    export_verdict_compare(data, out_path)
    data["output_file"] = str(out_path)
    return data


def _cmd_import_oldsrc_evidence(args: argparse.Namespace) -> dict[str, Any]:
    if args.apps_file:
        apps = _resolve_batch_apps(argparse.Namespace(apps_file=args.apps_file, apps=""))
    else:
        apps = [name.strip() for name in args.apps.split(",") if name.strip()]
    legacy_run = _path_arg(args.legacy_run_dir) or _default_legacy_run_dir()
    if not apps:
        apps = list_legacy_dynamic_apps(legacy_run)
    result = import_oldsrc_pipeline(
        app_names=apps,
        legacy_run_dir=legacy_run,
        static_out_dir=Path(args.static_out_dir),
        dynamic_out_dir=Path(args.dynamic_out_dir),
        taxonomy_file=Path(args.taxonomy_file),
    )
    if args.rebuild_bundles:
        bundle_result = build_and_export_evidence_bundles(
            app_names=apps,
            output_dir=Path(args.bundles_output_dir),
            taxonomy_file=Path(args.taxonomy_file),
            evidence_dir=Path(args.dynamic_evidence_dir) or Path("output"),
            static_dir=Path("output/stage2_static_samples"),
            legacy_run_dir=legacy_run,
            legacy_md_run_dir=_path_arg(args.legacy_md_run_dir) or _default_legacy_md_run_dir(),
            use_knowledge_base=not bool(getattr(args, "no_knowledge_base", False)),
            kb_min_confidence=float(getattr(args, "kb_min_confidence", 0.72)),
        )
        result["evidence_bundles"] = bundle_result
    return result


def _cmd_lab4_2(args: argparse.Namespace) -> dict[str, Any]:
    logging.getLogger("privacy_audit.cli").info(
        "lab4-2 app=%s before=%s after=%s",
        args.app_name,
        args.before_off_file,
        args.after_off_file,
    )
    return run_lab4_2(
        app_name=args.app_name,
        before_off_file=Path(args.before_off_file) if args.before_off_file else None,
        after_off_file=Path(args.after_off_file) if args.after_off_file else None,
        output_dir=Path(args.output_dir),
        toggle_name=args.toggle_name,
    )


def _cmd_lab4_3(args: argparse.Namespace) -> dict[str, Any]:
    logging.getLogger("privacy_audit.cli").info(
        "lab4-3 app=%s dynamic=%s copy=%s",
        args.app_name,
        args.dynamic_evidence_file,
        args.copy_file,
    )
    return run_lab4_3(
        app_name=args.app_name,
        policy_file=Path(args.policy_file),
        taxonomy_file=Path(args.taxonomy_file),
        dynamic_evidence_file=Path(args.dynamic_evidence_file) if args.dynamic_evidence_file else None,
        copy_file=Path(args.copy_file) if args.copy_file else None,
        output_dir=Path(args.output_dir),
    )


def _resolve_batch_apps(args: argparse.Namespace) -> list[str]:
    if args.apps_file:
        apps_path = Path(args.apps_file)
        if apps_path.suffix.lower() == ".json":
            payload = json.loads(apps_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [str(name).strip() for name in payload if str(name).strip()]
            if isinstance(payload, dict) and isinstance(payload.get("apps"), list):
                return [str(name).strip() for name in payload["apps"] if str(name).strip()]
        lines = apps_path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    return [name.strip() for name in args.apps.split(",") if name.strip()]


def _resolve_lab4_1_batch_paths(
    args: argparse.Namespace,
    app_name: str,
) -> dict[str, Path | None]:
    """
    When --evidence-dir points at flat lab4_1_evidence bundles, still merge
    static/dynamic/legacy via output/ roots and hydrate from bundle provenance.
    """
    evidence_file: Path | None = None
    evidence_dir = Path("output")
    static_dir = Path("output/stage2_static_samples")
    legacy_run = _path_arg(getattr(args, "legacy_run_dir", None)) or _default_legacy_run_dir()
    legacy_md = _path_arg(getattr(args, "legacy_md_run_dir", None)) or _default_legacy_md_run_dir()

    if args.evidence_dir:
        bundle_dir = Path(args.evidence_dir)
        flat_bundle = bundle_dir / f"{app_name}.json"
        if flat_bundle.exists():
            evidence_file = flat_bundle
        if bundle_dir.name != "lab4_1_evidence" and bundle_dir.exists():
            evidence_dir = bundle_dir

    if getattr(args, "static_evidence_dir", None):
        static_dir = Path(args.static_evidence_dir)

    return {
        "evidence_file": evidence_file,
        "evidence_dir": evidence_dir,
        "static_evidence_dir": static_dir,
        "legacy_run_dir": legacy_run,
        "legacy_md_run_dir": legacy_md,
    }


def _make_batch_processor(args: argparse.Namespace, output_dir: Path):
    taxonomy_file = Path(args.taxonomy_file)
    rules_file = Path(args.rules_file)

    def process_app(app_name: str) -> dict[str, Any]:
        if args.mode == "lab4-1":
            policy_file = Path(args.policy_dir) / f"{app_name}.txt"
            paths = _resolve_lab4_1_batch_paths(args, app_name)
            return run_lab4_1(
                app_name=app_name,
                policy_file=policy_file,
                output_dir=output_dir,
                taxonomy_file=taxonomy_file,
                rules_file=rules_file,
                evidence_file=paths["evidence_file"],
                evidence_dir=paths["evidence_dir"],
                static_evidence_dir=paths["static_evidence_dir"],
                legacy_run_dir=paths["legacy_run_dir"],
                legacy_md_run_dir=paths["legacy_md_run_dir"],
                use_knowledge_base=not bool(getattr(args, "no_knowledge_base", False)),
                kb_min_confidence=float(getattr(args, "kb_min_confidence", 0.72)),
            )
        if args.mode == "lab4-2":
            before_file = Path(args.before_dir) / f"{app_name}.json" if args.before_dir else None
            after_file = Path(args.after_dir) / f"{app_name}.json" if args.after_dir else None
            return run_lab4_2(
                app_name=app_name,
                before_off_file=before_file,
                after_off_file=after_file,
                output_dir=output_dir,
                toggle_name=args.toggle_name,
            )
        policy_file = Path(args.policy_dir) / f"{app_name}.txt"
        dynamic_file = Path(args.dynamic_dir) / f"{app_name}.json" if args.dynamic_dir else None
        copy_file = Path(args.copy_dir) / f"{app_name}.json" if args.copy_dir else None
        return run_lab4_3(
            app_name=app_name,
            policy_file=policy_file,
            taxonomy_file=taxonomy_file,
            dynamic_evidence_file=dynamic_file,
            copy_file=copy_file,
            output_dir=output_dir,
        )

    return process_app


def _cmd_batch(args: argparse.Namespace) -> dict[str, Any]:
    apps = _resolve_batch_apps(args)
    logging.getLogger("privacy_audit.cli").info(
        "batch mode=%s apps=%d job_id=%s output=%s",
        args.mode,
        len(apps),
        args.job_id or "(auto)",
        args.output_dir,
    )
    if not apps:
        raise SystemExit("batch requires --apps or --apps-file with at least one app name")
    output_dir = Path(args.output_dir)
    config = BatchJobConfig(
        apps=apps,
        mode=args.mode,
        output_dir=output_dir,
        job_id=args.job_id or "",
        resume=bool(args.resume),
        force_rerun=bool(args.force_rerun),
        retry_count=int(args.retry_count),
        retry_delay_sec=float(args.retry_delay_sec),
        retry_backoff=float(args.retry_backoff),
    )
    return run_batch_job(config, _make_batch_processor(args, output_dir))


def _cmd_batch_status(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if args.manifest_file:
        manifest_path = Path(args.manifest_file)
    else:
        job_dir = None
        if args.job_id:
            job_dir = output_dir / "batch_jobs" / args.job_id
        else:
            job_dir = find_latest_job_dir(output_dir, mode=args.mode or None)
        if job_dir is None:
            return {
                "status": "not_found",
                "message": "No batch job directory found",
                "output_dir": str(output_dir),
            }
        manifest_path = job_dir / "manifest.json"

    manifest = load_batch_manifest(manifest_path)
    tasks = manifest.get("tasks", {})
    counts: dict[str, int] = {}
    for task in tasks.values():
        key = str(task.get("status", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return {
        "status": "ok",
        "manifest_path": str(manifest_path),
        "job_id": manifest.get("job_id", ""),
        "mode": manifest.get("mode", ""),
        "task_status_counts": counts,
        "tasks": tasks,
        "updated_at_utc": manifest.get("updated_at_utc", ""),
    }


def _cmd_bridge_stage1(args: argparse.Namespace) -> dict[str, Any]:
    return import_claims_from_review_bundle(
        review_bundle_file=Path(args.review_bundle_file),
        mapping_file=Path(args.mapping_file),
        output_file=Path(args.output_file),
        nn_threshold=float(args.nn_threshold),
    )


def _cmd_refine_stage1_mapping(args: argparse.Namespace) -> dict[str, Any]:
    return refine_unknown_labels(
        bridge_output_file=Path(args.bridge_output_file),
        mapping_file=Path(args.mapping_file),
        clusters_output_file=Path(args.clusters_output_file),
        suggestions_output_file=Path(args.suggestions_output_file),
        min_confidence=float(args.min_confidence),
        use_similarity=bool(args.use_similarity),
        similarity_threshold=float(args.similarity_threshold),
    )


def _cmd_stage2_static(args: argparse.Namespace) -> dict[str, Any]:
    import os

    if getattr(args, "fast", False):
        os.environ["PRIVACY_AUDIT_STATIC_FAST"] = "1"
    taxonomy_file = Path(args.taxonomy_file)
    output_dir = Path(args.output_dir)
    if args.apk_file:
        return run_stage2_for_apk(
            apk_path=Path(args.apk_file),
            taxonomy_file=taxonomy_file,
            output_dir=output_dir,
        )
    return run_stage2_for_directory(
        apk_dir=Path(args.apk_dir),
        taxonomy_file=taxonomy_file,
        output_dir=output_dir,
        limit=int(args.limit),
    )


def _cmd_stage3_dynamic(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if getattr(args, "log_file", None):
        log_file = Path(args.log_file)
    else:
        log_file = output_dir / args.app_name / "stage3_run.log"
    configure_cli_logging(verbose=bool(getattr(args, "verbose", False)), log_file=log_file)
    log = logging.getLogger("privacy_audit.cli")
    log.info("stage3-dynamic log_file=%s", log_file)

    pii_text_file: Path | None = None
    if args.pii_text_file:
        pii_text_file = Path(args.pii_text_file)
    elif args.pii_text_dir:
        pii_dir = Path(args.pii_text_dir)
        candidate = pii_dir / f"{args.app_name}.txt"
        if candidate.exists():
            pii_text_file = candidate
        else:
            low_name = args.app_name.lower()
            for txt_file in pii_dir.glob("*.txt"):
                if txt_file.stem.lower() == low_name:
                    pii_text_file = txt_file
                    break

    stage2_file = Path(args.stage2_evidence_file) if args.stage2_evidence_file else None
    return run_stage3_dynamic(
        app_name=args.app_name,
        app_package=args.app_package,
        taxonomy_file=Path(args.taxonomy_file),
        pii_text_file=pii_text_file,
        output_dir=output_dir,
        stage2_evidence_file=stage2_file,
        appium_server_url=args.appium_server_url,
        dry_run=not bool(args.real_run),
        duration_sec=int(args.duration_sec),
        frida_launch_mode=args.frida_mode,
        frida_grace_sec=int(getattr(args, "frida_grace_sec", 5)),
        frida_tail_sec=int(getattr(args, "frida_tail_sec", 3)),
        appium_no_reset=not bool(args.appium_reset),
        skip_preflight=bool(getattr(args, "skip_preflight", False)),
        appium_connect_timeout_sec=float(getattr(args, "appium_connect_timeout", 5.0)),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Android privacy audit CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Logging: use -v for DEBUG; logs stream to stdout immediately. "
            "stage3-dynamic also writes output/<dir>/<App>/stage3_run.log . "
            "Example: python -m privacy_audit.cli -v stage3-dynamic --real-run ..."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging to stdout (recommended for stage3-dynamic)",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional log file path (stage3-dynamic sets a default under output-dir)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("lab4-1", help="Run Lab4-1 minimal pipeline")
    p1.add_argument("--app-name", required=True, help="App display name")
    p1.add_argument("--policy-file", required=True, help="Policy text file path")
    p1.add_argument("--evidence-file", help="Optional prebuilt evidence bundle json path")
    p1.add_argument(
        "--evidence-dir",
        help="Evidence root (flat <app>.json and/or stage3 <app>/dynamic_evidence.json)",
    )
    p1.add_argument(
        "--static-evidence-dir",
        help="Stage2 static evidence root (<alias>/static_evidence.json)",
    )
    p1.add_argument(
        "--legacy-run-dir",
        help="oldsrc run dir with dynamic_testing/per_app (default: run_20260427_001723)",
    )
    p1.add_argument(
        "--legacy-md-run-dir",
        help="oldsrc markdown report dir (default: run_20260427_001723z)",
    )
    p1.add_argument(
        "--taxonomy-file",
        default=str(_default_taxonomy_file()),
        help="Personal info taxonomy yaml path",
    )
    p1.add_argument(
        "--rules-file",
        default=str(_default_rules_file()),
        help="Decision rules yaml path",
    )
    p1.add_argument(
        "--output-dir",
        default="output/lab4_1",
        help="Output report directory",
    )

    p2 = subparsers.add_parser("lab4-2", help="Run Lab4-2 consent authenticity pipeline")
    p2.add_argument("--app-name", required=True, help="App display name")
    p2.add_argument(
        "--before-off-file",
        help="Stage3 dynamic evidence before disabling consent/switch",
    )
    p2.add_argument(
        "--after-off-file",
        help="Stage3 dynamic evidence after disabling consent/switch",
    )
    p2.add_argument(
        "--toggle-name",
        default="privacy_switch",
        help="Toggle/switch label for report",
    )
    p2.add_argument(
        "--output-dir",
        default="output/lab4_2",
        help="Output report directory",
    )

    p3 = subparsers.add_parser("lab4-3", help="Run Lab4-3 access authenticity pipeline")
    p3.add_argument("--app-name", required=True, help="App display name")
    p3.add_argument("--policy-file", required=True, help="Policy text file path")
    p3.add_argument(
        "--dynamic-evidence-file",
        help="Stage3 dynamic evidence json path",
    )
    p3.add_argument(
        "--copy-file",
        help="Exported copy JSON file path for access-right validation",
    )
    p3.add_argument(
        "--taxonomy-file",
        default=str(_default_taxonomy_file()),
        help="Personal info taxonomy yaml path",
    )
    p3.add_argument(
        "--output-dir",
        default="output/lab4_3",
        help="Output report directory",
    )

    p4 = subparsers.add_parser("batch", help="Run stage7 batch pipeline with retry/resume/logging")
    p4.add_argument("--apps", default="", help="Comma-separated app names")
    p4.add_argument(
        "--apps-file",
        help="App list file (.txt one per line, or .json list/dict with apps key)",
    )
    p4.add_argument(
        "--mode",
        default="lab4-1",
        choices=["lab4-1", "lab4-2", "lab4-3"],
        help="Batch mode",
    )
    p4.add_argument(
        "--policy-dir",
        default="input/pii_text",
        help="Policy text directory; files named <app>.txt",
    )
    p4.add_argument(
        "--evidence-dir",
        help="Evidence root: flat <app>.json bundles and/or stage3 dynamic paths",
    )
    p4.add_argument(
        "--static-evidence-dir",
        help="Stage2 static evidence root for live merge during batch",
    )
    p4.add_argument(
        "--legacy-run-dir",
        help="oldsrc dynamic_testing run root for legacy UI evidence",
    )
    p4.add_argument(
        "--legacy-md-run-dir",
        help="oldsrc markdown report root (run_20260427_001723z)",
    )
    p4.add_argument(
        "--taxonomy-file",
        default=str(_default_taxonomy_file()),
        help="Personal info taxonomy yaml path",
    )
    p4.add_argument(
        "--rules-file",
        default=str(_default_rules_file()),
        help="Decision rules yaml path",
    )
    p4.add_argument(
        "--output-dir",
        default="output/lab4_1",
        help="Output report directory",
    )
    p4.add_argument(
        "--before-dir",
        help="Batch input dir for lab4-2 before_off files, named <app>.json",
    )
    p4.add_argument(
        "--after-dir",
        help="Batch input dir for lab4-2 after_off files, named <app>.json",
    )
    p4.add_argument(
        "--dynamic-dir",
        help="Batch input dir for lab4-3 dynamic evidence files, named <app>.json",
    )
    p4.add_argument(
        "--copy-dir",
        help="Batch input dir for lab4-3 copy files, named <app>.json",
    )
    p4.add_argument(
        "--toggle-name",
        default="privacy_switch",
        help="Toggle/switch label for lab4-2 reports",
    )
    p4.add_argument(
        "--job-id",
        default="",
        help="Batch job id (default: timestamp). State stored under output/batch_jobs/<job-id>/",
    )
    p4.add_argument(
        "--resume",
        action="store_true",
        help="Resume job: skip apps already marked success in manifest",
    )
    p4.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun all apps even when --resume is set",
    )
    p4.add_argument(
        "--retry-count",
        default=2,
        type=int,
        help="Per-app retry count on failure (default: 2)",
    )
    p4.add_argument(
        "--retry-delay-sec",
        default=3.0,
        type=float,
        help="Initial retry delay in seconds (exponential backoff)",
    )
    p4.add_argument(
        "--retry-backoff",
        default=2.0,
        type=float,
        help="Retry backoff multiplier",
    )
    p4.add_argument(
        "--no-knowledge-base",
        action="store_true",
        help="Disable privacy_knowledgebase enrichment in lab4-1 batch",
    )
    p4.add_argument(
        "--kb-min-confidence",
        default=0.72,
        type=float,
        help="Min KB confidence for observed_pii merge (lab4-1)",
    )

    p4s = subparsers.add_parser("batch-status", help="Inspect batch job task state")
    p4s.add_argument(
        "--output-dir",
        default="output/lab4_1",
        help="Batch output root directory",
    )
    p4s.add_argument("--job-id", default="", help="Batch job id under output/batch_jobs/")
    p4s.add_argument(
        "--manifest-file",
        help="Direct path to manifest.json (overrides --job-id lookup)",
    )
    p4s.add_argument(
        "--mode",
        default="",
        help="When --job-id omitted, pick latest job for this mode",
    )

    p_ev = subparsers.add_parser(
        "build-evidence-bundles",
        help="Merge stage2/stage3/legacy evidence into flat bundles for Lab4-1 batch",
    )
    p_ev.add_argument("--apps", default="", help="Comma-separated app names")
    p_ev.add_argument(
        "--apps-file",
        help="App list file (.txt or .json); default: legacy per_app apps",
    )
    p_ev.add_argument(
        "--output-dir",
        default="output/lab4_1_evidence",
        help="Directory for <app>.json merged evidence bundles",
    )
    p_ev.add_argument(
        "--dynamic-evidence-dir",
        default="output",
        help="Root to search stage3_dynamic_* and per-app dynamic_evidence.json",
    )
    p_ev.add_argument(
        "--static-evidence-dir",
        default="output/stage2_static_samples",
        help="Stage2 static evidence root",
    )
    p_ev.add_argument(
        "--legacy-run-dir",
        default=str(_default_legacy_run_dir()),
        help="oldsrc run with dynamic_testing/per_app",
    )
    p_ev.add_argument(
        "--legacy-md-run-dir",
        default=str(_default_legacy_md_run_dir()),
        help="oldsrc markdown report dir (run_20260427_001723z)",
    )
    p_ev.add_argument(
        "--taxonomy-file",
        default=str(_default_taxonomy_file()),
        help="Personal info taxonomy yaml path",
    )
    p_ev.add_argument(
        "--no-knowledge-base",
        action="store_true",
        help="Disable privacy_knowledgebase feasible-evidence enrichment",
    )
    p_ev.add_argument(
        "--kb-min-confidence",
        default=0.72,
        type=float,
        help="Min confidence for KB hits merged into observed_pii",
    )

    p_kb = subparsers.add_parser(
        "kb-build",
        help="Regenerate privacy_knowledgebase JSON from input.md (legal matrix)",
    )
    p_kb.add_argument(
        "--input",
        help="Path to input.md (default: privacy_knowledgebase/input.md)",
    )
    p_kb.add_argument(
        "--output-dir",
        help="Output directory (default: privacy_knowledgebase/knowledge_base)",
    )
    p_kb.add_argument(
        "--ingest-pdfs",
        action="store_true",
        help="Parse PDFs from reference/法律参考文件/ into input.md appendix before build",
    )
    p_kb.add_argument(
        "--reference-dir",
        help="Legal reference PDF directory (default: project/reference/法律参考文件)",
    )
    p_kb.add_argument(
        "--dry-run",
        action="store_true",
        help="With --ingest-pdfs, show ingest result without writing input.md",
    )

    p_cmp = subparsers.add_parser(
        "compare-batch-verdicts",
        help="Compare Lab4-1 verdicts between two report directories (e.g. no-KB vs KB)",
    )
    p_cmp.add_argument("--apps", default="", help="Comma-separated app names")
    p_cmp.add_argument("--apps-file", help="App list file")
    p_cmp.add_argument(
        "--before-reports-dir",
        required=True,
        help="Directory with *_lab4_1_report.json (baseline, e.g. no-KB batch)",
    )
    p_cmp.add_argument(
        "--after-reports-dir",
        required=True,
        help="Directory with *_lab4_1_report.json (comparison, e.g. KB batch)",
    )
    p_cmp.add_argument("--before-label", default="no_kb", help="Label for before column")
    p_cmp.add_argument("--after-label", default="with_kb", help="Label for after column")
    p_cmp.add_argument(
        "--output-dir",
        default="output/lab4_1",
        help="Directory for comparison JSON output",
    )
    p_cmp.add_argument(
        "--output-file",
        help="Explicit comparison JSON path (default: output-dir/verdict_compare_<labels>.json)",
    )

    p_old = subparsers.add_parser(
        "import-oldsrc-evidence",
        help="Import oldsrc Lab3 audit/test_cases into stage2/stage3 derived evidence dirs",
    )
    p_old.add_argument("--apps", default="", help="Comma-separated app names")
    p_old.add_argument("--apps-file", help="App list file")
    p_old.add_argument(
        "--legacy-run-dir",
        default=str(_default_legacy_run_dir()),
        help="oldsrc run directory (default run_20260427_001723)",
    )
    p_old.add_argument(
        "--static-out-dir",
        default="output/stage2_oldsrc_derived",
        help="Output root for derived static_evidence.json per app slug",
    )
    p_old.add_argument(
        "--dynamic-out-dir",
        default="output/stage3_oldsrc_derived",
        help="Output root for derived dynamic_evidence.json per app",
    )
    p_old.add_argument(
        "--taxonomy-file",
        default=str(_default_taxonomy_file()),
        help="Personal info taxonomy yaml path",
    )
    p_old.add_argument(
        "--rebuild-bundles",
        action="store_true",
        help="After import, rebuild output/lab4_1_evidence bundles",
    )
    p_old.add_argument(
        "--bundles-output-dir",
        default="output/lab4_1_evidence",
        help="Flat bundle output dir when --rebuild-bundles is set",
    )
    p_old.add_argument(
        "--dynamic-evidence-dir",
        default="output",
        help="Evidence search root for bundle rebuild",
    )
    p_old.add_argument(
        "--legacy-md-run-dir",
        default=str(_default_legacy_md_run_dir()),
        help="oldsrc markdown report dir for bundle rebuild",
    )
    p_old.add_argument(
        "--no-knowledge-base",
        action="store_true",
        help="Disable KB enrichment when rebuilding bundles",
    )
    p_old.add_argument(
        "--kb-min-confidence",
        default=0.72,
        type=float,
        help="Min confidence for KB hits merged into observed_pii",
    )

    p5 = subparsers.add_parser(
        "bridge-stage1",
        help="Import legacy review labels to normalized claims",
    )
    p5.add_argument(
        "--review-bundle-file",
        required=True,
        help="Legacy review_bundle.json path",
    )
    p5.add_argument(
        "--mapping-file",
        default=str(_default_legacy_mapping_file()),
        help="Legacy label mapping yaml path",
    )
    p5.add_argument(
        "--output-file",
        default="output/stage1/claims_from_review_bundle.json",
        help="Converted claims output path",
    )
    p5.add_argument(
        "--nn-threshold",
        default=0.30,
        type=float,
        help="Nearest-neighbor similarity threshold for unmatched labels",
    )

    p6 = subparsers.add_parser(
        "refine-stage1-mapping",
        help="Cluster unknown labels and expand legacy mapping",
    )
    p6.add_argument(
        "--bridge-output-file",
        default="output/stage1/claims_from_review_bundle.json",
        help="Output json from bridge-stage1 command",
    )
    p6.add_argument(
        "--mapping-file",
        default=str(_default_legacy_mapping_file()),
        help="Legacy label mapping yaml path",
    )
    p6.add_argument(
        "--clusters-output-file",
        default="output/stage1/unknown_label_clusters.json",
        help="Cluster output json path",
    )
    p6.add_argument(
        "--suggestions-output-file",
        default="output/stage1/mapping_suggestions.json",
        help="Suggestions output json path",
    )
    p6.add_argument(
        "--min-confidence",
        default=0.8,
        type=float,
        help="Minimum confidence threshold for auto-apply",
    )
    p6.add_argument(
        "--use-similarity",
        action="store_true",
        default=True,
        help="Enable nearest-neighbor similarity matching",
    )
    p6.add_argument(
        "--similarity-threshold",
        default=0.52,
        type=float,
        help="Minimum NN similarity score for candidate mapping",
    )

    p7 = subparsers.add_parser(
        "stage2-static",
        help="Run stage2 static analysis (Exodus + Androguard)",
    )
    p7.add_argument(
        "--apk-file",
        help="Single APK path",
    )
    p7.add_argument(
        "--apk-dir",
        default="input/apks",
        help="APK directory for batch scan",
    )
    p7.add_argument(
        "--taxonomy-file",
        default=str(_default_taxonomy_file()),
        help="Personal info taxonomy yaml path",
    )
    p7.add_argument(
        "--output-dir",
        default="output/stage2_static",
        help="Output directory for static evidence reports",
    )
    p7.add_argument(
        "--limit",
        default=0,
        type=int,
        help="Optional max APK count when scanning directory",
    )
    p7.add_argument(
        "--fast",
        action="store_true",
        help="Skip full Androguard AnalyzeAPK; use dex string scan (faster batch)",
    )

    p8 = subparsers.add_parser(
        "stage3-dynamic",
        help="Run stage3 dynamic evidence pipeline (Appium + Frida)",
    )
    p8.add_argument("--app-name", required=True, help="App display name")
    p8.add_argument("--app-package", required=True, help="Android app package name")
    p8.add_argument(
        "--pii-text-file",
        help="Optional policy/plaintext file for path suggestions",
    )
    p8.add_argument(
        "--pii-text-dir",
        default="input/pii_text",
        help="Policy plaintext directory; uses <app-name>.txt when present",
    )
    p8.add_argument(
        "--stage2-evidence-file",
        help="Optional stage2 static_evidence.json path",
    )
    p8.add_argument(
        "--taxonomy-file",
        default=str(_default_taxonomy_file()),
        help="Personal info taxonomy yaml path",
    )
    p8.add_argument(
        "--output-dir",
        default="output/stage3_dynamic",
        help="Output directory for dynamic evidence",
    )
    p8.add_argument(
        "--appium-server-url",
        default="http://127.0.0.1:4723",
        help="Appium server URL",
    )
    p8.add_argument(
        "--duration-sec",
        default=20,
        type=int,
        help="Frida duration for dry-run only (real run uses Appium-sync window)",
    )
    p8.add_argument(
        "--frida-grace-sec",
        default=5,
        type=int,
        help="Seconds to capture before Appium starts (manual navigation, default: 5)",
    )
    p8.add_argument(
        "--frida-tail-sec",
        default=3,
        type=int,
        help="Extra Frida capture seconds after Appium finishes (default: 3)",
    )
    p8.add_argument(
        "--real-run",
        action="store_true",
        help="Run real Appium/Frida instead of dry-run simulation",
    )
    p8.add_argument(
        "--frida-mode",
        default="attach",
        choices=["attach", "spawn"],
        help="Frida launch mode: attach to running app (default) or cold spawn",
    )
    p8.add_argument(
        "--appium-reset",
        action="store_true",
        help="Reset app state before Appium session (default: preserve login)",
    )
    p8.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip adb/Appium/Frida preflight checks",
    )
    p8.add_argument(
        "--appium-connect-timeout",
        default=5.0,
        type=float,
        help="Seconds to wait for Appium /status before session (default: 5)",
    )

    return parser


def main() -> int:
    """Run command line interface."""
    parser = _build_parser()
    args = parser.parse_args()

    log_file = Path(args.log_file) if getattr(args, "log_file", None) else None
    configure_cli_logging(verbose=bool(getattr(args, "verbose", False)), log_file=log_file)
    cli_log = logging.getLogger("privacy_audit.cli")
    cli_log.info("Running command: %s", args.command)

    if args.command == "lab4-1":
        result = _cmd_lab4_1(args)
    elif args.command == "lab4-2":
        result = _cmd_lab4_2(args)
    elif args.command == "lab4-3":
        result = _cmd_lab4_3(args)
    elif args.command == "batch":
        result = _cmd_batch(args)
    elif args.command == "batch-status":
        result = _cmd_batch_status(args)
    elif args.command == "build-evidence-bundles":
        result = _cmd_build_evidence_bundles(args)
    elif args.command == "kb-build":
        result = _cmd_kb_build(args)
    elif args.command == "compare-batch-verdicts":
        result = _cmd_compare_batch_verdicts(args)
    elif args.command == "import-oldsrc-evidence":
        result = _cmd_import_oldsrc_evidence(args)
    elif args.command == "bridge-stage1":
        result = _cmd_bridge_stage1(args)
    elif args.command == "refine-stage1-mapping":
        result = _cmd_refine_stage1_mapping(args)
    elif args.command == "stage2-static":
        result = _cmd_stage2_static(args)
    else:
        result = _cmd_stage3_dynamic(args)

    cli_log.info("Command finished: %s", args.command)
    try:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        # Fallback for terminals with legacy code pages (e.g., cp1252).
        sys.stdout.buffer.write((json.dumps(result, ensure_ascii=True, indent=2) + "\n").encode("utf-8"))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
