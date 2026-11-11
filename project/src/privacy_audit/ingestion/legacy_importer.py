"""Import legacy Lab3/oldsrc artifacts as static/dynamic evidence files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from privacy_audit.analyzers.policy.policy_parser import load_personal_info_taxonomy
from privacy_audit.ingestion.evidence_loader import _match_taxonomy_in_text
from privacy_audit.utils.path_utils import output_provenance


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


from privacy_audit.ingestion.evidence_loader import _slug_for_app


def _load_dynamic_results_index(legacy_run_dir: Path, app_name: str) -> dict[str, dict[str, Any]]:
    per_app = legacy_run_dir / "dynamic_testing" / "per_app" / app_name
    index: dict[str, dict[str, Any]] = {}
    if not per_app.is_dir():
        return index
    for json_file in sorted(per_app.glob("*.json")):
        payload = _read_json(json_file)
        for result in payload.get("results", []):
            if isinstance(result, dict) and result.get("case_id"):
                index[str(result["case_id"])] = result
    return index


def extract_observed_from_legacy_cases(
    app_name: str,
    legacy_run_dir: Path | None,
    taxonomy: dict[str, dict[str, list[str]]],
) -> tuple[list[str], dict[str, Any]]:
    """Map legacy test_cases + per-app results to observed PII hints."""
    meta: dict[str, Any] = {"cases": [], "source": "oldsrc_test_cases"}
    if legacy_run_dir is None or not legacy_run_dir.exists():
        return [], meta

    cases_file = legacy_run_dir / "dynamic_testing" / "test_cases.json"
    if not cases_file.exists():
        return [], meta

    payload = _read_json(cases_file)
    cases = [c for c in payload.get("test_cases", []) if c.get("app_name") == app_name]
    results_index = _load_dynamic_results_index(legacy_run_dir, app_name)
    observed: set[str] = set()

    for case in cases:
        case_id = str(case.get("case_id", ""))
        result = results_index.get(case_id)
        base_text = " ".join(
            str(case.get(key, ""))
            for key in ("target_data", "sentence", "dynamic_test_goal", "path_text")
        )
        channel = str(case.get("channel", ""))
        status = str(result.get("status", "")).upper() if result else "NOT_RUN"
        notes = ""
        steps_text = ""
        if result:
            notes = str(result.get("notes", ""))
            steps = result.get("steps_taken", [])
            steps_text = " ".join(str(s) for s in steps) if isinstance(steps, list) else ""

        block = f"{base_text} {steps_text} {notes}".strip()
        matched: set[str] = set()
        if status == "PASS":
            matched = _match_taxonomy_in_text(block, taxonomy)
        elif status == "FAIL" and channel == "app_ui":
            # Partial UI run: use notes/steps when automation entered the app.
            if "主屏幕" not in notes and "未进入" not in notes:
                matched = _match_taxonomy_in_text(f"{notes} {case.get('target_data', '')}", taxonomy)
        elif status == "SKIP" and channel == "app_ui":
            # App UI case skipped before run: use target_data only as weak path hint.
            matched = _match_taxonomy_in_text(str(case.get("target_data", "")), taxonomy)

        if matched:
            observed.update(matched)
            meta["cases"].append(
                {
                    "case_id": case_id,
                    "status": status,
                    "channel": channel,
                    "matched": sorted(matched),
                }
            )

    return sorted(observed), meta


def extract_observed_from_legacy_audit(
    app_name: str,
    legacy_run_dir: Path | None,
    taxonomy: dict[str, dict[str, list[str]]],
) -> tuple[list[str], dict[str, Any]]:
    """Read audit_processed.csv rows flagged for dynamic testing."""
    meta: dict[str, Any] = {"source": "oldsrc_audit", "rows": 0}
    if legacy_run_dir is None or not legacy_run_dir.exists():
        return [], meta

    audit_path = legacy_run_dir / app_name / "audit_processed.csv"
    if not audit_path.exists():
        return [], meta

    observed: set[str] = set()
    with audit_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("app_test_candidate", "")).strip() not in {"1", "true", "True"}:
                continue
            meta["rows"] += 1
            text = " ".join(
                str(row.get(key, ""))
                for key in ("target_data", "dynamic_test_goal", "path_text", "method_claim")
            )
            observed.update(_match_taxonomy_in_text(text, taxonomy))

    return sorted(observed), meta


def export_oldsrc_derived_evidence(
    app_name: str,
    legacy_run_dir: Path,
    static_out_dir: Path,
    dynamic_out_dir: Path,
    taxonomy: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    """Write static/dynamic evidence under output/static/legacy and output/dynamic/legacy."""
    lab3_observed, lab3_meta = extract_observed_from_legacy_cases(app_name, legacy_run_dir, taxonomy)
    audit_observed, audit_meta = extract_observed_from_legacy_audit(app_name, legacy_run_dir, taxonomy)
    merged = sorted(set(lab3_observed) | set(audit_observed))

    slug = _slug_for_app(app_name)
    static_path = static_out_dir / slug / "static_evidence.json"
    dynamic_path = dynamic_out_dir / app_name / "dynamic_evidence.json"
    static_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_path.parent.mkdir(parents=True, exist_ok=True)

    static_doc = {
        "schema_version": 1,
        "app_name": app_name,
        "apk_path": "",
        "observed_pii": merged,
        "source": "oldsrc_derived",
        "oldsrc": {
            "lab3_test_cases": lab3_observed,
            "audit_processed": audit_observed,
            "legacy_run_dir": output_provenance("static", "legacy", slug),
            "lab3_meta": lab3_meta,
            "audit_meta": audit_meta,
        },
        "exodus": {"pii_permission_hits": {k: ["oldsrc_audit"] for k in audit_observed}},
        "androguard": {"api_hits": {k: [] for k in lab3_observed}},
    }
    dynamic_doc = {
        "schema_version": 1,
        "pipeline": "dynamic_capture",
        "app_name": app_name,
        "dry_run": True,
        "source": "oldsrc_lab3_derived",
        "candidate_pii": merged,
        "frida_result": {"events": []},
        "ui_result": {
            "runner": "oldsrc_lab3",
            "mode": "imported",
            "steps": [f"imported_cases={len(lab3_meta.get('cases', []))}"],
            "notes": lab3_meta,
        },
    }

    static_path.write_text(json.dumps(static_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    dynamic_path.write_text(json.dumps(dynamic_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "app_name": app_name,
        "observed_count": len(merged),
        "static_path": str(static_path),
        "dynamic_path": str(dynamic_path),
        "lab3_count": len(lab3_observed),
        "audit_count": len(audit_observed),
    }


def import_legacy_pipeline(
    app_names: list[str],
    legacy_run_dir: Path,
    static_out_dir: Path,
    dynamic_out_dir: Path,
    taxonomy_file: Path,
) -> dict[str, Any]:
    taxonomy = load_personal_info_taxonomy(taxonomy_file)
    written: list[dict[str, Any]] = []
    for app_name in app_names:
        if not (legacy_run_dir / app_name).is_dir() and not (
            legacy_run_dir / "dynamic_testing" / "per_app" / app_name
        ).is_dir():
            written.append({"app_name": app_name, "skipped": True, "reason": "no oldsrc folder"})
            continue
        written.append(
            export_oldsrc_derived_evidence(
                app_name=app_name,
                legacy_run_dir=legacy_run_dir,
                static_out_dir=static_out_dir,
                dynamic_out_dir=dynamic_out_dir,
                taxonomy=taxonomy,
            )
        )
    return {
        "legacy_run_dir": str(legacy_run_dir),
        "static_out_dir": str(static_out_dir),
        "dynamic_out_dir": str(dynamic_out_dir),
        "apps": app_names,
        "written": written,
    }

