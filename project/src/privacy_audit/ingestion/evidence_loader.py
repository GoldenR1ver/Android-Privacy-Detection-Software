"""Resolve and merge static/dynamic/legacy evidence into Lab4-1 observed_pii."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from privacy_audit.analyzers.policy.policy_parser import load_personal_info_taxonomy
from privacy_audit.utils.path_utils import as_output_path, output_provenance, provenance_base


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_from_output_path(rel_path: str) -> dict[str, Any]:
    """Load JSON under src/ from provenance path like output/stage2_.../file.json."""
    if not rel_path or not rel_path.startswith("output/"):
        return {}
    return _read_json(provenance_base() / Path(rel_path.replace("\\", "/")))


def is_prepared_evidence_bundle(data: dict[str, Any]) -> bool:
    """Flat lab4_1_evidence bundle with pre-merged observed_pii and KB metadata."""
    return bool(data.get("observed_pii")) and isinstance(data.get("knowledge_base"), dict)


def hydrate_artifacts_from_provenance(
    provenance: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load static/dynamic JSON referenced by a prepared bundle's provenance map."""
    static_data = _read_json_from_output_path(provenance.get("static", ""))
    dynamic_data = _read_json_from_output_path(provenance.get("dynamic", ""))
    return static_data, dynamic_data


def _app_name_variants(app_name: str) -> list[str]:
    variants = {app_name, app_name.lower(), app_name.upper()}
    if app_name.lower() == "keep":
        variants.update({"keep", "Keep"})
    if app_name in {"支付宝", "alipay"}:
        variants.update({"支付宝", "alipay", "Alipay"})
    if app_name in {"小红书", "xiaohongshu"}:
        variants.update({"小红书", "xiaohongshu", "Xiaohongshu"})
    return list(variants)


def extract_observed_from_static(static_data: dict[str, Any]) -> list[str]:
    observed = static_data.get("observed_pii", [])
    if isinstance(observed, list) and observed:
        return sorted({str(x) for x in observed if x})
    hits: set[str] = set()
    andro = static_data.get("androguard", {})
    if isinstance(andro, dict):
        api_hits = andro.get("api_hits", {})
        if isinstance(api_hits, dict):
            hits.update(str(k) for k in api_hits.keys())
    exodus = static_data.get("exodus", {})
    if isinstance(exodus, dict):
        pii_perm = exodus.get("pii_permission_hits", {})
        if isinstance(pii_perm, dict):
            hits.update(str(k) for k in pii_perm.keys())
    return sorted(hits)


def extract_observed_from_dynamic(dynamic_data: dict[str, Any], include_candidates: bool = True) -> list[str]:
    observed: set[str] = set()
    if include_candidates:
        for item in dynamic_data.get("candidate_pii", []):
            if item and item != "unknown":
                observed.add(str(item))

    frida = dynamic_data.get("frida_result", {})
    if isinstance(frida, dict):
        for event in frida.get("events", []):
            if not isinstance(event, dict):
                continue
            pii_type = str(event.get("pii_type", ""))
            api_name = str(event.get("api_name", ""))
            if not pii_type or pii_type == "unknown":
                continue
            if api_name and api_name not in {"", "hook_loaded", "hook_error"}:
                observed.add(pii_type)
            elif api_name == "hook_loaded" and include_candidates:
                observed.add(pii_type)
    return sorted(observed)


def _match_taxonomy_in_text(text: str, taxonomy: dict[str, dict[str, list[str]]]) -> set[str]:
    lowered = text.lower()
    matched: set[str] = set()
    for pii_type, meta in taxonomy.items():
        candidates = [pii_type, *meta.get("aliases", []), *meta.get("policy_keywords", [])]
        for token in candidates:
            token = str(token).strip()
            if len(token) < 2:
                continue
            if token.lower() in lowered:
                matched.add(pii_type)
                break
    return matched


def extract_observed_from_legacy(
    app_name: str,
    legacy_run_dir: Path | None,
    taxonomy: dict[str, dict[str, list[str]]],
) -> tuple[list[str], dict[str, Any]]:
    """Infer observed PII hints from oldsrc dynamic testing JSON artifacts."""
    meta: dict[str, Any] = {"legacy_run_dir": "", "files": []}
    if legacy_run_dir is None or not legacy_run_dir.exists():
        return [], meta

    per_app = legacy_run_dir / "dynamic_testing" / "per_app" / app_name
    if not per_app.exists():
        return [], meta

    meta["legacy_run_dir"] = output_provenance("stage3_legacy", app_name)

    combined_text_parts: list[str] = []
    pass_hints: list[str] = []
    for json_file in sorted(per_app.glob("*.json")):
        meta["files"].append(output_provenance("stage3_legacy", app_name, json_file.name))
        payload = _read_json(json_file)
        if not payload:
            continue
        for result in payload.get("results", []):
            if not isinstance(result, dict):
                continue
            steps = result.get("steps_taken", [])
            notes = str(result.get("notes", ""))
            block = " ".join(str(s) for s in steps) + " " + notes
            combined_text_parts.append(block)
            if str(result.get("status", "")).upper() == "PASS":
                pass_hints.append(block)

    full_text = "\n".join(combined_text_parts)
    pass_text = "\n".join(pass_hints)
    pass_only = _match_taxonomy_in_text(pass_text, taxonomy)
    if pass_only:
        observed = pass_only
    elif full_text.strip():
        observed = _match_taxonomy_in_text(full_text, taxonomy)
    else:
        observed = set()
    return sorted(observed), meta


def extract_observed_from_legacy_md(
    app_name: str,
    legacy_md_run_dir: Path | None,
    taxonomy: dict[str, dict[str, list[str]]],
) -> list[str]:
    if legacy_md_run_dir is None or not legacy_md_run_dir.exists():
        return []
    app_dir = legacy_md_run_dir / app_name
    if not app_dir.is_dir():
        return []
    chunks: list[str] = []
    for md_file in sorted(app_dir.glob("*.md")):
        chunks.append(md_file.read_text(encoding="utf-8", errors="ignore"))
    if not chunks:
        return []
    return sorted(_match_taxonomy_in_text("\n".join(chunks), taxonomy))


def _static_roots(static_dir: Path | None) -> list[Path]:
    roots: list[Path] = []
    if static_dir is not None and static_dir.exists():
        roots.append(static_dir)
        oldsrc_derived = static_dir.parent / "stage2_oldsrc_derived"
        if oldsrc_derived.exists():
            roots.append(oldsrc_derived)
    return roots


def _slug_for_app(app_name: str) -> str:
    mapping = {
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
    return mapping.get(app_name, app_name.lower().replace(" ", "_"))


def resolve_static_path(app_name: str, static_dir: Path | None) -> Path | None:
    slug = _slug_for_app(app_name)
    for root in _static_roots(static_dir):
        candidates: list[Path] = []
        for variant in _app_name_variants(app_name):
            candidates.append(root / variant / "static_evidence.json")
            candidates.append(root / f"{variant.lower()}" / "static_evidence.json")
        candidates.append(root / slug / "static_evidence.json")
        for path in candidates:
            if path.exists():
                return path
    return None


def _dynamic_roots(evidence_dir: Path | None) -> list[Path]:
    if evidence_dir is None or not evidence_dir.exists():
        return []
    roots: list[Path] = []
    if evidence_dir.name.startswith("stage3_"):
        roots.append(evidence_dir)
        parent = evidence_dir.parent
        for sub in ("stage3_dynamic_real", "stage3_dynamic_samples", "stage3_oldsrc_derived"):
            nested = parent / sub
            if nested.is_dir() and nested not in roots:
                roots.append(nested)
    else:
        roots.append(evidence_dir)
        for sub in ("stage3_dynamic_real", "stage3_dynamic_samples", "stage3_oldsrc_derived"):
            nested = evidence_dir / sub
            if nested.is_dir() and nested not in roots:
                roots.append(nested)
    return roots


def _dynamic_path_candidates(app_name: str, root: Path) -> list[Path]:
    candidates = [
        root / app_name / "dynamic_evidence.json",
        root / f"{app_name.lower()}" / "dynamic_evidence.json",
    ]
    if root.name not in {
        "stage3_dynamic_real",
        "stage3_dynamic_samples",
        "stage3_oldsrc_derived",
        "output",
    }:
        candidates.extend(
            [
                root / "stage3_dynamic_real" / app_name / "dynamic_evidence.json",
                root / "stage3_dynamic_samples" / app_name / "dynamic_evidence.json",
                root / "stage3_oldsrc_derived" / app_name / "dynamic_evidence.json",
            ]
        )
    for variant in _app_name_variants(app_name):
        candidates.append(root / variant / "dynamic_evidence.json")
        if root.name in {"output", "stage3_dynamic_real", "stage3_dynamic_samples", "stage3_oldsrc_derived"}:
            continue
        candidates.append(root / "stage3_dynamic_real" / variant / "dynamic_evidence.json")
        candidates.append(root / "stage3_dynamic_samples" / variant / "dynamic_evidence.json")
        candidates.append(root / "stage3_oldsrc_derived" / variant / "dynamic_evidence.json")
    return candidates


def resolve_dynamic_path(app_name: str, evidence_dir: Path | None) -> Path | None:
    for root in _dynamic_roots(evidence_dir):
        for path in _dynamic_path_candidates(app_name, root):
            if path.exists():
                return path
    return None


def resolve_evidence_bundle_path(app_name: str, evidence_dir: Path | None) -> Path | None:
    if evidence_dir is None or not evidence_dir.exists():
        return None
    candidates = [
        evidence_dir / f"{app_name}.json",
        evidence_dir / app_name / "lab4_1_evidence.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def build_merged_evidence(
    app_name: str,
    taxonomy: dict[str, dict[str, list[str]]],
    evidence_dir: Path | None = None,
    static_dir: Path | None = None,
    legacy_run_dir: Path | None = None,
    legacy_md_run_dir: Path | None = None,
    evidence_file: Path | None = None,
    include_dynamic_candidates: bool = True,
    use_knowledge_base: bool = True,
    kb_min_confidence: float = 0.72,
) -> dict[str, Any]:
    """Build normalized evidence dict for compare_engine."""
    sources: dict[str, list[str]] = {
        "static": [],
        "dynamic": [],
        "legacy": [],
        "oldsrc_lab3": [],
        "oldsrc_audit": [],
        "knowledge_base": [],
        "bundle": [],
    }
    provenance: dict[str, str] = {}
    static_data: dict[str, Any] = {}
    dynamic_data: dict[str, Any] = {}

    bundle_path = evidence_file or resolve_evidence_bundle_path(app_name, evidence_dir)
    prepared_bundle: dict[str, Any] | None = None

    if bundle_path is not None and bundle_path.exists():
        bundle = _read_json(bundle_path)
        if is_prepared_evidence_bundle(bundle):
            prepared_bundle = bundle
        bundle_sources = bundle.get("sources")
        if isinstance(bundle_sources, dict) and bundle_sources:
            for key, vals in bundle_sources.items():
                if key == "knowledge_base":
                    continue
                if vals:
                    existing = set(sources.get(key, []) or [])
                    existing.update(str(v) for v in vals)
                    sources[key] = sorted(existing)
        elif bundle.get("observed_pii"):
            sources["bundle"] = [str(x) for x in bundle["observed_pii"]]
        provenance["bundle"] = as_output_path(bundle_path) or output_provenance(
            "lab4_1_evidence", f"{app_name}.json"
        )
        for key, value in (bundle.get("provenance") or {}).items():
            if value and key not in provenance:
                provenance[key] = str(value)

    if prepared_bundle:
        prov_static, prov_dynamic = hydrate_artifacts_from_provenance(
            prepared_bundle.get("provenance") or {}
        )
        if prov_static and not static_data:
            static_data = prov_static
        if prov_dynamic and not dynamic_data:
            dynamic_data = prov_dynamic

    static_path = resolve_static_path(app_name, static_dir)
    if static_path is not None:
        static_data = _read_json(static_path)
        sources["static"] = extract_observed_from_static(static_data)
        provenance["static"] = as_output_path(static_path) or output_provenance(
            "stage2_static_samples", _slug_for_app(app_name), "static_evidence.json"
        )

    dynamic_path = resolve_dynamic_path(app_name, evidence_dir)
    if dynamic_path is not None:
        dynamic_data = _read_json(dynamic_path)
        sources["dynamic"] = extract_observed_from_dynamic(
            dynamic_data,
            include_candidates=include_dynamic_candidates,
        )
        provenance["dynamic"] = as_output_path(dynamic_path) or output_provenance(
            "stage3_dynamic_real", app_name, "dynamic_evidence.json"
        )

    legacy_observed, legacy_meta = extract_observed_from_legacy(app_name, legacy_run_dir, taxonomy)
    legacy_md_observed = extract_observed_from_legacy_md(app_name, legacy_md_run_dir, taxonomy)
    sources["legacy"] = sorted(set(legacy_observed) | set(legacy_md_observed))
    if legacy_observed:
        provenance["legacy"] = output_provenance("stage3_legacy", app_name)
    if legacy_md_observed and legacy_md_run_dir:
        provenance["legacy_md"] = output_provenance("stage2_legacy_md", app_name)

    if legacy_run_dir is not None:
        from privacy_audit.ingestion.oldsrc_evidence_importer import (
            extract_observed_from_oldsrc_audit,
            extract_observed_from_oldsrc_test_cases,
        )

        lab3_observed, lab3_meta = extract_observed_from_oldsrc_test_cases(
            app_name, legacy_run_dir, taxonomy
        )
        audit_observed, audit_meta = extract_observed_from_oldsrc_audit(
            app_name, legacy_run_dir, taxonomy
        )
        sources["oldsrc_lab3"] = lab3_observed
        sources["oldsrc_audit"] = audit_observed
        if lab3_observed:
            provenance["oldsrc_lab3"] = output_provenance(
                "stage3_oldsrc_derived", app_name, "dynamic_evidence.json"
            )
        if audit_observed:
            provenance["oldsrc_audit"] = output_provenance(
                "stage2_oldsrc_audit", app_name, "audit_processed.csv"
            )

    merged: set[str] = set()
    for key in (
        "static",
        "dynamic",
        "legacy",
        "oldsrc_lab3",
        "oldsrc_audit",
        "bundle",
    ):
        merged.update(sources[key])

    result: dict[str, Any] = {
        "app_name": app_name,
        "observed_pii": sorted(merged),
        "source": "merged:pending",
        "sources": sources,
        "provenance": provenance,
    }

    if use_knowledge_base:
        from privacy_knowledgebase.enricher import enrich_merged_evidence

        previous_kb = dict((prepared_bundle or {}).get("knowledge_base") or {})
        enrich_merged_evidence(
            result,
            static_data=static_data,
            dynamic_data=dynamic_data,
            min_confidence=kb_min_confidence,
            fallback_knowledge_base=previous_kb if prepared_bundle else None,
        )
        sources = result.get("sources", sources)
        merged = set(result.get("observed_pii", []))

    source_label = "+".join(k for k, vals in sources.items() if vals) or "empty"
    result["source"] = f"merged:{source_label}"
    result["observed_pii"] = sorted(merged)

    from privacy_audit.engine.kb_verdict import finalize_observed_layers

    finalize_observed_layers(result, use_knowledge_base=use_knowledge_base)
    return result


def list_legacy_dynamic_apps(legacy_run_dir: Path | None) -> list[str]:
    if legacy_run_dir is None or not legacy_run_dir.exists():
        return []
    per_app = legacy_run_dir / "dynamic_testing" / "per_app"
    if not per_app.is_dir():
        return []
    return sorted(p.name for p in per_app.iterdir() if p.is_dir())


def build_and_export_evidence_bundles(
    app_names: list[str],
    output_dir: Path,
    taxonomy_file: Path,
    evidence_dir: Path | None = None,
    static_dir: Path | None = None,
    legacy_run_dir: Path | None = None,
    legacy_md_run_dir: Path | None = None,
    use_knowledge_base: bool = True,
    kb_min_confidence: float = 0.72,
) -> dict[str, Any]:
    """Generate per-app evidence bundle JSON files for batch consumption."""
    from privacy_audit.ingestion.output_mirror import (
        mirror_knowledge_base,
        mirror_oldsrc_test_cases,
        mirror_provenance_artifacts,
    )

    taxonomy = load_personal_info_taxonomy(taxonomy_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    mirror_knowledge_base()
    mirror_oldsrc_test_cases(legacy_run_dir)
    written: list[dict[str, Any]] = []
    for app_name in app_names:
        mirror_provenance_artifacts(
            app_name,
            legacy_run_dir=legacy_run_dir,
            legacy_md_run_dir=legacy_md_run_dir,
        )
        bundle = build_merged_evidence(
            app_name=app_name,
            taxonomy=taxonomy,
            evidence_dir=evidence_dir,
            static_dir=static_dir,
            legacy_run_dir=legacy_run_dir,
            legacy_md_run_dir=legacy_md_run_dir,
            use_knowledge_base=use_knowledge_base,
            kb_min_confidence=kb_min_confidence,
        )
        out_path = output_dir / f"{app_name}.json"
        out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        kb_meta = bundle.get("knowledge_base", {})
        written.append(
            {
                "app_name": app_name,
                "observed_count": len(bundle.get("observed_pii", [])),
                "feasible_evidence_count": kb_meta.get("feasible_count", 0),
                "path": str(out_path),
            }
        )
    return {
        "output_dir": str(output_dir),
        "apps": app_names,
        "written": written,
        "total_apps": len(app_names),
    }
