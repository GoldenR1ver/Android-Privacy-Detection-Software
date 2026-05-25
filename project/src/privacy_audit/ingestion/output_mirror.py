"""Mirror external artifacts under output/ so provenance paths stay output-relative."""

from __future__ import annotations

import shutil
from pathlib import Path

from privacy_audit.utils.path_utils import output_root


def _copy_file(src: Path, dst: Path) -> Path | None:
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _copy_tree_files(src_dir: Path, dst_dir: Path, pattern: str) -> list[Path]:
    if not src_dir.is_dir():
        return []
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in sorted(src_dir.glob(pattern)):
        if not src.is_file():
            continue
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def mirror_stage3_legacy(app_name: str, legacy_run_dir: Path | None) -> Path | None:
    """Copy per_app dynamic JSON into output/stage3_legacy/<app>/."""
    if legacy_run_dir is None:
        return None
    per_app = legacy_run_dir / "dynamic_testing" / "per_app" / app_name
    if not per_app.is_dir():
        return None
    dst_dir = output_root() / "stage3_legacy" / app_name
    copied = _copy_tree_files(per_app, dst_dir, "*.json")
    return dst_dir if copied else None


def mirror_stage2_legacy_md(app_name: str, legacy_md_run_dir: Path | None) -> Path | None:
    """Copy legacy markdown reports into output/stage2_legacy_md/<app>/."""
    if legacy_md_run_dir is None:
        return None
    src_dir = legacy_md_run_dir / app_name
    if not src_dir.is_dir():
        return None
    dst_dir = output_root() / "stage2_legacy_md" / app_name
    copied = _copy_tree_files(src_dir, dst_dir, "*.md")
    return dst_dir if copied else None


def mirror_oldsrc_test_cases(legacy_run_dir: Path | None) -> Path | None:
    """Copy shared Lab3 test_cases.json into output/oldsrc_shared/."""
    if legacy_run_dir is None:
        return None
    src = legacy_run_dir / "dynamic_testing" / "test_cases.json"
    dst = output_root() / "oldsrc_shared" / "test_cases.json"
    return _copy_file(src, dst)


def mirror_oldsrc_audit_csv(app_name: str, legacy_run_dir: Path | None) -> Path | None:
    """Copy audit_processed.csv into output/stage2_oldsrc_audit/<app>/."""
    if legacy_run_dir is None:
        return None
    src = legacy_run_dir / app_name / "audit_processed.csv"
    dst = output_root() / "stage2_oldsrc_audit" / app_name / "audit_processed.csv"
    return _copy_file(src, dst)


def mirror_knowledge_base(source_root: Path | None = None) -> Path | None:
    """Sync privacy_knowledgebase/knowledge_base into output/knowledge_base/."""
    if source_root is None:
        from privacy_knowledgebase.kb_store import default_kb_root

        source_root = default_kb_root()
    if not source_root.is_dir():
        return None
    dst_root = output_root() / "knowledge_base"
    dst_root.mkdir(parents=True, exist_ok=True)
    copied = False
    for src in sorted(source_root.glob("*.json")):
        shutil.copy2(src, dst_root / src.name)
        copied = True
    return dst_root if copied else None


def mirror_provenance_artifacts(
    app_name: str,
    *,
    legacy_run_dir: Path | None = None,
    legacy_md_run_dir: Path | None = None,
) -> None:
    mirror_stage3_legacy(app_name, legacy_run_dir)
    mirror_stage2_legacy_md(app_name, legacy_md_run_dir)
    mirror_oldsrc_audit_csv(app_name, legacy_run_dir)
    mirror_oldsrc_test_cases(legacy_run_dir)
