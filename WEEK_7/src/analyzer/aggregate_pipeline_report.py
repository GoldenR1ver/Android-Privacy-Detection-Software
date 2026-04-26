"""
读取流水线输出根目录（含 pipeline_summary.json），
汇总各应用的 stats、audit 三标签正例数、聚类/送标产物是否存在，写入 week5_aggregate_report.json。
（WEEK_5 自包含工程默认由 run_pipeline.ps1 在每次 run_* 末尾调用。）
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    # utf-8-sig: pipeline_summary.json may be UTF-8 with BOM (PowerShell Set-Content -Encoding UTF8).
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _audit_counts(audit_csv: Path) -> Dict[str, int]:
    inc = incomp = incons = 0
    n = 0
    with audit_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            try:
                if int(str(row.get("incorrect", "0")).strip() or 0) == 1:
                    inc += 1
            except ValueError:
                pass
            try:
                if int(str(row.get("incomplete", "0")).strip() or 0) == 1:
                    incomp += 1
            except ValueError:
                pass
            try:
                if int(str(row.get("inconsistent", "0")).strip() or 0) == 1:
                    incons += 1
            except ValueError:
                pass
    return {"rows": n, "incorrect_1": inc, "incomplete_1": incomp, "inconsistent_1": incons}


def _count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def aggregate(pipeline_root: Path) -> Dict[str, Any]:
    summary_path = pipeline_root / "pipeline_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Missing pipeline_summary.json under {pipeline_root}")
    summary = _read_json(summary_path)
    docs = summary.get("documents") or []
    per_app: List[Dict[str, Any]] = []
    total_sent = 0
    for d in docs:
        doc_id = d.get("doc_id", "?")
        out_dir = Path(d.get("output_dir", ""))
        block: Dict[str, Any] = {"doc_id": doc_id, "output_dir": str(out_dir)}
        stats_path = out_dir / "stats.json"
        if stats_path.is_file():
            st = _read_json(stats_path)
            block["total_sentences"] = int(st.get("total_sentences", 0))
            total_sent += block["total_sentences"]
        else:
            block["total_sentences"] = None
        ap = d.get("audit_processed")
        if ap and Path(ap).is_file():
            block["audit_summary"] = _audit_counts(Path(ap))
        else:
            block["audit_summary"] = None
        ca_dir = out_dir / "cluster_analysis"
        ca_manifest = ca_dir / "cluster_analysis_manifest.json"
        block["cluster_analysis"] = {
            "dir": str(ca_dir) if ca_dir.is_dir() else None,
            "manifest_present": ca_manifest.is_file(),
        }
        block["labeling"] = {
            "for_labeling_jsonl_rows": _count_jsonl_rows(out_dir / "for_labeling.jsonl"),
            "review_bundle_present": (out_dir / "review_bundle.json").is_file(),
        }
        per_app.append(block)
    out = {
        "pipeline_root": str(pipeline_root.resolve()),
        "source_summary": str(summary_path),
        "total_sentences_all_apps": total_sent,
        "per_app": per_app,
        "flags_from_summary": {
            "bundle": summary.get("bundle"),
            "prepare_mode": summary.get("prepare_mode"),
            "provider": summary.get("provider"),
            "run_audit": summary.get("run_audit"),
        },
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="WEEK_5 aggregate report for a pipeline output directory.")
    p.add_argument("--pipeline-root", type=Path, required=True)
    args = p.parse_args()
    root = args.pipeline_root.resolve()
    data = aggregate(root)
    out_path = root / "week5_aggregate_report.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
