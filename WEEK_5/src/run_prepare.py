from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from labeling_queue import labeling_export_rows, write_labeling_jsonl
from pii_shots import load_shots_for_classify
from pipeline import build_rows_for_text
from review_store import build_bundle, save_bundle
from week3_csv import compute_privacy_stats, write_week3_sentence_csv


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            out = {k: v for k, v in row.items() if not str(k).startswith("_")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def build_manifest(
    args: argparse.Namespace,
    num_sentences: int,
    num_classified: int,
) -> Dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "mode": args.mode,
        "provider": args.provider if args.mode == "classify" else None,
        "ollama_model": args.ollama_model if args.mode == "classify" else None,
        "keyword_hint": True,
        "limit": args.limit,
        "max_chars": args.max_chars,
        "num_sentences": num_sentences,
        "num_classified": num_classified,
        "write_week3_csv": bool(args.write_week3_csv),
        "week3_csv_path": str(args.output_dir / "sentences_week3_2_2.csv")
        if args.write_week3_csv
        else None,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="Split privacy policy into sentences; optional LLM pii-related labels.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to policy .txt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for sentences.jsonl, manifest.json, stats.json",
    )
    parser.add_argument(
        "--mode",
        choices=("split-only", "classify"),
        default="split-only",
        help="split-only: no LLM; classify: call provider per sentence",
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "ollama", "deepseek"),
        default="deepseek",
        help="Used when mode=classify（默认 DeepSeek Chat API）",
    )
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    parser.add_argument(
        "--ollama-base-url",
        default="http://127.0.0.1:11434/api/chat",
    )
    parser.add_argument(
        "--deepseek-api-key",
        default=None,
        help="Or set env DEEPSEEK_API_KEY (required when provider=deepseek)",
    )
    parser.add_argument("--deepseek-model", default="deepseek-chat")
    parser.add_argument(
        "--deepseek-base-url",
        default="https://api.deepseek.com/chat/completions",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout seconds for ollama/deepseek",
    )
    parser.add_argument(
        "--keyword-hint",
        action="store_true",
        help="Deprecated: keyword_hint is always computed and stored in JSONL/stats.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Merge fragments so units do not exceed this many chars (0 = disabled)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max sentences to classify (0 = all); split-only ignores except for manifest",
    )
    parser.add_argument(
        "--write-week3-csv",
        action="store_true",
        help="Also write sentences_week3_2_2.csv (same columns as WEEK_3/src/2-2/data/*.csv)",
    )
    parser.add_argument(
        "--app-pkg",
        default="",
        help="app_pkg column for WEEK_3-style CSV (default empty)",
    )
    parser.add_argument(
        "--app-name",
        default="",
        help="app_name column (default: input file stem)",
    )
    parser.add_argument("--category-id", type=int, default=0)
    parser.add_argument(
        "--app-id-start",
        type=int,
        default=1,
        help="CSV app_id for first sentence = app_id_start + sent_index",
    )
    parser.add_argument(
        "--export-labeling-queue",
        action="store_true",
        help="Write for_labeling.jsonl: rows sorted by pii_related + keyword_hint (送标优先级)",
    )
    parser.add_argument(
        "--labeling-top-n",
        type=int,
        default=0,
        help="With --export-labeling-queue / --export-review-json, cap rows (0 = all)",
    )
    parser.add_argument(
        "--export-review-json",
        action="store_true",
        help="Write review_bundle.json（送标句 + AI + 待填 human）",
    )
    parser.add_argument(
        "--pii-shots-json",
        type=Path,
        default=None,
        help="Few-shot 参考文件（如 ref/shots.json）；仅 mode=classify 且 provider 非 mock 时注入提示",
    )
    parser.add_argument(
        "--pii-shots-max",
        type=int,
        default=12,
        help="每次分类最多带入几条 shot（控 token）",
    )
    parser.add_argument(
        "--cluster-embed-backend",
        choices=("flag", "sentence_transformers"),
        default="flag",
        help="与 sentence_cluster / cluster_analysis 一致",
    )
    parser.add_argument(
        "--cluster-model",
        default="BAAI/bge-small-zh-v1.5",
        help="嵌入模型 id 或本地目录",
    )
    parser.add_argument("--cluster-batch-size", type=int, default=32)
    parser.add_argument("--cluster-device", default="", help="如 cuda:0 / cpu；空则自动")
    parser.add_argument(
        "--bge-query-instruction",
        default="",
        help="BGE 检索前缀；空则按模型名自动",
    )
    parser.add_argument(
        "--cluster-use-fp16",
        choices=("auto", "true", "false"),
        default="auto",
    )
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.08)
    parser.add_argument("--umap-metric", default="cosine")
    parser.add_argument("--umap-random-state", type=int, default=42)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=8)
    parser.add_argument("--hdbscan-min-samples", type=int, default=3)
    parser.add_argument(
        "--audit-ds-max-chars",
        type=int,
        default=12000,
        help="簇内 DS 文本最大字符数（控 token）",
    )
    parser.add_argument(
        "--audit-ds-max-peers",
        type=int,
        default=50,
        help="簇内最多拼接多少条同伴句",
    )
    args = parser.parse_args()

    ds_key = (args.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if args.mode == "classify" and args.provider == "deepseek" and not ds_key:
        parser.error("provider=deepseek requires --deepseek-api-key or DEEPSEEK_API_KEY")

    pii_shots: List[Dict[str, Any]] = []
    if args.mode == "classify" and args.provider != "mock" and args.pii_shots_json:
        pii_shots = load_shots_for_classify(args.pii_shots_json, max_n=args.pii_shots_max)

    doc_id = args.input.stem
    raw = load_text(args.input)
    lim = args.limit if args.limit and args.limit > 0 else None

    rows, num_classified = build_rows_for_text(
        raw,
        doc_id,
        mode=args.mode,
        provider=args.provider,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        deepseek_api_key=ds_key,
        deepseek_model=args.deepseek_model,
        deepseek_base_url=args.deepseek_base_url,
        timeout_sec=args.timeout,
        limit=lim,
        max_chars=args.max_chars,
        pii_shots=pii_shots or None,
    )

    out_dir = args.output_dir
    pre_audit_clustering_applied = False
    if args.write_week3_csv:
        if not os.getenv("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        if len(rows) >= 2:
            from sentence_cluster import run_clustering

            ufp: Optional[bool] = None
            if args.cluster_use_fp16 == "true":
                ufp = True
            elif args.cluster_use_fp16 == "false":
                ufp = False
            clustered, summary, _, _ = run_clustering(
                rows,
                embed_backend=args.cluster_embed_backend,
                model_name=args.cluster_model,
                batch_size=args.cluster_batch_size,
                device=(args.cluster_device.strip() or None),
                query_instruction=(args.bge_query_instruction.strip() or None),
                use_fp16=ufp,
                umap_neighbors=args.umap_neighbors,
                umap_min_dist=args.umap_min_dist,
                umap_metric=args.umap_metric,
                umap_random_state=args.umap_random_state,
                hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
                hdbscan_min_samples=args.hdbscan_min_samples,
                max_samples_per_cluster=5,
            )
            for r, cr in zip(rows, clustered, strict=True):
                r["cluster_id"] = cr["cluster_id"]
                r["is_cluster_noise"] = cr["is_cluster_noise"]
                r["umap_x_pre_audit"] = cr["umap_x"]
                r["umap_y_pre_audit"] = cr["umap_y"]
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "pre_audit_cluster_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            pre_audit_clustering_applied = True

    write_jsonl(rows, out_dir / "sentences.jsonl")

    label_top = args.labeling_top_n if args.labeling_top_n > 0 else None
    labeling_info = None
    if args.export_labeling_queue:
        n_q = write_labeling_jsonl(rows, out_dir / "for_labeling.jsonl", limit=label_top)
        labeling_info = {"path": str(out_dir / "for_labeling.jsonl"), "rows": n_q}

    review_info = None
    if args.export_review_json:
        export_rows = labeling_export_rows(rows, limit=label_top, include_meta=True)
        src = (
            str((out_dir / "for_labeling.jsonl").resolve())
            if args.export_labeling_queue
            else str((out_dir / "sentences.jsonl").resolve())
        )
        bundle = build_bundle(export_rows, source_path=src, note="run_prepare")
        save_bundle(out_dir / "review_bundle.json", bundle)
        review_info = {
            "path": str((out_dir / "review_bundle.json").resolve()),
            "items": len(export_rows),
        }

    app_name = args.app_name.strip() or doc_id
    if args.write_week3_csv:
        csv_path = out_dir / "sentences_week3_2_2.csv"
        write_week3_sentence_csv(
            csv_path,
            rows,
            app_pkg=args.app_pkg,
            app_name=app_name,
            category_id=args.category_id,
            app_id_start=args.app_id_start,
            ds_mode="cluster_peers" if pre_audit_clustering_applied else "empty",
            cluster_ds_max_chars=args.audit_ds_max_chars,
            cluster_ds_max_peers=args.audit_ds_max_peers,
        )

    stats = compute_privacy_stats(rows)
    stats["doc_id"] = doc_id
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = build_manifest(args, len(rows), num_classified)
    if args.write_week3_csv:
        manifest["pre_audit_clustering"] = {
            "applied": pre_audit_clustering_applied,
            "reason_skipped": None
            if pre_audit_clustering_applied
            else ("need_at_least_2_sentences" if len(rows) < 2 else "unknown"),
            "cluster_embed_backend": args.cluster_embed_backend,
            "cluster_model": args.cluster_model,
            "audit_ds_max_chars": args.audit_ds_max_chars,
            "audit_ds_max_peers": args.audit_ds_max_peers,
            "pre_audit_cluster_summary": str((out_dir / "pre_audit_cluster_summary.json").resolve())
            if pre_audit_clustering_applied
            else None,
        }
    if labeling_info:
        manifest["labeling_queue"] = labeling_info
    if review_info:
        manifest["review_bundle"] = review_info
    if pii_shots:
        manifest["pii_shots"] = {
            "path": str(args.pii_shots_json.resolve()) if args.pii_shots_json else None,
            "used_in_prompt": len(pii_shots),
            "max_configured": args.pii_shots_max,
        }
    manifest["stats_summary"] = {
        "total_sentences": stats["total_sentences"],
        "pii_related_ratio_of_total": stats["pii_related"]["ratio_of_total"],
        "pii_related_ratio_of_labeled": stats["pii_related"]["ratio_of_labeled"],
        "keyword_hint_ratio_of_total": stats["keyword_hint"]["ratio_of_total"],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_msg: Dict[str, Any] = {
        "wrote": str(out_dir),
        "sentences": len(rows),
        "classified": num_classified,
        "stats": manifest["stats_summary"],
    }
    if args.write_week3_csv:
        out_msg["pre_audit_clustering_applied"] = pre_audit_clustering_applied
    if labeling_info:
        out_msg["labeling_queue"] = labeling_info
    if review_info:
        out_msg["review_bundle"] = review_info
    print(json.dumps(out_msg, ensure_ascii=False))


if __name__ == "__main__":
    main()
