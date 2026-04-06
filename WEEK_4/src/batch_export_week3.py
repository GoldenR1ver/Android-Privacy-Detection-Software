from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from pipeline import build_rows_for_text
from week3_csv import compute_privacy_stats, write_week3_sentence_csv


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            out = {k: v for k, v in row.items() if not str(k).startswith("_")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-split policies in a folder; merge JSONL + WEEK_3-style CSV + stats.",
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--glob",
        default="*.txt",
        help="Glob relative to input-dir (default: *.txt)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Process at most N files (0 = all), sorted by name",
    )
    parser.add_argument(
        "--mode",
        choices=("split-only", "classify"),
        default="split-only",
    )
    parser.add_argument("--provider", choices=("mock", "ollama", "deepseek"), default="mock")
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--deepseek-api-key", default=None)
    parser.add_argument("--deepseek-model", default="deepseek-chat")
    parser.add_argument(
        "--deepseek-base-url",
        default="https://api.deepseek.com/chat/completions",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit-per-doc", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=0)
    parser.add_argument("--app-pkg-prefix", default="", help="Prefix for export_app_pkg per doc")
    parser.add_argument("--category-id", type=int, default=0)
    parser.add_argument("--app-id-start", type=int, default=1)
    args = parser.parse_args()

    ds_key = (args.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if args.mode == "classify" and args.provider == "deepseek" and not ds_key:
        parser.error("provider=deepseek requires --deepseek-api-key or DEEPSEEK_API_KEY")

    paths = sorted(args.input_dir.glob(args.glob))
    if args.max_files and args.max_files > 0:
        paths = paths[: args.max_files]

    lim = args.limit_per_doc if args.limit_per_doc > 0 else None
    all_rows: List[Dict[str, Any]] = []
    by_doc: Dict[str, Any] = {}
    next_id = args.app_id_start
    total_classified = 0

    for path in paths:
        if not path.is_file():
            continue
        doc_id = path.stem
        rows, n_cls = build_rows_for_text(
            load_text(path),
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
        )
        total_classified += n_cls
        pkg = f"{args.app_pkg_prefix}{doc_id}" if args.app_pkg_prefix else doc_id
        for r in rows:
            r["_export_app_id"] = next_id
            r["export_app_pkg"] = pkg
            r["export_app_name"] = doc_id
            r["export_category_id"] = args.category_id
            next_id += 1
            all_rows.append(r)
        by_doc[path.name] = compute_privacy_stats(rows)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(all_rows, out / "all_sentences.jsonl")

    write_week3_sentence_csv(
        out / "all_sentences_week3_2_2.csv",
        all_rows,
        app_pkg="",
        app_name="",
        category_id=args.category_id,
        app_id_start=args.app_id_start,
    )

    agg = compute_privacy_stats(all_rows)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "glob": args.glob,
        "files_processed": len(by_doc),
        "total_sentences": agg["total_sentences"],
        "aggregate_stats": agg,
        "by_file": by_doc,
        "mode": args.mode,
        "limit_per_doc": args.limit_per_doc,
    }
    (out / "batch_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "wrote": str(out),
                "files": len(by_doc),
                "sentences": len(all_rows),
                "classified": total_classified,
                "aggregate": {
                    "keyword_hint_ratio": agg["keyword_hint"]["ratio_of_total"],
                    "pii_related_ratio_of_total": agg["pii_related"]["ratio_of_total"],
                    "pii_related_ratio_of_labeled": agg["pii_related"]["ratio_of_labeled"],
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
