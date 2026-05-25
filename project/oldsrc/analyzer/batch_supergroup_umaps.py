"""
对某次流水线输出根目录下所有 cluster_analysis/sentences_cluster_full.jsonl
批量生成 umap_taxonomy_merge_cross.png、umap_taxonomy_merge_layers.png、
umap_taxonomy_merge_review_six.png、umap_taxonomy_merge_data_driven.png（不重算嵌入）。
会先根据全 run 样本写（或更新）macro_taxonomy_umap.json。

示例:
  python batch_supergroup_umaps.py --pipeline-root output/run_20260413_150151
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cluster_analysis import write_taxonomy_supergroup_plots_from_cluster_jsonl
from derive_macro_taxonomy_umap import derive_and_write_macro_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch regenerate taxonomy supergroup UMAP plots")
    ap.add_argument(
        "--pipeline-root",
        type=Path,
        required=True,
        help="含各应用子目录的 run 根目录（其下 **/cluster_analysis/sentences_cluster_full.jsonl）",
    )
    ap.add_argument(
        "--skip-derive",
        action="store_true",
        help="不重新计算 macro_taxonomy_umap.json（沿用已有）",
    )
    args = ap.parse_args()
    root: Path = args.pipeline_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    paths = sorted(root.glob("**/cluster_analysis/sentences_cluster_full.jsonl"))
    if not paths:
        raise SystemExit(f"No sentences_cluster_full.jsonl under {root}")

    macro_path = root / "macro_taxonomy_umap.json"
    if not args.skip_derive:
        derive_and_write_macro_json(root)

    results = []
    for j in paths:
        wrote = write_taxonomy_supergroup_plots_from_cluster_jsonl(j, macro_json=macro_path if macro_path.is_file() else None)
        results.append({"jsonl": str(j), "wrote": wrote})

    print(json.dumps({"pipeline_root": str(root), "processed": len(results), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
