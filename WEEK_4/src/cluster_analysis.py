"""
强化聚类与合规维度分析（WEEK_4）：

1. 读取 sentences.jsonl（句向量）与可选 audit_processed.csv，合并 incorrect / incomplete / inconsistent。
2. 加载 22 项标准 + 个保法参考段落 JSON，用句向量与参考段落向量做最近类（余弦相似度）。
3. UMAP 降维后：按 HDBSCAN 簇、按 audit 优先级、按 22 项 best-id 分别着色出图。
4. 输出 policy_vs_law_comparison.json：每句到 22 个法参考维度的相似度摘要、按项聚合的对比统计。

依赖：与 sentence_cluster.py 相同（sentence-transformers、umap-learn、hdbscan、matplotlib）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib_zh import configure_matplotlib_chinese_font

from sentence_cluster import (
    _encode_texts,
    _encode_texts_deepseek_api,
    _hdbscan_labels,
    _umap_reduce,
    load_jsonl,
    write_jsonl,
)

configure_matplotlib_chinese_font()

_DEFAULT_ITEMS = Path(__file__).resolve().parent.parent / "ref" / "taxonomy" / "pipl_22_items.json"
_DEFAULT_LAW = Path(__file__).resolve().parent.parent / "ref" / "taxonomy" / "pipl_law_reference_segments.json"


def load_taxonomy_pair(
    items_path: Path,
    law_segments_path: Path,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    items_obj = json.loads(items_path.read_text(encoding="utf-8"))
    law_obj = json.loads(law_segments_path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = sorted(items_obj["items"], key=lambda x: int(x["id"]))
    seg_by_id = {int(s["item_id"]): s for s in law_obj["segments"]}
    law_texts: List[str] = []
    for it in items:
        sid = int(it["id"])
        seg = seg_by_id.get(sid)
        if not seg:
            raise ValueError(f"Missing law segment for item_id={sid}")
        law_texts.append(str(seg["text_for_embedding"]))
    return items, law_texts


def _parse_audit_int(v: Any) -> int:
    try:
        return int(str(v).strip() or 0)
    except ValueError:
        return 0


def merge_audit_into_sentences(
    sentences: List[Dict[str, Any]],
    audit_csv: Optional[Path],
) -> List[Dict[str, Any]]:
    if not audit_csv or not audit_csv.is_file():
        for r in sentences:
            r["incorrect"] = 0
            r["incomplete"] = 0
            r["inconsistent"] = 0
            r["audit_priority"] = 0
        return sentences
    with audit_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(sentences):
        raise ValueError(
            f"audit 行数 ({len(rows)}) 与 sentences.jsonl ({len(sentences)}) 不一致；"
            "请保证同一流水线、同序导出，或先裁剪为等长子集。"
        )
    out: List[Dict[str, Any]] = []
    for s, a in zip(sentences, rows):
        nr = dict(s)
        inc = _parse_audit_int(a.get("incorrect"))
        incomp = _parse_audit_int(a.get("incomplete"))
        incons = _parse_audit_int(a.get("inconsistent"))
        nr["incorrect"] = inc
        nr["incomplete"] = incomp
        nr["inconsistent"] = incons
        if incons == 1:
            nr["audit_priority"] = 3
        elif incomp == 1:
            nr["audit_priority"] = 2
        elif inc == 1:
            nr["audit_priority"] = 1
        else:
            nr["audit_priority"] = 0
        out.append(nr)
    return out


def _encode_backend(
    texts: List[str],
    *,
    embed_backend: str,
    model_name: str,
    batch_size: int,
    device: Optional[str],
    deepseek_api_key: str,
    deepseek_embedding_model: str,
    deepseek_embeddings_url: str,
    timeout_sec: int,
) -> np.ndarray:
    if embed_backend == "local":
        return _encode_texts(texts, model_name, batch_size, device)
    if embed_backend == "deepseek-api":
        return _encode_texts_deepseek_api(
            texts,
            api_key=deepseek_api_key,
            model=deepseek_embedding_model,
            url=deepseek_embeddings_url,
            batch_size=batch_size,
            timeout_sec=timeout_sec,
        )
    raise ValueError(embed_backend)


def taxonomy_scores_and_best(
    sent_emb: np.ndarray,
    law_emb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """余弦相似度（向量已 L2 归一化时即为点积）。"""
    sim = np.asarray(sent_emb @ law_emb.T, dtype=np.float32)
    best_idx = np.argmax(sim, axis=1).astype(np.int32)
    best_score = sim[np.arange(len(sim)), best_idx].tolist()
    return sim, best_idx, best_score


def _plot_umap_colored_int(
    xy: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str,
    label_names: Dict[int, str],
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    uniq = sorted(set(int(x) for x in np.unique(labels)))
    cmap = plt.cm.get_cmap("tab10")
    for i, u in enumerate(uniq):
        m = labels == u
        name = label_names.get(u, str(u))
        c = cmap(i / max(1, len(uniq) - 1))
        ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.75, color=c, label=name)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_umap_taxonomy_22(xy: np.ndarray, best_ids: np.ndarray, out_path: Path) -> None:
    """22 类用离散色（HSV 分圈）。"""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    bid = best_ids.astype(np.int32)
    for k in range(1, 23):
        m = bid == (k - 1)
        if not np.any(m):
            continue
        hue = (k - 1) / 22.0
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.7, color=plt.cm.hsv(hue), label=f"{k}")
    ax.set_title("UMAP 着色：22 项标准（嵌入最近邻）")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=5, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_comparison_report(
    sim: np.ndarray,
    best_idx: np.ndarray,
    best_score: List[float],
    items: List[Dict[str, Any]],
    law_texts: List[str],
) -> Dict[str, Any]:
    n_item = sim.shape[1]
    per_item: List[Dict[str, Any]] = []
    for j in range(n_item):
        item_id = j + 1
        mask = best_idx == j
        cnt = int(np.sum(mask))
        if cnt > 0:
            mean_cos_assigned = float(np.mean(sim[mask, j]))
            mean_cos_all = float(np.mean(sim[:, j]))
        else:
            mean_cos_assigned = None
            mean_cos_all = float(np.mean(sim[:, j]))
        per_item.append(
            {
                "item_id": item_id,
                "title": items[j]["title"],
                "assigned_sentence_count": cnt,
                "mean_cosine_to_own_law_ref_when_assigned": mean_cos_assigned,
                "mean_cosine_to_law_ref_over_all_sentences": mean_cos_all,
            }
        )
    cross = np.zeros((n_item, n_item), dtype=np.float32)
    for j_assigned in range(n_item):
        m = best_idx == j_assigned
        if not np.any(m):
            continue
        for i_law in range(n_item):
            cross[j_assigned, i_law] = float(np.mean(sim[m, i_law]))
    return {
        "description": "rows=j_assigned(句子被判为最近邻的项), cols=i_law(与第 i 个法参考段的平均余弦相似度)",
        "per_item_summary": per_item,
        "cross_mean_cosine_assigned_bucket_vs_law_dim": cross.tolist(),
    }


def run(
    sentences_path: Path,
    output_dir: Path,
    *,
    items_path: Path,
    law_segments_path: Path,
    audit_csv: Optional[Path],
    embed_backend: str,
    model_name: str,
    batch_size: int,
    device: Optional[str],
    deepseek_api_key: str,
    deepseek_embedding_model: str,
    deepseek_embeddings_url: str,
    timeout_sec: int,
    umap_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_random_state: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    skip_hdbscan_plot: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    items, law_texts = load_taxonomy_pair(items_path, law_segments_path)
    sentences = load_jsonl(sentences_path)
    if len(sentences) < 2:
        raise SystemExit("Need at least 2 sentences.")
    sentences = merge_audit_into_sentences(sentences, audit_csv)

    sent_texts = [str(r.get("text", "")).strip() or " " for r in sentences]
    all_texts = law_texts + sent_texts
    emb_all = _encode_backend(
        all_texts,
        embed_backend=embed_backend,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        deepseek_api_key=deepseek_api_key,
        deepseek_embedding_model=deepseek_embedding_model,
        deepseek_embeddings_url=deepseek_embeddings_url,
        timeout_sec=timeout_sec,
    )
    k = len(law_texts)
    law_emb = emb_all[:k]
    sent_emb = emb_all[k:]
    sim, best_idx, best_score = taxonomy_scores_and_best(sent_emb, law_emb)

    out_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(sentences):
        bid = int(best_idx[i])
        item = items[bid]
        nr = dict(r)
        nr["taxonomy_22_id"] = int(item["id"])
        nr["taxonomy_22_title"] = item["title"]
        nr["taxonomy_22_group"] = item["group"]
        nr["taxonomy_22_best_cosine"] = float(best_score[i])
        nr["taxonomy_22_scores"] = [float(x) for x in sim[i].tolist()]
        out_rows.append(nr)

    write_jsonl(out_rows, output_dir / "sentences_taxonomy_22.jsonl")

    xy = _umap_reduce(
        sent_emb,
        umap_neighbors,
        umap_min_dist,
        umap_metric,
        umap_random_state,
    )
    labels_hdb = _hdbscan_labels(xy, hdbscan_min_cluster_size, hdbscan_min_samples)
    for i, r in enumerate(out_rows):
        r["umap_x"] = float(xy[i, 0])
        r["umap_y"] = float(xy[i, 1])
        r["cluster_id_hdbscan"] = int(labels_hdb[i])
        r["is_cluster_noise"] = bool(labels_hdb[i] < 0)

    write_jsonl(out_rows, output_dir / "sentences_cluster_full.jsonl")

    # --- plots ---
    if not skip_hdbscan_plot:
        from sentence_cluster import _plot_scatter

        _plot_scatter(
            xy,
            labels_hdb,
            output_dir / "umap_hdbscan.png",
            title="UMAP + HDBSCAN（句向量）",
        )

    audit_pri = np.array([int(r.get("audit_priority", 0)) for r in out_rows], dtype=np.int32)
    _plot_umap_colored_int(
        xy,
        audit_pri,
        output_dir / "umap_audit_priority.png",
        title="UMAP：Data Safety 句级审计标签（优先级着色）",
        label_names={
            0: "无三标签",
            1: "incorrect",
            2: "incomplete",
            3: "inconsistent",
        },
    )

    _plot_umap_taxonomy_22(xy, best_idx, output_dir / "umap_taxonomy_22.png")

    report = build_comparison_report(sim, best_idx, best_score, items, law_texts)
    report["embed_backend"] = embed_backend
    report["model_name"] = model_name if embed_backend == "local" else deepseek_embedding_model
    report["items_path"] = str(items_path)
    report["law_segments_path"] = str(law_segments_path)
    report["sentences_source"] = str(sentences_path)
    report["audit_csv"] = str(audit_csv) if audit_csv else None
    (output_dir / "policy_vs_law_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "wrote": [
            "sentences_taxonomy_22.jsonl",
            "sentences_cluster_full.jsonl",
            "policy_vs_law_comparison.json",
            "umap_audit_priority.png",
            "umap_taxonomy_22.png",
        ]
        + ([] if skip_hdbscan_plot else ["umap_hdbscan.png"]),
        "rows": len(out_rows),
        "num_law_refs": k,
    }
    (output_dir / "cluster_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(
        description="WEEK_4 强化聚类：audit 三标签 + 22 项个保法维度 + UMAP/HDBSCAN + 法参考对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例（需已安装 requirements-optional.txt）：\n"
            "  python cluster_analysis.py --sentences-jsonl ..\\out\\pipeline_xxx\\Soul\\sentences.jsonl "
            "--audit-processed ..\\out\\pipeline_xxx\\Soul\\audit_processed.csv "
            "--output-dir ..\\out\\pipeline_xxx\\Soul\\cluster_analysis\n"
            "若无 audit CSV，可省略 --audit-processed，则三标签全 0，仅做 22 项与聚类。"
        ),
    )
    p.add_argument("--sentences-jsonl", type=Path, required=True)
    p.add_argument("--audit-processed", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--items-json", type=Path, default=_DEFAULT_ITEMS)
    p.add_argument("--law-segments-json", type=Path, default=_DEFAULT_LAW)
    p.add_argument("--embed-backend", choices=("local", "deepseek-api"), default="local")
    p.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--deepseek-api-key", default=None)
    p.add_argument("--deepseek-embedding-model", default="deepseek-embedding")
    p.add_argument(
        "--deepseek-embeddings-url",
        default="https://api.deepseek.com/v1/embeddings",
    )
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.08)
    p.add_argument("--umap-metric", default="cosine")
    p.add_argument("--umap-random-state", type=int, default=42)
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=8)
    p.add_argument("--hdbscan-min-samples", type=int, default=3)
    p.add_argument("--no-hdbscan-plot", action="store_true")
    args = p.parse_args()

    ds_key = (args.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if args.embed_backend == "deepseek-api" and not ds_key:
        raise SystemExit("deepseek-api 需要 --deepseek-api-key 或 DEEPSEEK_API_KEY")

    run(
        args.sentences_jsonl,
        args.output_dir,
        items_path=args.items_json,
        law_segments_path=args.law_segments_json,
        audit_csv=args.audit_processed,
        embed_backend=args.embed_backend,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        deepseek_api_key=ds_key,
        deepseek_embedding_model=args.deepseek_embedding_model,
        deepseek_embeddings_url=args.deepseek_embeddings_url.strip(),
        timeout_sec=args.timeout,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        umap_random_state=args.umap_random_state,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        skip_hdbscan_plot=args.no_hdbscan_plot,
    )


if __name__ == "__main__":
    main()
