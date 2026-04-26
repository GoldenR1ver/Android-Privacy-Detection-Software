"""
强化聚类与合规维度分析（WEEK_5 自包含工程内）：

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 合组方案：键为 1..22 的 taxonomy_22_id。同一合组内所有分点使用同一颜色；跨组合组用 first-wins（先出现的合组优先）。
TAXONOMY_SUPERGROUP_SCHEMES: Dict[str, List[Tuple[str, frozenset]]] = {
    "merge_cross": [
        ("核心告知与行权路径", frozenset({2, 3, 5, 6, 7, 8})),
        ("特定告知与单独同意", frozenset({9, 10, 14, 20})),
        ("决策透明与用户控制", frozenset({13, 8, 22})),
    ],
    "merge_layers": [
        ("格式与可执行性基础", frozenset({17, 18, 19, 21, 22})),
        ("核心告知义务", frozenset({1, 2, 3, 4, 5, 6, 7, 8})),
        ("增强告知与特殊程序义务", frozenset({9, 10, 11, 12, 13, 14, 15, 16, 20})),
    ],
    # 六分审查视角：互斥覆盖 1..22，与「内容交叉」「格式/核心/增强」并列的第三种合组图
    "merge_review_six": [
        ("一、主体与基础告知义务", frozenset({1, 2, 4})),
        ("二、用户权利与行权通路", frozenset({3, 5, 6, 7, 8})),
        ("三、高风险场景实体规则告知", frozenset({9, 10, 11, 13, 14, 15})),
        ("四、第三方处理与数据流转告知", frozenset({12, 16})),
        ("五、同意与授权的程序性规范", frozenset({20, 22})),
        ("六、文本的形式与实质真实性", frozenset({17, 18, 19, 21})),
    ],
}
SUPERGROUP_COLORS_DEFAULT = ["#1f77b4", "#2ca02c", "#d62728"]
SUPERGROUP_COLORS_ALT = ["#9467bd", "#ff7f0e", "#17becf"]
SUPERGROUP_COLORS_SIX = [
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
]
# 数据驱动大类（最多 7 类）
SUPERGROUP_COLORS_DATA_DRIVEN = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]


def load_macro_groups_from_json(path: Path) -> Tuple[List[Tuple[str, frozenset]], Dict[str, Any]]:
    """读取 derive_macro_taxonomy_umap.py 写出的 macro_taxonomy_umap.json。"""
    meta = json.loads(path.read_text(encoding="utf-8"))
    raw_groups = meta.get("groups") or []
    ordered = sorted(raw_groups, key=lambda g: int(g.get("macro_id", 0)))
    groups: List[Tuple[str, frozenset]] = []
    for g in ordered:
        name = str(g.get("label", "?"))
        ids = frozenset(int(x) for x in g.get("item_ids", []))
        groups.append((name, ids))
    return groups, meta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib_zh import configure_matplotlib_chinese_font

from sentence_cluster import (
    _encode_texts,
    _hdbscan_labels,
    _umap_reduce,
    load_jsonl,
    write_jsonl,
)
from umap_multiclass_plots import plot_umap_pii_keyword_joint, plot_umap_taxonomy_group

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
    if len(rows) > len(sentences):
        raise ValueError(
            f"audit 行数 ({len(rows)}) 多于 sentences.jsonl ({len(sentences)})；"
            "请检查 CSV 是否与当前 jsonl 对应。"
        )
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(sentences):
        nr = dict(s)
        if i < len(rows):
            a = rows[i]
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
        else:
            nr["incorrect"] = 0
            nr["incomplete"] = 0
            nr["inconsistent"] = 0
            nr["audit_priority"] = 0
        out.append(nr)
    return out


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
    cmap = (
        matplotlib.colormaps["tab10"]
        if hasattr(matplotlib, "colormaps")
        else plt.cm.get_cmap("tab10")
    )
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


def _build_item_to_supergroup_first_wins(
    groups: List[Tuple[str, frozenset]],
) -> Dict[int, int]:
    """taxonomy_22_id（1..22）→ 合组下标 0..G-1；先列出的合组优先（解决 8、13 等跨组合组）。"""
    m: Dict[int, int] = {}
    for gi, (_name, item_ids) in enumerate(groups):
        for tid in item_ids:
            if tid not in m:
                m[tid] = gi
    return m


def _plot_umap_taxonomy_supergroups(
    xy: np.ndarray,
    taxonomy_ids_1based: np.ndarray,
    out_path: Path,
    title: str,
    groups: List[Tuple[str, frozenset]],
    colors: List[str],
    other_label: str = "其它（最近邻项未落在上述合组）",
    *,
    figsize: Tuple[float, float] = (9, 6.5),
    legend_fontsize: Optional[float] = None,
) -> None:
    """按合组着色：同一合组内各分点同色。"""
    item_to_g = _build_item_to_supergroup_first_wins(groups)
    tid = taxonomy_ids_1based.astype(np.int32)
    group_idx = np.full(len(tid), -1, dtype=np.int32)
    for i in range(len(tid)):
        t = int(tid[i])
        if t in item_to_g:
            group_idx[i] = item_to_g[t]

    fig, ax = plt.subplots(figsize=figsize)
    # 先画其它（底层）
    m_other = group_idx < 0
    if np.any(m_other):
        ax.scatter(
            xy[m_other, 0],
            xy[m_other, 1],
            s=8,
            alpha=0.35,
            color="#bfbfbf",
            label=other_label,
            rasterized=True,
        )
    for gi, (gname, _) in enumerate(groups):
        m = group_idx == gi
        if not np.any(m):
            continue
        c = colors[gi % len(colors)]
        ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.78, color=c, label=gname, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ncol = 2 if len(groups) > 4 else 1
    if legend_fontsize is not None:
        leg_fs = legend_fontsize
    else:
        leg_fs = 5.8 if len(groups) > 4 else 8
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=leg_fs,
        ncol=ncol,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_taxonomy_supergroup_plots(
    xy: np.ndarray,
    taxonomy_ids_1based: np.ndarray,
    output_dir: Path,
    macro_json: Optional[Path] = None,
) -> List[str]:
    """在已有 UMAP 坐标与每句 taxonomy_22_id 上写出合组 UMAP 图（交叉 / 三层 / 六分审查 + 可选数据驱动）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    wrote: List[str] = []
    cross = TAXONOMY_SUPERGROUP_SCHEMES["merge_cross"]
    layers = TAXONOMY_SUPERGROUP_SCHEMES["merge_layers"]
    review_six = TAXONOMY_SUPERGROUP_SCHEMES["merge_review_six"]
    p_cross = output_dir / "umap_taxonomy_merge_cross.png"
    p_layers = output_dir / "umap_taxonomy_merge_layers.png"
    p_six = output_dir / "umap_taxonomy_merge_review_six.png"
    _plot_umap_taxonomy_supergroups(
        xy,
        taxonomy_ids_1based,
        p_cross,
        "UMAP：内容交叉合组（22 项按合组同色）",
        cross,
        SUPERGROUP_COLORS_DEFAULT,
    )
    wrote.append(p_cross.name)
    _plot_umap_taxonomy_supergroups(
        xy,
        taxonomy_ids_1based,
        p_layers,
        "UMAP：格式 / 核心告知 / 增强告知（合组同色）",
        layers,
        SUPERGROUP_COLORS_ALT,
    )
    wrote.append(p_layers.name)
    _plot_umap_taxonomy_supergroups(
        xy,
        taxonomy_ids_1based,
        p_six,
        "UMAP：六分合规审查视角（互斥覆盖 1–22，合组同色）",
        review_six,
        SUPERGROUP_COLORS_SIX,
    )
    wrote.append(p_six.name)

    mj = macro_json
    if mj is None:
        cand = output_dir.parents[2] / "macro_taxonomy_umap.json"
        if cand.is_file():
            mj = cand
    if mj is not None and mj.is_file():
        dd_groups, meta = load_macro_groups_from_json(mj)
        p_dd = output_dir / "umap_taxonomy_merge_data_driven.png"
        k = int(meta.get("best_k", len(dd_groups)))
        sil = float(meta.get("best_mean_silhouette", 0.0))
        _plot_umap_taxonomy_supergroups(
            xy,
            taxonomy_ids_1based,
            p_dd,
            f"UMAP：六分命名大类（K={k}，轮廓≈{sil:.3f}）",
            dd_groups,
            SUPERGROUP_COLORS_DATA_DRIVEN,
            figsize=(10.5, 6.8),
            legend_fontsize=4.85,
        )
        wrote.append(p_dd.name)

    return wrote


def write_taxonomy_supergroup_plots_from_cluster_jsonl(
    cluster_jsonl: Path,
    macro_json: Optional[Path] = None,
) -> List[str]:
    """从已生成的 sentences_cluster_full.jsonl 仅重绘合组 UMAP（不重算嵌入）。"""
    rows = load_jsonl(cluster_jsonl)
    if not rows:
        return []
    xy = np.zeros((len(rows), 2), dtype=np.float64)
    tid = np.zeros(len(rows), dtype=np.int32)
    for i, r in enumerate(rows):
        xy[i, 0] = float(r.get("umap_x", 0.0))
        xy[i, 1] = float(r.get("umap_y", 0.0))
        tid[i] = int(r.get("taxonomy_22_id", 0))
    out_dir = cluster_jsonl.parent
    return write_taxonomy_supergroup_plots(xy, tid, out_dir, macro_json=macro_json)


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
    query_instruction: Optional[str],
    use_fp16: Optional[bool],
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
    emb_all = _encode_texts(
        all_texts,
        model_name,
        batch_size,
        device,
        embed_backend=embed_backend,
        query_instruction=query_instruction,
        use_fp16=use_fp16,
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

    plot_umap_taxonomy_group(out_rows, output_dir / "umap_taxonomy_group.png")
    plot_umap_pii_keyword_joint(out_rows, output_dir / "umap_pii_keyword_joint.png")

    tax_ids_1b = (best_idx.astype(np.int32) + 1).clip(1, 22)
    super_wrote = write_taxonomy_supergroup_plots(xy, tax_ids_1b, output_dir)

    report = build_comparison_report(sim, best_idx, best_score, items, law_texts)
    report["embed_backend"] = embed_backend
    report["model_name"] = model_name
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
            "umap_taxonomy_group.png",
            "umap_pii_keyword_joint.png",
        ]
        + super_wrote
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
        description="强化聚类：audit 三标签 + 22 项个保法维度 + UMAP/HDBSCAN + 法参考对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例（需已安装 requirements-optional.txt）：\n"
            "  python cluster_analysis.py --sentences-jsonl ..\\out\\pipeline_xxx\\Soul\\sentences.jsonl "
            "--audit-processed ..\\out\\pipeline_xxx\\Soul\\audit_processed.csv "
            "--output-dir ..\\out\\pipeline_xxx\\Soul\\cluster_analysis\n"
            "若无 audit CSV，可省略 --audit-processed，则三标签全 0，仅做 22 项与聚类。"
        ),
    )
    p.add_argument(
        "--sentences-jsonl",
        type=Path,
        default=None,
        help="句向量 jsonl（--only-supergroup-plots 时可省略）",
    )
    p.add_argument("--audit-processed", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--items-json", type=Path, default=_DEFAULT_ITEMS)
    p.add_argument("--law-segments-json", type=Path, default=_DEFAULT_LAW)
    p.add_argument(
        "--embed-backend",
        choices=("flag", "sentence_transformers"),
        default="flag",
        help="flag=FlagEmbedding BGE；sentence_transformers=原 ST 路径",
    )
    p.add_argument(
        "--model",
        default="BAAI/bge-small-zh-v1.5",
        help="HF 模型 id 或本地目录",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--bge-query-instruction",
        default="",
        help="BGE 检索前缀；空则按模型名自动（见仓库 README_zh 模型表）",
    )
    p.add_argument(
        "--use-fp16",
        choices=("auto", "true", "false"),
        default="auto",
    )
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.08)
    p.add_argument("--umap-metric", default="cosine")
    p.add_argument("--umap-random-state", type=int, default=42)
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=8)
    p.add_argument("--hdbscan-min-samples", type=int, default=3)
    p.add_argument("--no-hdbscan-plot", action="store_true")
    p.add_argument(
        "--only-supergroup-plots",
        action="store_true",
        help="仅从已有 sentences_cluster_full.jsonl 重绘合组 UMAP（不重算嵌入）；需配合 --output-dir 指向含该 jsonl 的 cluster_analysis 目录",
    )
    args = p.parse_args()

    if args.only_supergroup_plots:
        jpath = args.output_dir / "sentences_cluster_full.jsonl"
        if not jpath.is_file():
            raise SystemExit(f"--only-supergroup-plots 需要 {jpath} 存在")
        wrote = write_taxonomy_supergroup_plots_from_cluster_jsonl(jpath)
        print(json.dumps({"mode": "only_supergroup_plots", "wrote": wrote}, ensure_ascii=False, indent=2))
        return

    if not args.sentences_jsonl:
        raise SystemExit("请提供 --sentences-jsonl，或使用 --only-supergroup-plots")

    ufp: Optional[bool] = None
    if args.use_fp16 == "true":
        ufp = True
    elif args.use_fp16 == "false":
        ufp = False

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
        query_instruction=(args.bge_query_instruction or None),
        use_fp16=ufp,
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
