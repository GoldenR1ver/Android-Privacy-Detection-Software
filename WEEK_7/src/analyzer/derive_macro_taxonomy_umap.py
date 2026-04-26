"""
从某次 run 下全部 sentences_cluster_full.jsonl 聚合样本，
为 22 个 taxonomy_22_id 各构造一个「类原型」向量（该类样本的 UMAP 均值 + 22 维相似度均值），
在 3..7 大类上用 Ward 层次聚类，以轮廓系数选最优 K，写出 macro_taxonomy_umap.json。

不依赖 sklearn；使用 scipy.cluster.hierarchy + 手写轮廓系数（22 点很小）。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from sentence_cluster import load_jsonl

# 与 macro_id 1..6（本工程 $K=6$ 最优划分）一一对应的汇报命名
MACRO_DISPLAY_NAMES_K6: Dict[int, str] = {
    1: "信息处理基础与台账",
    2: "用户控制与算法交互核心",
    3: "未成年人保护专项",
    4: "高风险场景及单独同意",
    5: "主体告知与复合程序要求",
    6: "撤回同意便捷性专项",
}


def _silhouette_euclidean(X: np.ndarray, labels: np.ndarray) -> float:
    """平均轮廓系数；labels 为 1..K。"""
    n = X.shape[0]
    if n < 2:
        return -1.0
    lab = labels.astype(np.int32)
    uniq = np.unique(lab)
    if len(uniq) < 2:
        return -1.0
    dist = squareform(pdist(X, metric="euclidean"))
    scores: List[float] = []
    for i in range(n):
        own = lab[i]
        mask_own = lab == own
        if mask_own.sum() <= 1:
            a = 0.0
        else:
            a = float(dist[i, mask_own].sum() / (mask_own.sum() - 1))
        others = [u for u in uniq if u != own]
        b_min = float("inf")
        for u in others:
            m = lab == u
            if not np.any(m):
                continue
            b_min = min(b_min, float(dist[i, m].mean()))
        if b_min == float("inf"):
            b_min = 0.0
        denom = max(a, b_min)
        scores.append((b_min - a) / denom if denom > 1e-12 else 0.0)
    return float(np.mean(scores))


def _zscore_rows(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X - mu) / sd


def collect_prototypes(pipeline_root: Path) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    返回 X (22, D) 行对应 item_id 1..22；D=2+22=24。
    若某类无样本，用全局均值填充该行。
    """
    paths = sorted(pipeline_root.glob("**/cluster_analysis/sentences_cluster_full.jsonl"))
    if not paths:
        raise SystemExit(f"No sentences_cluster_full.jsonl under {pipeline_root}")

    umap_lists: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    score_lists: Dict[int, List[np.ndarray]] = defaultdict(list)

    for p in paths:
        for r in load_jsonl(p):
            tid = int(r.get("taxonomy_22_id", 0) or 0)
            if tid < 1 or tid > 22:
                continue
            umap_lists[tid].append((float(r.get("umap_x", 0.0)), float(r.get("umap_y", 0.0))))
            sc = r.get("taxonomy_22_scores")
            if isinstance(sc, list) and len(sc) == 22:
                score_lists[tid].append(np.asarray(sc, dtype=np.float64))

    X_rows: List[np.ndarray] = []
    counts: Dict[int, int] = {}
    all_umap: List[Tuple[float, float]] = []
    all_scores: List[np.ndarray] = []

    for tid in range(1, 23):
        ul = umap_lists[tid]
        sl = score_lists[tid]
        counts[tid] = len(ul)
        if ul:
            um = np.asarray(ul, dtype=np.float64).mean(axis=0)
            for u in ul:
                all_umap.append((float(u[0]), float(u[1])))
        else:
            um = np.zeros(2, dtype=np.float64)

        if sl:
            sm = np.stack(sl, axis=0).mean(axis=0)
            for s in sl:
                all_scores.append(s)
        else:
            sm = np.zeros(22, dtype=np.float64)

        X_rows.append(np.concatenate([um, sm], axis=0))

    X = np.stack(X_rows, axis=0)

    # 无样本类：用全局均值替换（避免全零扭曲距离）
    g_umap = np.asarray(all_umap, dtype=np.float64).mean(axis=0) if all_umap else np.zeros(2)
    g_score = np.stack(all_scores, axis=0).mean(axis=0) if all_scores else np.zeros(22)
    for i, tid in enumerate(range(1, 23)):
        if counts[tid] == 0:
            X[i] = np.concatenate([g_umap, g_score])

    Xn = _zscore_rows(X)
    return Xn, counts


def choose_partition(Xn: np.ndarray, k_min: int = 3, k_max: int = 7) -> Tuple[int, Dict[int, float], np.ndarray]:
    Z = linkage(Xn, method="ward")
    best_k = k_min
    best_s = -2.0
    by_k: Dict[int, float] = {}
    best_lab = np.ones(22, dtype=np.int32)
    for k in range(k_min, k_max + 1):
        lab = fcluster(Z, k, criterion="maxclust").astype(np.int32)
        s = _silhouette_euclidean(Xn, lab)
        by_k[k] = s
        if s > best_s:
            best_s = s
            best_k = k
            best_lab = lab
    return best_k, by_k, best_lab


def build_groups(labels: np.ndarray, best_k: int) -> List[Dict[str, Any]]:
    """labels[i] 对应 item_id = i+1。$K=6$ 时写入固定汇报命名。"""
    buckets: Dict[int, List[int]] = defaultdict(list)
    for i in range(22):
        buckets[int(labels[i])].append(i + 1)
    out: List[Dict[str, Any]] = []
    for gid in sorted(buckets.keys()):
        ids = sorted(buckets[gid])
        if best_k == 6 and gid in MACRO_DISPLAY_NAMES_K6:
            label = MACRO_DISPLAY_NAMES_K6[gid]
        else:
            label = f"数据驱动-大类{gid}"
        out.append(
            {
                "macro_id": gid,
                "item_ids": ids,
                "label": label,
            }
        )
    return out


def derive_and_write_macro_json(pipeline_root: Path) -> Path:
    root = pipeline_root.resolve()
    Xn, counts = collect_prototypes(root)
    best_k, by_k, lab = choose_partition(Xn, 3, 7)
    groups = build_groups(lab, best_k)
    item_to_macro = {str(i + 1): int(lab[i]) for i in range(22)}
    out = {
        "pipeline_root": str(root),
        "description": "每类原型 = concat(mean UMAP, mean 22-dim cosine profile)，列 z-score 后 Ward 聚类；K=3..7 最大化轮廓系数。K=6 时 groups[].label 为固定汇报命名。",
        "macro_display_names_k6": {str(k): v for k, v in MACRO_DISPLAY_NAMES_K6.items()},
        "per_class_sentence_counts": {str(k): counts[k] for k in range(1, 23)},
        "silhouette_by_k": {str(k): round(v, 4) for k, v in sorted(by_k.items())},
        "best_k": best_k,
        "best_mean_silhouette": round(float(by_k[best_k]), 4),
        "groups": groups,
        "item_to_macro": item_to_macro,
    }
    out_path = root / "macro_taxonomy_umap.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive macro taxonomy from clustered sentences")
    ap.add_argument("--pipeline-root", type=Path, required=True)
    args = ap.parse_args()
    out_path = derive_and_write_macro_json(args.pipeline_root)
    meta = json.loads(out_path.read_text(encoding="utf-8"))
    print(json.dumps({"wrote": str(out_path), "best_k": meta["best_k"], "groups": meta["groups"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
