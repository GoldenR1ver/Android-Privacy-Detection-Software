from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib_zh import configure_matplotlib_chinese_font

configure_matplotlib_chinese_font()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            out = {k: v for k, v in row.items() if not str(k).startswith("_")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def _l2_normalize_rows(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return emb / n


def default_bge_query_instruction(model_name: str) -> str:
    """BGE 官方表格中的 query_instruction（用于检索；对称句向量场景下仍作前缀，与手册一致）。"""
    m = model_name.lower()
    if "bge-m3" in m:
        return ""
    if "/bge-" in m or "bge-" in m:
        if "-zh" in m or "zh-v" in m or m.endswith("zh") or "chinese" in m:
            return "为这个句子生成表示以用于检索相关文章："
        if "-en" in m or "en-v" in m or m.endswith("en"):
            return "Represent this sentence for searching relevant passages: "
    return "为这个句子生成表示以用于检索相关文章："


def _devices_for_flag(device: Optional[str]) -> Optional[List[str]]:
    if not device:
        return None
    d = str(device).strip()
    if not d:
        return None
    return [d]


def _encode_texts_sentence_transformers(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: Optional[str],
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "embed-backend=sentence_transformers 需要 sentence-transformers。"
            "pip install -r requirements-optional.txt"
        ) from e
    kwargs: Dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    model = SentenceTransformer(model_name, **kwargs)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def _encode_texts_flag_embedding(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: Optional[str],
    *,
    query_instruction: Optional[str],
    use_fp16: Optional[bool],
) -> np.ndarray:
    import torch

    dev_str = (device or "").strip().lower()
    on_cpu = dev_str == "cpu" or (
        not dev_str.startswith("cuda") and not torch.cuda.is_available()
    )
    if use_fp16 is None:
        use_fp16_flag = not on_cpu
    else:
        use_fp16_flag = bool(use_fp16) and not on_cpu

    devices_kw = _devices_for_flag(device)
    mlower = model_name.lower()
    is_m3 = "bge-m3" in mlower or "bge_m3" in mlower

    if is_m3:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise SystemExit(
                "BGE-M3 需完整 FlagEmbedding（常含 finetune 依赖）。请 pip install -U \"FlagEmbedding[finetune]\"，"
                "或改用 BAAI/bge-small-zh-v1.5 等非 M3 模型。"
            ) from e
        model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16_flag,
            devices=devices_kw,
        )
        out = model.encode(
            texts,
            batch_size=max(1, batch_size),
            max_length=min(8192, 2048),
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = out.get("dense_vecs")
        if dense is None:
            raise SystemExit("BGE-M3 encode 未返回 dense_vecs")
        emb = np.asarray(dense, dtype=np.float32)
    else:
        try:
            from FlagEmbedding import FlagModel
        except ImportError as e:
            raise SystemExit(
                "embed-backend=flag 需要 FlagEmbedding（pip install -U FlagEmbedding）。"
                "若导入失败，请升级 accelerate / peft / transformers，"
                "或改用 --embed-backend sentence_transformers。"
            ) from e
        qinstr = (query_instruction or "").strip() or default_bge_query_instruction(model_name)
        model = FlagModel(
            model_name,
            query_instruction_for_retrieval=qinstr,
            use_fp16=use_fp16_flag,
            devices=devices_kw,
            normalize_embeddings=True,
        )
        emb = model.encode(
            texts,
            batch_size=max(1, batch_size),
            convert_to_numpy=True,
        )
        emb = np.asarray(emb, dtype=np.float32)

    return _l2_normalize_rows(emb)


def _encode_texts(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: Optional[str],
    *,
    embed_backend: str = "flag",
    query_instruction: Optional[str] = None,
    use_fp16: Optional[bool] = None,
) -> np.ndarray:
    eb = (embed_backend or "flag").strip().lower()
    if eb in ("sentence_transformers", "st", "sentence-transformers"):
        return _encode_texts_sentence_transformers(texts, model_name, batch_size, device)
    if eb in ("flag", "flagembedding", "bge"):
        return _encode_texts_flag_embedding(
            texts,
            model_name,
            batch_size,
            device,
            query_instruction=query_instruction,
            use_fp16=use_fp16,
        )
    raise ValueError(f"Unknown embed_backend: {embed_backend!r} (use flag or sentence_transformers)")


def _umap_reduce(
    emb: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    try:
        import umap
    except ImportError as e:
        raise SystemExit(
            "sentence-cluster requires umap-learn. "
            "Install: pip install -r requirements-optional.txt"
        ) from e
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, max(2, len(emb) - 1)),
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(emb), dtype=np.float32)


def _hdbscan_labels(
    xy: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> np.ndarray:
    try:
        import hdbscan
    except ImportError as e:
        raise SystemExit(
            "sentence-cluster requires hdbscan. "
            "Install: pip install -r requirements-optional.txt"
        ) from e
    n = len(xy)
    mc = max(2, min(min_cluster_size, n))
    ms = max(1, min(min_samples, n - 1)) if n > 1 else 1
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mc, min_samples=ms)
    return np.asarray(clusterer.fit_predict(xy), dtype=np.int32)


def _plot_scatter(
    xy: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    noise = labels < 0
    cmap = (
        matplotlib.colormaps["tab20"]
        if hasattr(matplotlib, "colormaps")
        else plt.cm.get_cmap("tab20")
    )
    try:
        tab = cmap.colors  # type: ignore[attr-defined]
    except AttributeError:
        tab = [cmap(i / 19.0) for i in range(20)]
    if np.any(~noise):
        uniq = sorted(int(x) for x in np.unique(labels[~noise]).tolist())
        for i, lid in enumerate(uniq):
            m = (~noise) & (labels == lid)
            c = tab[i % len(tab)]
            lbl = f"C{lid}" if i < 14 else None
            ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.75, color=c, label=lbl)
    if np.any(noise):
        ax.scatter(
            xy[noise, 0],
            xy[noise, 1],
            s=6,
            c="#7f8c8d",
            alpha=0.4,
            label="noise",
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=7, markerscale=1.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_cluster_summary(
    rows: List[Dict[str, Any]],
    labels: np.ndarray,
    texts: List[str],
    max_samples_per_cluster: int,
) -> Dict[str, Any]:
    uniq, counts = np.unique(labels, return_counts=True)
    clusters: Dict[str, Any] = {}
    for lid, cnt in zip(uniq.tolist(), counts.tolist()):
        key = str(int(lid))
        idxs = [i for i in range(len(labels)) if int(labels[i]) == int(lid)]
        sample_texts: List[str] = []
        step = max(1, len(idxs) // max_samples_per_cluster) if idxs else 1
        for j in idxs[::step][:max_samples_per_cluster]:
            t = texts[j]
            sample_texts.append(t[:200] + ("…" if len(t) > 200 else ""))
        clusters[key] = {"size": int(cnt), "sample_texts": sample_texts}

    noise_count = int(np.sum(labels < 0))
    hard_indices = [i for i in range(len(labels)) if labels[i] < 0]
    return {
        "num_clusters": int(np.sum(uniq >= 0)),
        "noise_count": noise_count,
        "noise_fraction": noise_count / len(labels) if len(labels) else 0.0,
        "hard_example_indices": hard_indices[:500],
        "clusters": clusters,
    }


def run_clustering(
    rows: List[Dict[str, Any]],
    *,
    embed_backend: str,
    model_name: str,
    batch_size: int,
    device: Optional[str],
    query_instruction: Optional[str] = None,
    use_fp16: Optional[bool] = None,
    umap_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    umap_random_state: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    max_samples_per_cluster: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], np.ndarray, np.ndarray]:
    texts = [str(r.get("text", "")).strip() or " " for r in rows]
    emb = _encode_texts(
        texts,
        model_name,
        batch_size,
        device,
        embed_backend=embed_backend,
        query_instruction=query_instruction,
        use_fp16=use_fp16,
    )
    xy = _umap_reduce(emb, umap_neighbors, umap_min_dist, umap_metric, umap_random_state)
    if len(xy) < 2:
        raise ValueError("Need at least 2 sentences for UMAP/HDBSCAN.")
    labels = _hdbscan_labels(xy, hdbscan_min_cluster_size, hdbscan_min_samples)

    out_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        nr = dict(r)
        nr["umap_x"] = float(xy[i, 0])
        nr["umap_y"] = float(xy[i, 1])
        nr["cluster_id"] = int(labels[i])
        nr["is_cluster_noise"] = bool(labels[i] < 0)
        out_rows.append(nr)

    summary = build_cluster_summary(out_rows, labels, texts, max_samples_per_cluster)
    summary["embed_backend"] = embed_backend
    summary["model_name"] = model_name
    summary["umap"] = {
        "n_neighbors": umap_neighbors,
        "min_dist": umap_min_dist,
        "metric": umap_metric,
    }
    summary["hdbscan"] = {
        "min_cluster_size": hdbscan_min_cluster_size,
        "min_samples": hdbscan_min_samples,
    }
    return out_rows, summary, xy, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sentence embeddings + UMAP + HDBSCAN on sentences.jsonl (topic / hard-example assist).",
        epilog=(
            "默认 embed-backend=flag 使用 FlagEmbedding.FlagModel（BGE），模型从 Hugging Face 拉取；"
            "超时请设 HF_ENDPOINT=https://hf-mirror.com 。"
            "可选 sentence_transformers 走原 SentenceTransformer 路径。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="sentences.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--embed-backend",
        choices=("flag", "sentence_transformers"),
        default="flag",
        help="flag: FlagEmbedding BGE；sentence_transformers: 原 minilm 等 ST 模型",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-zh-v1.5",
        help="HF 模型名或本地目录；flag 时推荐 BAAI/bge-*-zh-v1.5",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="e.g. cuda:0, cpu；传给 FlagModel/SentenceTransformer")
    parser.add_argument(
        "--bge-query-instruction",
        default="",
        help="覆盖默认检索前缀；留空则按模型名自动选（见 README_zh 模型表）",
    )
    parser.add_argument(
        "--use-fp16",
        choices=("auto", "true", "false"),
        default="auto",
        help="FlagEmbedding：GPU 上默认 fp16；CPU 强制 fp32",
    )
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.08)
    parser.add_argument("--umap-metric", default="cosine")
    parser.add_argument("--umap-random-state", type=int, default=42)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=8)
    parser.add_argument("--hdbscan-min-samples", type=int, default=3)
    parser.add_argument("--max-samples-per-cluster", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if len(rows) < 2:
        raise SystemExit("Need at least 2 rows in JSONL.")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ufp: Optional[bool] = None
    if args.use_fp16 == "true":
        ufp = True
    elif args.use_fp16 == "false":
        ufp = False

    out_rows, summary, xy, labels = run_clustering(
        rows,
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
        max_samples_per_cluster=args.max_samples_per_cluster,
    )

    write_jsonl(out_rows, out_dir / "sentences_clustered.jsonl")
    (out_dir / "cluster_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.no_plot:
        _plot_scatter(
            xy,
            labels,
            out_dir / "umap_hdbscan.png",
            title="UMAP + HDBSCAN（句向量聚类）",
        )

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "rows": len(out_rows),
                "num_clusters": summary["num_clusters"],
                "noise_count": summary["noise_count"],
                "wrote": ["sentences_clustered.jsonl", "cluster_summary.json"]
                + ([] if args.no_plot else ["umap_hdbscan.png"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
