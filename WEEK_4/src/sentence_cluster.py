from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
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
    with path.open(encoding="utf-8") as f:
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


def _encode_texts(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: Optional[str],
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "sentence-cluster requires sentence-transformers. "
            'Install: pip install -r requirements-optional.txt (or pip install sentence-transformers umap-learn hdbscan numpy)'
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


def _l2_normalize_rows(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return emb / n


def _deepseek_embeddings_batch(
    texts: List[str],
    *,
    api_key: str,
    model: str,
    url: str,
    timeout_sec: int,
) -> np.ndarray:
    payload = json.dumps({"model": model, "input": texts}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise SystemExit(
            f"DeepSeek embeddings HTTP {e.code}: {err[:800]}\n"
            "确认模型名为官方「嵌入」模型（不是 deepseek-chat）；见 https://api-docs.deepseek.com/"
        ) from e
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise SystemExit(f"DeepSeek embeddings 请求失败: {e}") from e

    data = body.get("data")
    if not isinstance(data, list) or not data:
        raise SystemExit(f"DeepSeek embeddings 响应异常: {json.dumps(body, ensure_ascii=False)[:500]}")

    data_sorted = sorted(data, key=lambda x: int(x.get("index", 0)))
    vectors: List[List[float]] = []
    for item in data_sorted:
        vec = item.get("embedding")
        if not isinstance(vec, list):
            raise SystemExit("DeepSeek embeddings 条目不包含 embedding 数组")
        vectors.append([float(x) for x in vec])
    return np.asarray(vectors, dtype=np.float32)


def _encode_texts_deepseek_api(
    texts: List[str],
    *,
    api_key: str,
    model: str,
    url: str,
    batch_size: int,
    timeout_sec: int,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    bs = max(1, batch_size)
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        chunks.append(
            _deepseek_embeddings_batch(
                batch,
                api_key=api_key,
                model=model,
                url=url,
                timeout_sec=timeout_sec,
            )
        )
        print(f"  embedded {min(i + bs, len(texts))}/{len(texts)}", flush=True)
    emb = np.vstack(chunks)
    return _l2_normalize_rows(emb)


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
    cmap = plt.cm.get_cmap("tab20")
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
    max_samples_per_cluster: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], np.ndarray, np.ndarray]:
    texts = [str(r.get("text", "")).strip() or " " for r in rows]
    if embed_backend == "local":
        emb = _encode_texts(texts, model_name, batch_size, device)
    elif embed_backend == "deepseek-api":
        emb = _encode_texts_deepseek_api(
            texts,
            api_key=deepseek_api_key,
            model=deepseek_embedding_model,
            url=deepseek_embeddings_url,
            batch_size=batch_size,
            timeout_sec=timeout_sec,
        )
    else:
        raise ValueError(f"Unknown embed_backend: {embed_backend}")
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
    summary["model_name"] = (
        deepseek_embedding_model if embed_backend == "deepseek-api" else model_name
    )
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
            "说明：默认 local 从 Hugging Face 下载句向量模型；若 huggingface.co 超时，可设环境变量 HF_ENDPOINT=https://hf-mirror.com 后重试，"
            "或改用 --embed-backend deepseek-api（使用 DeepSeek「嵌入」接口，不是 deepseek-chat）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="sentences.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--embed-backend",
        choices=("local", "deepseek-api"),
        default="local",
        help="local: sentence-transformers + HuggingFace；deepseek-api: 云端嵌入（需密钥，模型见 --deepseek-embedding-model）",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="仅 embed-backend=local：SentenceTransformer 模型 id",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="仅 local：e.g. cuda, cuda:0, cpu")
    parser.add_argument(
        "--deepseek-api-key",
        default=None,
        help="embed-backend=deepseek-api：API Key，默认读环境变量 DEEPSEEK_API_KEY",
    )
    parser.add_argument(
        "--deepseek-embedding-model",
        default="deepseek-embedding",
        help="embed-backend=deepseek-api：嵌入模型名（非 deepseek-chat；以官方文档为准）",
    )
    parser.add_argument(
        "--deepseek-embeddings-url",
        default="https://api.deepseek.com/v1/embeddings",
        help="OpenAI 兼容的 embeddings 端点 URL",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="deepseek-api 单批请求超时（秒）",
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

    ds_key = (args.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if args.embed_backend == "deepseek-api" and not ds_key:
        raise SystemExit(
            "embed-backend=deepseek-api 需要 --deepseek-api-key 或环境变量 DEEPSEEK_API_KEY"
        )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_rows, summary, xy, labels = run_clustering(
        rows,
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
