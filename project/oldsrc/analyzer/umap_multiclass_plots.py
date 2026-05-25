"""
从 cluster_analysis 产出的 sentences_cluster_full.jsonl 绘制多分类 UMAP（不重算嵌入）。

- taxonomy_22_group：22 项在课表中的大类（核心内容 / 特定情形 / 格式表述），3 类着色。
- pii_related × keyword_hint：模型二分类与规则弱信号的 4 象限联合标签。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib_zh import configure_matplotlib_chinese_font

configure_matplotlib_chinese_font()


def _rows_xy(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    xy_list: List[Tuple[float, float]] = []
    kept: List[Dict[str, Any]] = []
    for r in rows:
        try:
            x = float(r.get("umap_x"))
            y = float(r.get("umap_y"))
        except (TypeError, ValueError):
            continue
        xy_list.append((x, y))
        kept.append(r)
    if not xy_list:
        raise ValueError("无有效 umap_x / umap_y")
    return np.asarray(xy_list, dtype=np.float64), kept


_TAX_GROUP_ORDER = ("核心内容", "特定情形", "格式表述", "（缺）")


def plot_umap_taxonomy_group(rows: List[Dict[str, Any]], out_path: Path, title: str | None = None) -> None:
    xy, rr = _rows_xy(rows)
    groups = [str(r.get("taxonomy_22_group") or "（缺）") for r in rr]
    present = set(groups)
    uniq = [g for g in _TAX_GROUP_ORDER if g in present] + sorted(present - set(_TAX_GROUP_ORDER))
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for i, g in enumerate(uniq):
        m = np.array([groups[j] == g for j in range(len(groups))], dtype=bool)
        if not np.any(m):
            continue
        c = cmap(i % 10 / 9.0)
        ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.75, color=c, label=g)
    ax.set_title(title or "UMAP 着色：22 项课表大类（最近邻项所属 group）")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _joint_label(pii: Any, kh: Any) -> str:
    if pii is True and kh is True:
        return "模型相关 + 规则命中"
    if pii is True and kh is False:
        return "模型相关 + 规则未命中"
    if pii is False and kh is True:
        return "模型不相关 + 规则命中"
    if pii is False and kh is False:
        return "模型不相关 + 规则未命中"
    return "模型未标注 / 其它"


def plot_umap_pii_keyword_joint(rows: List[Dict[str, Any]], out_path: Path, title: str | None = None) -> None:
    xy, rr = _rows_xy(rows)
    labels = [_joint_label(r.get("pii_related"), r.get("keyword_hint")) for r in rr]
    order = [
        "模型相关 + 规则命中",
        "模型相关 + 规则未命中",
        "模型不相关 + 规则命中",
        "模型不相关 + 规则未命中",
        "模型未标注 / 其它",
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for lab, col in zip(order, colors):
        m = np.array([labels[j] == lab for j in range(len(labels))], dtype=bool)
        if not np.any(m):
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.75, color=col, label=lab)
    ax.set_title(
        title
        or "UMAP 着色：句级「个信相关」× 规则弱信号（四象限；任务本身仍为二分类）"
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="从 sentences_cluster_full.jsonl 绘制多分类 UMAP（不重算嵌入）")
    p.add_argument(
        "--cluster-jsonl",
        type=Path,
        required=True,
        help="含 umap_x/umap_y、taxonomy_22_group、pii_related、keyword_hint 的 jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认同 jsonl 所在目录",
    )
    args = p.parse_args()
    out_dir = args.output_dir or args.cluster_jsonl.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.cluster_jsonl)
    p1 = out_dir / "umap_taxonomy_group.png"
    p2 = out_dir / "umap_pii_keyword_joint.png"
    plot_umap_taxonomy_group(rows, p1)
    plot_umap_pii_keyword_joint(rows, p2)
    print(json.dumps({"wrote": [str(p1), str(p2)], "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
