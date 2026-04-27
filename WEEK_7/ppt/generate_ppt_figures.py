"""Generate PNG figures for WEEK_4/ppt/slide.tex from pipeline output directories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
from matplotlib_zh import configure_matplotlib_chinese_font  # noqa: E402
from plot_experiment import (  # noqa: E402
    load_jsonl,
    load_stats,
    plot_audit_label_counts,
    plot_keyword_and_pii,
    plot_sentence_lengths,
)

configure_matplotlib_chinese_font()

# 与汇报 PPT 中五款应用顺序一致：小红书、LOFTER、Blued、世纪佳缘、Soul
DOC_ORDER: List[Tuple[str, str]] = [
    ("小红书", "小红书"),
    ("LOFTER", "LOFTER"),
    ("Blued极速版", "Blued"),
    ("世纪佳缘", "世纪佳缘"),
    ("Soul", "Soul"),
]


def short_label(doc_id: str) -> str:
    for folder, short in DOC_ORDER:
        if folder == doc_id:
            return short
    return doc_id


def collect_stats(pipeline_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for folder, _short in DOC_ORDER:
        p = pipeline_root / folder / "stats.json"
        if not p.is_file():
            continue
        s = load_stats(p)
        s["_folder"] = folder
        rows.append(s)
    return rows


def plot_ratios_by_app(rows: List[Dict[str, Any]], out: Path) -> None:
    labels = [short_label(str(r.get("doc_id", ""))) for r in rows]
    kh = [(r.get("keyword_hint") or {}).get("ratio_of_total") or 0.0 for r in rows]
    # 与汇报 PPT 一致：蓝色柱 = 模型判「相关」句数 / 总句数（不区分 null）
    pr = []
    for r in rows:
        pii = r.get("pii_related") or {}
        total = int(r.get("total_sentences") or 0)
        p_true = int(pii.get("true") or 0)
        pr.append((p_true / total) if total else 0.0)
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - w / 2, [100 * v for v in kh], w, label="关键词弱信号占比", color="#2ecc71")
    ax.bar(x + w / 2, [100 * v for v in pr], w, label="模型「个信相关」占比", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("占比（%）")
    ax.set_title("五款应用：规则弱信号 vs DeepSeek 句子级分类")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_totals_by_app(rows: List[Dict[str, Any]], out: Path) -> None:
    labels = [short_label(str(r.get("doc_id", ""))) for r in rows]
    totals = [int(r.get("total_sentences") or 0) for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(labels, totals, color="#9b59b6", edgecolor="white")
    ax.set_ylabel("句子数")
    ax.set_title("各应用隐私政策切句规模")
    for i, v in enumerate(totals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_pii_true_counts(rows: List[Dict[str, Any]], out: Path) -> None:
    labels = [short_label(str(r.get("doc_id", ""))) for r in rows]
    true_ct = [int((r.get("pii_related") or {}).get("true") or 0) for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(labels, true_ct, color="#3498db", edgecolor="white")
    ax.set_ylabel("句数")
    ax.set_title("模型判为「个人信息处理相关」的句子数量")
    for i, v in enumerate(true_ct):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_audit_by_app(audit_root: Path, out: Path) -> None:
    labels: List[str] = []
    inc: List[int] = []
    incomp: List[int] = []
    incons: List[int] = []
    for folder, short in DOC_ORDER:
        csv_path = audit_root / folder / "audit_processed.csv"
        if not csv_path.is_file():
            continue
        c_inc = c_incomp = c_incons = 0
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if int(str(row.get("incorrect", "0")).strip() or 0) == 1:
                        c_inc += 1
                except ValueError:
                    pass
                try:
                    if int(str(row.get("incomplete", "0")).strip() or 0) == 1:
                        c_incomp += 1
                except ValueError:
                    pass
                try:
                    if int(str(row.get("inconsistent", "0")).strip() or 0) == 1:
                        c_incons += 1
                except ValueError:
                    pass
        labels.append(short)
        inc.append(c_inc)
        incomp.append(c_incomp)
        incons.append(c_incons)
    if not labels:
        return
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, inc, w, label="incorrect", color="#e67e22")
    ax.bar(x, incomp, w, label="incomplete", color="#1abc9c")
    ax.bar(x + w, incons, w, label="inconsistent", color="#34495e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("预测为 1 的行数")
    ax.set_title("句子级 audit 抽样：三标签正例计数（每应用前 N 行）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "out"
        / "pipeline_20260406_201540",
        help="Full classify output (stats.json per app folder)",
    )
    parser.add_argument(
        "--audit-pipeline-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "out"
        / "pipeline_20260406_194301",
        help="Optional audit run with audit_processed.csv per app",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_stats(args.pipeline_dir)
    if not rows:
        print(json.dumps({"error": "no stats.json found", "pipeline": str(args.pipeline_dir)}, ensure_ascii=False))
        sys.exit(1)

    plot_ratios_by_app(rows, out_dir / "fig_ratios_by_app.png")
    plot_totals_by_app(rows, out_dir / "fig_totals_by_app.png")
    plot_pii_true_counts(rows, out_dir / "fig_pii_true_by_app.png")

    soul_dir = args.pipeline_dir / "Soul"
    if (soul_dir / "stats.json").is_file():
        s = load_stats(soul_dir / "stats.json")
        soul_fig = out_dir / "soul_detail"
        soul_fig.mkdir(parents=True, exist_ok=True)
        plot_keyword_and_pii(s, soul_fig)
        if (soul_dir / "sentences.jsonl").is_file():
            jrows = load_jsonl(soul_dir / "sentences.jsonl")
            plot_sentence_lengths(jrows, soul_fig)

    if args.audit_pipeline_dir.is_dir():
        plot_audit_by_app(args.audit_pipeline_dir, out_dir / "fig_audit_by_app.png")
        for folder, _ in DOC_ORDER:
            exp = args.audit_pipeline_dir / folder
            ap = exp / "audit_processed.csv"
            if ap.is_file() and (exp / "stats.json").is_file():
                sub = out_dir / "audit" / folder
                sub.mkdir(parents=True, exist_ok=True)
                plot_audit_label_counts(ap, sub)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "wrote": sorted(p.name for p in out_dir.rglob("*.png")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
