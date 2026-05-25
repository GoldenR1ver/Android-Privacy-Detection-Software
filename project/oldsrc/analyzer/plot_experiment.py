from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib_zh import configure_matplotlib_chinese_font

configure_matplotlib_chinese_font()


def load_stats(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_keyword_and_pii(stats: Dict[str, Any], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    kh = stats.get("keyword_hint", {})
    ax = axes[0]
    labels_k = ["keyword_hint=True", "keyword_hint=False"]
    vals_k = [kh.get("true", 0), kh.get("false", 0)]
    ax.bar(labels_k, vals_k, color=["#2ecc71", "#bdc3c7"])
    ax.set_title("Keyword hint (rule-based)")
    ax.set_ylabel("Sentence count")
    for i, v in enumerate(vals_k):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)

    pr = stats.get("pii_related", {})
    ax2 = axes[1]
    labels_p = ["True", "False", "Unlabeled"]
    vals_p = [pr.get("true", 0), pr.get("false", 0), pr.get("null", 0)]
    ax2.bar(labels_p, vals_p, color=["#3498db", "#e74c3c", "#95a5a6"])
    ax2.set_title("PII-related (model / null if not classified)")
    ax2.set_ylabel("Sentence count")
    for i, v in enumerate(vals_p):
        ax2.text(i, v, str(v), ha="center", va="bottom", fontsize=10)

    fig.suptitle(f"doc_id={stats.get('doc_id', '?')}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "stats_counts.png", dpi=160)
    plt.close(fig)


def plot_sentence_lengths(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    lens = [len(r.get("text", "")) for r in rows]
    if not lens:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lens, bins=min(40, max(10, len(lens) // 20)), color="#9b59b6", edgecolor="white")
    ax.set_title("Sentence length (characters)")
    ax.set_xlabel("Chars per sentence")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "sentence_length_hist.png", dpi=160)
    plt.close(fig)


def plot_audit_label_counts(audit_csv: Path, out_dir: Path) -> Dict[str, Any]:
    cols = ["right_claim", "method_claim", "app_test_candidate"]
    counts = {c: 0 for c in cols}
    n = 0
    with audit_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            for c in cols:
                try:
                    if int(str(row.get(c, "0")).strip() or 0) == 1:
                        counts[c] += 1
                except ValueError:
                    pass
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(counts.keys()), list(counts.values()), color=["#e67e22", "#1abc9c", "#34495e"])
    ax.set_title(f"Rows with Lab3 label=1 (n={n} rows)")
    ax.set_ylabel("Count")
    for i, c in enumerate(cols):
        ax.text(i, counts[c], str(counts[c]), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "audit_positive_counts.png", dpi=160)
    plt.close(fig)

    return {"audit_rows": n, **{f"audit_{k}_count": v for k, v in counts.items()}}


def write_summary_table(
    stats: Dict[str, Any],
    out_dir: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    rows_out: List[Dict[str, Any]] = [
        {
            "metric": "total_sentences",
            "value": stats.get("total_sentences"),
        },
        {
            "metric": "keyword_hint_ratio_of_total",
            "value": (stats.get("keyword_hint") or {}).get("ratio_of_total"),
        },
        {
            "metric": "pii_related_ratio_of_total",
            "value": (stats.get("pii_related") or {}).get("ratio_of_total"),
        },
        {
            "metric": "pii_related_ratio_of_labeled",
            "value": (stats.get("pii_related") or {}).get("ratio_of_labeled"),
        },
    ]
    if extra:
        for k, v in extra.items():
            rows_out.append({"metric": k, "value": v})
    p = out_dir / "summary_metrics.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plots + CSV summary from WEEK_4 experiment outputs (and optional WEEK_3 audit CSV).",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Folder containing stats.json (and optionally sentences.jsonl)",
    )
    parser.add_argument(
        "--audit-processed-csv",
        type=Path,
        default=None,
        help="Optional: run_audit postprocess output with Lab3 right/method/test-candidate columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figures + summary_metrics.csv (default: <experiment-dir>/figures)",
    )
    args = parser.parse_args()

    exp = args.experiment_dir
    out = args.output_dir or (exp / "figures")
    out.mkdir(parents=True, exist_ok=True)

    stats = load_stats(exp / "stats.json")
    plot_keyword_and_pii(stats, out)

    jsonl_path = exp / "sentences.jsonl"
    extra: Dict[str, Any] = {}
    if jsonl_path.is_file():
        jrows = load_jsonl(jsonl_path)
        plot_sentence_lengths(jrows, out)
        extra["jsonl_sentences_loaded"] = len(jrows)

    if args.audit_processed_csv and args.audit_processed_csv.is_file():
        audit_extra = plot_audit_label_counts(args.audit_processed_csv, out)
        if audit_extra:
            extra.update(audit_extra)

    write_summary_table(stats, out, extra if extra else None)
    wrote = ["stats_counts.png", "summary_metrics.csv"]
    if jsonl_path.is_file():
        wrote.append("sentence_length_hist.png")
    if args.audit_processed_csv and args.audit_processed_csv.is_file():
        wrote.append("audit_positive_counts.png")
    print(json.dumps({"figures_dir": str(out), "wrote": wrote}, ensure_ascii=False))


if __name__ == "__main__":
    main()
