"""
从 week5_aggregate_report.json 绘制「全量句级审计」图表（数据与各应用 audit_processed.csv 行数一致）。

用法:
  python plot_audit_aggregate_figures.py --aggregate-json output/run_20260413_150151/week5_aggregate_report.json
  python plot_audit_aggregate_figures.py --aggregate-json ... --out-dir ../ppt/figures
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


def _load_aggregate(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_series(data: Dict[str, Any]) -> Tuple[List[str], List[int], List[int], List[int], List[int]]:
    apps: List[str] = []
    totals: List[int] = []
    right: List[int] = []
    method: List[int] = []
    candidate: List[int] = []
    for p in data.get("per_app") or []:
        apps.append(str(p.get("doc_id", "?")))
        a = p.get("audit_summary") or {}
        totals.append(int(a.get("rows", 0)))
        right.append(int(a.get("right_claim_1", 0)))
        method.append(int(a.get("method_claim_1", 0)))
        candidate.append(int(a.get("app_test_candidate_1", 0)))
    return apps, totals, right, method, candidate


def plot_grouped_counts_by_app(
    apps: List[str],
    right: List[int],
    method: List[int],
    candidate: List[int],
    out_path: Path,
    run_label: str,
) -> None:
    x = np.arange(len(apps))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - w, right, width=w, label="right_claim (=1)", color="#e67e22")
    ax.bar(x, method, width=w, label="method_claim (=1)", color="#1abc9c")
    ax.bar(x + w, candidate, width=w, label="app_test_candidate (=1)", color="#34495e")
    ax.set_xticks(x)
    ax.set_xticklabels(apps, rotation=18, ha="right")
    ax.set_ylabel("句数（全量 audit 行）")
    ax.set_title(f"各应用 Lab3 行权三标签计数（全量）— {run_label}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_rates_by_app(
    apps: List[str],
    totals: List[int],
    right: List[int],
    method: List[int],
    candidate: List[int],
    out_path: Path,
    run_label: str,
) -> None:
    """各应用内 label=1 占该应用审计行数的比例（%）。"""
    def rate(num: int, den: int) -> float:
        return 100.0 * num / den if den else 0.0

    r_right = [rate(a, t) for a, t in zip(right, totals)]
    r_method = [rate(a, t) for a, t in zip(method, totals)]
    r_candidate = [rate(a, t) for a, t in zip(candidate, totals)]

    x = np.arange(len(apps))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - w, r_right, width=w, label="right_claim %", color="#e67e22")
    ax.bar(x, r_method, width=w, label="method_claim %", color="#1abc9c")
    ax.bar(x + w, r_candidate, width=w, label="app_test_candidate %", color="#34495e")
    ax.set_xticks(x)
    ax.set_xticklabels(apps, rotation=18, ha="right")
    ax.set_ylabel("占该应用审计行比例（%）")
    ax.set_title(f"各应用 Lab3 行权三标签正例率（全量）— {run_label}")
    ax.legend(loc="upper right", fontsize=9)
    combined = r_right + r_method + r_candidate
    ymax = max(combined) if combined else 0.0
    ax.set_ylim(0, ymax * 1.15 + 0.5)
    ax.grid(axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_global_totals_stacked(
    right: List[int],
    method: List[int],
    candidate: List[int],
    out_path: Path,
    run_label: str,
) -> None:
    tot_right = sum(right)
    tot_method = sum(method)
    tot_candidate = sum(candidate)
    labels = ["right_claim", "method_claim", "app_test_candidate"]
    vals = [tot_right, tot_method, tot_candidate]
    colors = ["#e67e22", "#1abc9c", "#34495e"]
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    left = 0
    for lab, v, c in zip(labels, vals, colors):
        ax.barh(0, v, left=left, height=0.55, label=f"{lab} ({v})", color=c)
        ax.text(left + v / 2, 0, str(v), ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        left += v
    ax.set_yticks([])
    ax.set_xlabel("句数（全库合计，与各行 audit_processed 一致）")
    ax.set_title(f"全库 Lab3 行权三标签正例合计 — {run_label}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=9)
    ax.set_xlim(0, left * 1.02)
    ax.grid(axis="x", alpha=0.28)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    default_out = here.parent / "ppt" / "figures"

    ap = argparse.ArgumentParser(description="Plot full-audit charts from week5_aggregate_report.json")
    ap.add_argument("--aggregate-json", type=Path, required=True, help="Path to week5_aggregate_report.json")
    ap.add_argument("--out-dir", type=Path, default=default_out, help="Directory for PNG outputs")
    ap.add_argument(
        "--run-label",
        type=str,
        default="",
        help="Short label for figure titles (default: parent folder name of aggregate json)",
    )
    args = ap.parse_args()

    agg_path: Path = args.aggregate_json
    if not agg_path.is_file():
        raise SystemExit(f"Missing file: {agg_path}")

    run_label = args.run_label.strip() or agg_path.parent.name
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_aggregate(agg_path)
    apps, totals, right, method, candidate = _extract_series(data)

    p1 = out_dir / "fig_audit_full_grouped_by_app.png"
    p2 = out_dir / "fig_audit_full_rates_by_app.png"
    p3 = out_dir / "fig_audit_full_totals_stacked.png"

    plot_grouped_counts_by_app(apps, right, method, candidate, p1, run_label)
    plot_rates_by_app(apps, totals, right, method, candidate, p2, run_label)
    plot_global_totals_stacked(right, method, candidate, p3, run_label)

    print(
        json.dumps(
            {
                "wrote": [str(p1), str(p2), str(p3)],
                "apps": apps,
                "totals": {
                    "right_claim": sum(right),
                    "method_claim": sum(method),
                    "app_test_candidate": sum(candidate),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
