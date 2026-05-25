"""Generate Lab4-1 only presentation: figures + PPTX.

Usage (from project/ppt):
  python generate_lab4_1_ppt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PPT = Path(__file__).resolve().parent
SRC = PPT.parent / "src"
FIG = PPT / "figures_lab4_1"
SUMMARY = SRC / "output" / "lab4_1" / "batch_lab4-1_exp_stage3_20260525_lab4_1_summary.json"

APPS = [
    "Keep",
    "小红书",
    "支付宝",
    "LOFTER",
    "智联招聘",
    "世纪佳缘",
    "好大夫医生版",
    "东方财富",
    "小荷健康",
]


def _configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(SRC))
    try:
        from matplotlib_zh import configure_matplotlib_chinese_font

        configure_matplotlib_chinese_font()
    except Exception:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
    return plt


def load_lab41_reports() -> list[dict]:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for rep in data.get("reports", []):
        sm = rep.get("summary") or {}
        sc = sm.get("status_counts") or {}
        rows.append(
            {
                "app": rep.get("claims", {}).get("app_name")
                or rep.get("evidence", {}).get("app_name")
                or "?",
                "declared": sm.get("declared_count", 0),
                "observed": sm.get("observed_count", 0),
                "consistent": sc.get("consistent", 0),
                "high_risk": sc.get("inconsistent_high_risk", 0),
                "not_obs": sc.get("declared_not_observed", 0),
            }
        )
    order = {a: i for i, a in enumerate(APPS)}
    rows.sort(key=lambda r: order.get(r["app"], 99))
    return rows


def plot_lab41_pipeline(plt, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 2.0))
    ax.axis("off")
    boxes = [
        "隐私政策\npolicy_parser",
        "静态 APK\nStage2",
        "动态 UI\nStage3",
        "证据合并\n+ KB",
        "compare_engine\nLab4-1",
        "batch 报告",
    ]
    colors = ["#d6eaf8", "#d5f5e3", "#fdebd0", "#e8daef", "#fadbd8", "#f9e79f"]
    for i, (text, color) in enumerate(zip(boxes, colors)):
        ax.add_patch(plt.Rectangle((i * 1.55, 0.2), 1.35, 0.9, fc=color, ec="#333", lw=1.2))
        ax.text(i * 1.55 + 0.675, 0.65, text, ha="center", va="center", fontsize=9)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(i * 1.55 + 1.35, 0.65),
                xytext=((i + 1) * 1.55, 0.65),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )
    ax.set_xlim(-0.2, len(boxes) * 1.55 + 0.3)
    ax.set_ylim(0, 1.4)
    fig.savefig(out / "fig_lab41_pipeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_verdict_totals(plt, out: Path) -> None:
    labels = ["consistent\n一致", "inconsistent_\nhigh_risk", "declared_not_\nobserved"]
    values = [90, 41, 6]
    colors = ["#27ae60", "#e74c3c", "#f39c12"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("Verdict 条数（9 App 合计）")
    ax.set_title("Lab4-1 知情权：全样本判定合计")
    fig.tight_layout()
    fig.savefig(out / "fig_lab41_verdict_totals.png", dpi=160)
    plt.close(fig)


def plot_lab41_verdict_stacked(rows: list[dict], plt, out: Path) -> None:
    import numpy as np

    labels = [r["app"] for r in rows]
    x = np.arange(len(labels))
    cons = [r["consistent"] for r in rows]
    high = [r["high_risk"] for r in rows]
    not_obs = [r["not_obs"] for r in rows]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x, cons, label="consistent", color="#27ae60")
    ax.bar(x, high, bottom=cons, label="inconsistent_high_risk", color="#e74c3c")
    bottom2 = [c + h for c, h in zip(cons, high)]
    ax.bar(x, not_obs, bottom=bottom2, label="declared_not_observed", color="#f39c12")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Verdict 条数")
    ax.set_title("Lab4-1：分应用判定分布（exp_stage3_20260525_lab4_1）")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_lab41_verdict_by_app.png", dpi=160)
    plt.close(fig)


def plot_declared_vs_observed(rows: list[dict], plt, out: Path) -> None:
    import numpy as np

    labels = [r["app"] for r in rows]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(x - w / 2, [r["declared"] for r in rows], w, label="政策声明", color="#3498db")
    ax.bar(x + w / 2, [r["observed"] for r in rows], w, label="合并观测", color="#9b59b6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("PII 类型数量")
    ax.set_title("政策声明 vs 多源合并观测")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_lab41_declared_observed.png", dpi=160)
    plt.close(fig)


def build_pptx(rows: list[dict], fig_dir: Path) -> None:
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except ImportError as exc:
        print("python-pptx not installed; skip PPTX:", exc)
        return

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    def title_slide(title: str, subtitle: str) -> None:
        s = prs.slides.add_slide(prs.slide_layouts[0])
        s.shapes.title.text = title
        s.placeholders[1].text = subtitle

    def bullet_slide(title: str, bullets: list[str]) -> None:
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = title
        tf = s.placeholders[1].text_frame
        tf.clear()
        for i, line in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.size = Pt(18)

    def pic_slide(title: str, img: Path, cap: str = "") -> None:
        s = prs.slides.add_slide(prs.slide_layouts[5])
        s.shapes.title.text = title
        if img.is_file():
            s.shapes.add_picture(str(img), Inches(0.8), Inches(1.4), width=Inches(11.5))
        if cap:
            box = s.shapes.add_textbox(Inches(0.8), Inches(6.6), Inches(11.5), Inches(0.6))
            box.text_frame.text = cap

    title_slide(
        "Lab4-1 知情权审计汇报",
        "Android 隐私检测系统 · 证据驱动规则引擎\n"
        "9 App · exp_stage3_20260525_lab4_1 · 我的刀盾队",
    )

    bullet_slide(
        "汇报范围",
        [
            "本次仅汇报 Lab4-1（知情权），不展开 Lab4-2/3。",
            "核心问题：隐私政策声明的 PII 类型，是否与多源技术证据一致。",
            "样本：Keep、小红书、支付宝、LOFTER、智联招聘、世纪佳缘、好大夫、东方财富、小荷健康。",
            "批量结果：9/9 成功，合计 consistent=90、high_risk=41、not_observed=6。",
        ],
    )

    bullet_slide(
        "系统如何支撑 Lab4-1",
        [
            "policy_parser：从政策文本抽取 declared_pii[]",
            "Stage2 静态 + Stage3 动态 + oldsrc + KB → evidence_loader 合并 observed_pii[]",
            "compare_engine：输出 consistent / inconsistent_high_risk / declared_not_observed",
            "每条 verdict 带 sources / provenance，可答辩溯源",
        ],
    )

    pic_slide("Lab4-1 数据流", fig_dir / "fig_lab41_pipeline.png")

    pic_slide(
        "全样本判定合计",
        fig_dir / "fig_lab41_verdict_totals.png",
        "主结论：41 条高风险不一致 = 观测到但政策未充分声明",
    )

    pic_slide(
        "政策声明 vs 合并观测",
        fig_dir / "fig_lab41_declared_observed.png",
        "观测集常大于声明集（含 static、oldsrc、KB）",
    )

    pic_slide("分应用 Verdict 分布", fig_dir / "fig_lab41_verdict_by_app.png")

    bullet_slide(
        "分 App 摘要",
        [
            f"{r['app']}: 声明{r['declared']} / 观测{r['observed']} — 一致{r['consistent']} 高风险{r['high_risk']} 未观测{r['not_obs']}"
            for r in rows
        ],
    )

    bullet_slide(
        "证据说明（答辩必讲）",
        [
            "observed_pii 来自多源合并，不单等于本次 Frida 实测。",
            "本批 dynamic_evidence 中 frida_runtime_events 多为 0。",
            "inconsistent_high_risk 需结合 sources 分项解释（static / kb / oldsrc）。",
            "详细报告：src/output/result/result_20260525.md",
        ],
    )

    bullet_slide(
        "典型案例话术",
        [
            "现象：observed_count > declared_count",
            "解读：技术侧或 KB 有信号，政策未写全 → inconsistent_high_risk",
            "演示：打开 output/lab4_1/<App>_lab4_1_report.json 展示一条 sources",
        ],
    )

    bullet_slide(
        "总结",
        [
            "完成 9 App 可解释知情权批量审计（规则引擎 + 证据溯源）。",
            "发现：多款 App 存在「声明不足」类高风险信号。",
            "后续（不在本次汇报）：加强 Frida 运行时、扩展 Lab4-2/3。",
        ],
    )

    title_slide("感谢聆听", "Q & A")

    out = PPT / "Lab4-1知情权汇报.pptx"
    prs.save(str(out))
    print("Wrote", str(out))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_lab41_reports()
    plt = _configure_matplotlib()
    plot_lab41_pipeline(plt, FIG)
    plot_verdict_totals(plt, FIG)
    plot_lab41_verdict_stacked(rows, plt, FIG)
    plot_declared_vs_observed(rows, plt, FIG)
    print("Wrote figures to", FIG)
    build_pptx(rows, FIG)


if __name__ == "__main__":
    main()
