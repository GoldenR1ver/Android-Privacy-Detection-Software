"""Generate Lab4 presentation assets: figures + optional PPTX.

Reads batch summaries from project/src/output and writes:
  - ppt/figures_lab4/*.png
  - ppt/系统设计Lab4.pptx  (requires python-pptx)

Usage (from project/ppt):
  python generate_lab4_ppt.py
  python generate_lab4_ppt.py --pptx-only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PPT = Path(__file__).resolve().parent
SRC = PPT.parent / "src"
FIG = PPT / "figures_lab4"
SUMMARY_LAB41 = SRC / "output" / "lab4_1" / "batch_lab4-1_exp_stage3_20260525_lab4_1_summary.json"

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


def _configure_matplotlib() -> None:
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
    data = json.loads(SUMMARY_LAB41.read_text(encoding="utf-8"))
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


def plot_lab41_verdict_stacked(rows: list[dict], plt, out: Path) -> None:
    import numpy as np

    labels = [r["app"] for r in rows]
    x = np.arange(len(labels))
    cons = [r["consistent"] for r in rows]
    high = [r["high_risk"] for r in rows]
    not_obs = [r["not_obs"] for r in rows]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x, cons, label="consistent（一致）", color="#27ae60")
    ax.bar(x, high, bottom=cons, label="inconsistent_high_risk", color="#e74c3c")
    bottom2 = [c + h for c, h in zip(cons, high)]
    ax.bar(x, not_obs, bottom=bottom2, label="declared_not_observed", color="#f39c12")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Verdict 条数")
    ax.set_title("Lab4-1 知情权：9 款应用判定分布（exp_stage3_20260525_lab4_1）")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_lab41_verdict_stacked.png", dpi=160)
    plt.close(fig)


def plot_declared_vs_observed(rows: list[dict], plt, out: Path) -> None:
    import numpy as np

    labels = [r["app"] for r in rows]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(x - w / 2, [r["declared"] for r in rows], w, label="政策声明类型数", color="#3498db")
    ax.bar(x + w / 2, [r["observed"] for r in rows], w, label="合并观测类型数", color="#9b59b6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("PII 类型数量")
    ax.set_title("政策声明 vs 多源合并观测（Lab4-1）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_lab41_declared_observed.png", dpi=160)
    plt.close(fig)


def plot_batch_status(plt, out: Path) -> None:
    labels = ["Lab4-1\n知情权", "Lab4-3\n访问权", "Lab4-2\n同意权"]
    success = [9, 9, 0]
    fail = [0, 0, 9]
    import numpy as np

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x, success, color="#27ae60", label="成功")
    ax.bar(x, fail, bottom=success, color="#c0392b", label="失败")
    for i, (s, f) in enumerate(zip(success, fail)):
        ax.text(i, s / 2, str(s), ha="center", va="center", color="white", fontweight="bold")
        if f:
            ax.text(i, s + f / 2, str(f), ha="center", va="center", color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("App 数量")
    ax.set_title("Lab4 批量任务完成情况（9 App）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_lab4_batch_status.png", dpi=160)
    plt.close(fig)


def plot_pipeline_overview(plt, out: Path) -> None:
    """Simple horizontal pipeline schematic."""
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.axis("off")
    boxes = [
        "隐私政策\npolicy_parser",
        "APK 静态\nStage2",
        "真机动态\nStage3",
        "证据合并\nevidence_loader",
        "规则引擎\nLab4-1/2/3",
        "Batch 报告",
    ]
    colors = ["#d6eaf8", "#d5f5e3", "#fdebd0", "#e8daef", "#fadbd8", "#f9e79f"]
    n = len(boxes)
    for i, (text, color) in enumerate(zip(boxes, colors)):
        ax.add_patch(plt.Rectangle((i * 1.55, 0.2), 1.35, 0.9, fc=color, ec="#333", lw=1.2))
        ax.text(i * 1.55 + 0.675, 0.65, text, ha="center", va="center", fontsize=9)
        if i < n - 1:
            ax.annotate(
                "",
                xy=(i * 1.55 + 1.35, 0.65),
                xytext=((i + 1) * 1.55, 0.65),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )
    ax.set_xlim(-0.2, n * 1.55 + 0.3)
    ax.set_ylim(0, 1.4)
    fig.savefig(out / "fig_pipeline_overview.png", dpi=160, bbox_inches="tight")
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

    def add_title_slide(title: str, subtitle: str) -> None:
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text = title
        slide.placeholders[1].text = subtitle

    def add_bullet_slide(title: str, bullets: list[str]) -> None:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        tf = slide.placeholders[1].text_frame
        tf.clear()
        for i, line in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.level = 0
            p.font.size = Pt(18)

    def add_picture_slide(title: str, img: Path, caption: str = "") -> None:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = title
        if img.is_file():
            slide.shapes.add_picture(str(img), Inches(0.8), Inches(1.4), width=Inches(11.5))
        if caption:
            box = slide.shapes.add_textbox(Inches(0.8), Inches(6.6), Inches(11.5), Inches(0.6))
            box.text_frame.text = caption

    add_title_slide(
        "Android 隐私检测系统 — Lab4 设计与实验",
        "证据驱动的知情权 / 同意权 / 访问权自动化审计\n"
        "批次 exp_stage3_20260525 · 9 App · 我的刀盾队 · 2026-05-25",
    )

    add_bullet_slide(
        "汇报大纲",
        [
            "1. 研究背景与 Lab4 三项权利",
            "2. 系统总体架构（policy → static/dynamic → compare → batch）",
            "3. 核心模块：规则引擎 + 法律知识库 + Appium/Frida",
            "4. 实验设计：9 App、MuMu、job exp_stage3_20260525",
            "5. Lab4-1 知情权结果（主结论）",
            "6. Lab4-2 / Lab4-3 与 Stage3 UI 自动化",
            "7. 局限性与后续工作",
        ],
    )

    add_bullet_slide(
        "研究背景（对齐开发计划）",
        [
            "从「LLM 主导判断」转向「证据主导 + 规则引擎 + 可溯源报告」",
            "Lab4-1 知情权：隐私政策声明 vs 技术观测是否一致",
            "Lab4-2 同意权：关闭隐私开关前后运行时采集是否停止",
            "Lab4-3 访问权：政策声明、运行时与副本字段覆盖比对",
            "样本：Keep、小红书、支付宝、LOFTER、智联招聘、世纪佳缘、好大夫、东方财富、小荷健康",
        ],
    )

    add_bullet_slide(
        "系统总体架构",
        [
            "输入：input/pii_text/*.txt + APK + oldsrc 历史证据",
            "Stage2：Exodus + Androguard → static_evidence.json",
            "Stage3：Appium 政策路径导航 + Frida hooks → dynamic_evidence.json",
            "ingestion：evidence_loader 多源合并 + privacy_knowledgebase 增强",
            "输出：lab4_1_evidence/<App>.json → Lab4-1/2/3 batch 报告",
            "CLI：python -m privacy_audit.cli（kb-build / import-oldsrc / batch / stage3-dynamic）",
        ],
    )

    add_picture_slide(
        "端到端数据流",
        fig_dir / "fig_pipeline_overview.png",
        "政策 → 静态/动态 → 合并证据包 → 三项权利 batch",
    )

    add_bullet_slide(
        "核心设计原则（vs 早期 plan.md）",
        [
            "证据优先：每条 verdict 可追溯到 provenance / sources",
            "LLM 退居二线：Part1 Prompt 审计仅作历史输入，Lab4 判定不依赖 LLM",
            "统一 CLI 与 batch：retry、manifest、logs 归档",
            "App 适配：element_tap（底部 Tab、齿轮设置、头像入口等）",
        ],
    )

    add_picture_slide(
        "Lab4 批量任务完成情况",
        fig_dir / "fig_lab4_batch_status.png",
        "Lab4-1/3：9/9 成功；Lab4-2：0/9（无有效 Frida 前后对比）",
    )

    add_picture_slide(
        "Lab4-1：政策声明 vs 合并观测",
        fig_dir / "fig_lab41_declared_observed.png",
        "观测集常大于声明集（含 static、oldsrc、KB）",
    )

    add_picture_slide(
        "Lab4-1：分应用 Verdict 分布",
        fig_dir / "fig_lab41_verdict_stacked.png",
        "合计 consistent=90, high_risk=41, declared_not_observed=6",
    )

    table_lines = [
        f"{r['app']}: 声明{r['declared']} / 观测{r['observed']} — 一致{r['consistent']} 高风险{r['high_risk']}"
        for r in rows
    ]
    add_bullet_slide("Lab4-1 分 App 摘要", table_lines[:9])

    add_bullet_slide(
        "Stage3 动态取证要点",
        [
            "环境：MuMu + Appium 4723 + Frida attach",
            "9/9 生成 dynamic_evidence.json；frida_runtime_events 均为 0",
            "UI：世纪佳缘底部「我」Tab；小荷右上角齿轮进设置；支付宝精确「我的」",
            "截图与 stage3_run.log 可复核导航路径",
        ],
    )

    add_bullet_slide(
        "Lab4-2 同意权（局限说明）",
        [
            "设计：对比开关关闭前/后 Frida 有效命中",
            "实际：before=本次真机，after=oldsrc derived；且无 runtime 事件",
            "batch 结果：9/9 report_status_failed（review_needed）",
            "后续：关「个性化推荐」后二采 + 稳定 Frida attach",
        ],
    )

    add_bullet_slide(
        "Lab4-3 访问权（局限说明）",
        [
            "设计：declared vs runtime vs copy 字段覆盖",
            "实际：copy 来自 oldsrc access_result（可用性测试字段）",
            "runtime_observed_pii 为空；coverage_rate=0",
            "batch 程序 9/9 成功 ≠ 访问权合规结论",
        ],
    )

    add_bullet_slide(
        "总结与展望",
        [
            "已完成：端到端工程化流水线 + 9 App Lab4-1 可解释审计报告",
            "主要发现：多款 App 存在 inconsistent_high_risk（观测多于声明）",
            "待加强：Frida 运行时证据、Lab4-2 双采、Lab4-3 真实副本 JSON",
            "详细数据：src/output/result/result_20260525.md",
        ],
    )

    add_title_slide("感谢聆听", "Q & A")

    out_pptx = PPT / "系统设计Lab4.pptx"
    prs.save(str(out_pptx))
    print("Wrote", str(out_pptx))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pptx-only", action="store_true")
    args = parser.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_lab41_reports()

    if not args.pptx_only:
        plt = _configure_matplotlib()
        plot_lab41_verdict_stacked(rows, plt, FIG)
        plot_declared_vs_observed(rows, plt, FIG)
        plot_batch_status(plt, FIG)
        plot_pipeline_overview(plt, FIG)
        print("Wrote figures to", FIG)

    build_pptx(rows, FIG)


if __name__ == "__main__":
    main()
