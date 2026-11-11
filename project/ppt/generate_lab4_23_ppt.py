"""Generate Lab4-2 & Lab4-3 presentation: figures + PPTX.

Usage (from project/ppt):
  python generate_lab4_23_ppt.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PPT = Path(__file__).resolve().parent
SRC = PPT.parent / "src"
FIG = PPT / "figures_lab4_23"
OUT = SRC / "output"

APPS_FILE = SRC / "input" / "apps_full_batch_15.txt"
LAB41_SUMMARY = OUT / "lab4_1" / "batch_lab4-1_lab4_all_20260608_lab4_1_summary.json"
MANUAL_COPY_APPS = ["小荷健康", "世纪佳缘", "智联招聘", "支付宝", "东方财富"]

SCREENSHOT_SOURCES = {
    "fig_alipay_setting.png": SRC
    / "privacy_audit/utils/screenGrab/output/20260609_050409_setting/screenshot.png",
    "fig_alipay_user_center.png": SRC
    / "privacy_audit/utils/screenGrab/output/20260609_050429_userDemand/screenshot.png",
    "fig_alipay_path_fail.png": OUT
    / "stage3_lab42_after/支付宝/screenshots/支付宝_20260609_133225_path1.png",
    "fig_keep_privacy.png": OUT
    / "stage3_lab42_after/Keep/screenshots/Keep_20260609_133910_path3.png",
    "fig_xhs_privacy.png": OUT
    / "stage3_lab42_after/小红书/screenshots/小红书_20260609_132510_path3.png",
    "fig_zhaopin_privacy.png": OUT
    / "stage3_lab42_after/智联招聘/screenshots/智联招聘_20260609_132158_path2.png",
    "fig_diaoyuren_network.png": OUT
    / "stage3_lab42_after/钓鱼人/screenshots/钓鱼人_20260612_075600_path1.png",
}


def _load_app_list() -> list[str]:
    apps: list[str] = []
    for line in APPS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            apps.append(line)
    return apps


LAB42_APPS = _load_app_list()


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


def load_lab42_rows() -> list[dict]:
    rows: list[dict] = []
    for app in LAB42_APPS:
        path = OUT / "lab4_2" / f"{app}_lab4_2_result.json"
        if not path.exists():
            continue
        rep = json.loads(path.read_text(encoding="utf-8"))
        t3 = rep.get("track3_network") or {}
        bs, afs = t3.get("before_summary") or {}, t3.get("after_summary") or {}
        trackers = [
            f"{x['tracker_company']}({x['before_hits']}→{x['after_hits']})"
            for x in (t3.get("tracker_comparisons") or [])
            if "still" in x.get("verdict", "") or "new_tracker" in x.get("verdict", "")
        ]
        rows.append(
            {
                "app": app,
                "status": rep.get("status", "review_needed"),
                "b_flow": bs.get("flow_count", 0),
                "a_flow": afs.get("flow_count", 0),
                "b_trk": bs.get("tracker_hit_count", 0),
                "a_trk": afs.get("tracker_hit_count", 0),
                "leak": t3.get("still_leaking_count", 0),
                "trackers": trackers,
            }
        )
    return rows


def load_lab41_rows() -> list[dict]:
    if not LAB41_SUMMARY.is_file():
        return []
    data = json.loads(LAB41_SUMMARY.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for rep in data.get("reports", []):
        app = rep.get("claims", {}).get("app_name") or rep.get("evidence", {}).get("app_name")
        if app not in LAB42_APPS:
            continue
        sm = rep.get("summary") or {}
        sc = sm.get("status_counts") or {}
        rows.append(
            {
                "app": app,
                "declared": sm.get("declared_count", 0),
                "observed": sm.get("observed_count", 0),
                "consistent": sc.get("consistent", 0),
                "high_risk": sc.get("inconsistent_high_risk", 0),
            }
        )
    order = {a: i for i, a in enumerate(LAB42_APPS)}
    rows.sort(key=lambda r: order.get(r["app"], 99))
    return rows


def nav_stats(evidence_path: Path) -> dict:
    d = json.loads(evidence_path.read_text(encoding="utf-8"))
    steps = d.get("ui_result", {}).get("steps", [])
    paths = d.get("ui_result", {}).get("navigation_paths", [])
    ok = fail = 0
    for i in range(1, len(paths) + 1):
        if any(s.startswith(f"nav_path_{i}_stopped_at") for s in steps):
            fail += 1
        else:
            ok += 1
    return {"paths": len(paths), "ok": ok, "fail": fail}


def load_nav_rows(phase: str) -> list[dict]:
    rows = []
    for app in LAB42_APPS:
        p = OUT / phase / app / "dynamic_evidence.json"
        if not p.exists():
            continue
        s = nav_stats(p)
        total = s["paths"] or 1
        rows.append(
            {
                "app": app,
                "ok": s["ok"],
                "fail": s["fail"],
                "rate": 100.0 * s["ok"] / total,
            }
        )
    return rows


def load_lab43_rows() -> list[dict]:
    rows = []
    for app in LAB42_APPS:
        path = OUT / "lab4_3" / f"{app}_lab4_3_result.json"
        if not path.exists():
            continue
        rep = json.loads(path.read_text(encoding="utf-8"))
        sm = rep.get("summary") or {}
        auto_fields = [
            row.get("copy_field", "")
            for row in (rep.get("manual_copy_audit") or [])
            if row.get("auto_match")
        ]
        rows.append(
            {
                "app": rep.get("app_name", app),
                "declared": sm.get("declared_count", 0),
                "copy_fields": sm.get("user_copy_field_count", 0),
                "auto_matched": sm.get("manual_audit_auto_matched_fields", 0),
                "unmatched": sm.get("manual_audit_unmatched_fields", 0),
                "auto_field_labels": "、".join(auto_fields[:4]) or "—",
                "verdict": rep.get("verdict", "review_needed"),
            }
        )
    return rows


def copy_screenshots() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for name, src in SCREENSHOT_SOURCES.items():
        dst = FIG / name
        if src.is_file():
            shutil.copy2(src, dst)
            print("Copied", name)
        else:
            print("MISSING screenshot:", src)


def plot_stage3_pipeline(plt, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 2.4))
    ax.axis("off")
    boxes = [
        "政策解析\npolicy_path_hints",
        "Appium\n路径导航",
        "LLM 桥接\nllm_path_assistant",
        "mitmproxy\n网络主轨(R3)",
        "截图+关键词\npage_source",
        "dynamic_\nevidence.json",
    ]
    colors = ["#d6eaf8", "#fdebd0", "#e8daef", "#fadbd8", "#d5f5e3", "#f9e79f"]
    for i, (text, color) in enumerate(zip(boxes, colors)):
        ax.add_patch(plt.Rectangle((i * 1.85, 0.15), 1.65, 1.0, fc=color, ec="#333", lw=1.2))
        ax.text(i * 1.85 + 0.825, 0.65, text, ha="center", va="center", fontsize=9)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(i * 1.85 + 1.65, 0.65),
                xytext=((i + 1) * 1.85, 0.65),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )
    ax.set_xlim(-0.2, len(boxes) * 1.85)
    ax.set_ylim(0, 1.4)
    fig.savefig(out / "fig_stage3_pipeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_lab42_pipeline(plt, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 2.0))
    ax.axis("off")
    boxes = [
        "Stage3 before\n(mitm+UI)",
        "Appium 导航\n+ mitm 抓包",
        "人工关闭\n隐私开关",
        "Stage3 after\n(mitm+UI)",
        "网络/UI\nbefore-after 对比",
        "Lab4-2 报告\ntrack3_network",
    ]
    colors = ["#aed6f1", "#fdebd0", "#f5b7b1", "#aed6f1", "#d5f5e3", "#f9e79f"]
    for i, (text, color) in enumerate(zip(boxes, colors)):
        ax.add_patch(plt.Rectangle((i * 1.85, 0.2), 1.65, 0.9, fc=color, ec="#333", lw=1.2))
        ax.text(i * 1.85 + 0.825, 0.65, text, ha="center", va="center", fontsize=8.5)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(i * 1.85 + 1.65, 0.65),
                xytext=((i + 1) * 1.85, 0.65),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )
    ax.set_xlim(-0.2, len(boxes) * 1.85)
    ax.set_ylim(0, 1.4)
    fig.savefig(out / "fig_lab42_pipeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_lab43_pipeline(plt, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 2.0))
    ax.axis("off")
    boxes = [
        "政策 declared_pii",
        "动态证据\nlab4_3_dynamic",
        "副本导出\nlab4_3_copy",
        "字段映射\nlab4-3-audit",
        "覆盖率报告",
    ]
    colors = ["#d6eaf8", "#fdebd0", "#d5f5e3", "#e8daef", "#f9e79f"]
    for i, (text, color) in enumerate(zip(boxes, colors)):
        ax.add_patch(plt.Rectangle((i * 1.95, 0.2), 1.75, 0.9, fc=color, ec="#333", lw=1.2))
        ax.text(i * 1.95 + 0.875, 0.65, text, ha="center", va="center", fontsize=9)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(i * 1.95 + 1.75, 0.65),
                xytext=((i + 1) * 1.95, 0.65),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )
    ax.set_xlim(-0.2, len(boxes) * 1.95)
    ax.set_ylim(0, 1.4)
    fig.savefig(out / "fig_lab43_pipeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_lab42_status(lab42: list[dict], plt, out: Path) -> None:
    import numpy as np

    pass_n = sum(1 for r in lab42 if r["status"] == "pass")
    rev_n = sum(1 for r in lab42 if r["status"] == "review_needed")
    trk_n = sum(1 for r in lab42 if r["trackers"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].pie(
        [pass_n, rev_n],
        labels=[f"pass ({pass_n})", f"review_needed ({rev_n})"],
        colors=["#27ae60", "#e67e22"],
        autopct="%1.0f%%",
        startangle=90,
    )
    axes[0].set_title(f"Lab4-2 判定（{len(lab42)} App）")
    axes[1].bar(
        ["Tracker仍传输", "仅PII/其他", "pass无泄漏"],
        [trk_n, rev_n - trk_n, pass_n],
        color=["#c0392b", "#f39c12", "#27ae60"],
    )
    axes[1].set_ylabel("App 数")
    axes[1].set_title("关开关后网络证据分类")
    fig.tight_layout()
    fig.savefig(out / "fig_lab42_status.png", dpi=160)
    plt.close(fig)


def plot_lab42_tracker_leaks(lab42: list[dict], plt, out: Path) -> None:
    import numpy as np

    rows = [r for r in lab42 if r["trackers"]]
    if not rows:
        return
    labels = [r["app"] for r in rows]
    leaks = [r["leak"] for r in rows]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
    ax.barh(y, leaks, color="#c0392b", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("still_leaking 项数")
    ax.set_title("Lab4-2：关隐私开关后仍有网络泄漏的 App（Tracker 命中）")
    for i, r in enumerate(rows):
        ax.text(leaks[i] + 0.15, i, " ".join(r["trackers"][:2]), va="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig_lab42_tracker_leaks.png", dpi=160)
    plt.close(fig)


def plot_lab42_flows(lab42: list[dict], plt, out: Path) -> None:
    import numpy as np

    labels = [r["app"] for r in lab42]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, [r["b_flow"] for r in lab42], w, label="before 流量", color="#3498db")
    ax.bar(x + w / 2, [r["a_flow"] for r in lab42], w, label="after 流量", color="#9b59b6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mitm 捕获条数")
    ax.set_title("Lab4-2：before / after 网络流量（15 App）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_lab42_flows.png", dpi=160)
    plt.close(fig)


def plot_nav_success(rows: list[dict], plt, out: Path) -> None:
    import numpy as np

    if not rows:
        return
    labels = [r["app"] for r in rows]
    x = np.arange(len(labels))
    ok = [r["ok"] for r in rows]
    fail = [r["fail"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x, ok, label="路径成功", color="#27ae60")
    ax.bar(x, fail, bottom=ok, label="路径失败", color="#e74c3c")
    for i, r in enumerate(rows):
        if r["ok"] + r["fail"] > 0:
            ax.text(i, ok[i] + fail[i] + 0.05, f"{r['rate']:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("政策路径条数")
    ax.set_title("Lab4-2 动态验证：政策路径导航成功率（after 阶段）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_lab42_nav_success.png", dpi=160)
    plt.close(fig)


def plot_lab41_focus(rows: list[dict], plt, out: Path) -> None:
    import numpy as np

    if not rows:
        return
    labels = [r["app"] for r in rows]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w / 2, [r["declared"] for r in rows], w, label="政策声明", color="#3498db")
    ax.bar(x + w / 2, [r["observed"] for r in rows], w, label="合并观测", color="#9b59b6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("PII 类型数")
    ax.set_title("Lab4-1 知情权（本批样本；mitm 可能影响动态观测）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_lab41_focus.png", dpi=160)
    plt.close(fig)


def plot_lab43_coverage(lab43: list[dict], plt, out: Path) -> None:
    import numpy as np

    if not lab43:
        return
    labels = [r["app"] for r in lab43]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w / 2, [r["declared"] for r in lab43], w, label="声明类型数", color="#3498db")
    ax.bar(x + w / 2, [r["auto_matched"] for r in lab43], w, label="副本自动匹配", color="#16a085")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("字段/类型数")
    ax.set_title(f"Lab4-3：副本字段自动匹配 / 声明类型（{len(lab43)} App 有报告）")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_lab43_copy_vs_declared.png", dpi=160)
    plt.close(fig)


def plot_lab43_manual_results(lab43: list[dict], plt, out: Path) -> None:
    """Real user-copy batch: field counts only (no per-PII-type column detail)."""
    import numpy as np

    by_app = {r["app"]: r for r in lab43}
    rows = [by_app[a] for a in MANUAL_COPY_APPS if a in by_app]
    if not rows:
        return
    labels = [r["app"] for r in rows]
    x = np.arange(len(labels))
    w = 0.32
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w / 2, [r["copy_fields"] for r in rows], w, label="副本字段数", color="#9b59b6")
    ax.bar(x + w / 2, [r["declared"] for r in rows], w, label="政策声明类型数", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("数量")
    ax.set_title(f"Lab4-3：真实副本批（{len(rows)} App）— 字段规模对比")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_lab43_manual_results.png", dpi=160)
    plt.close(fig)


def plot_verification_stack(plt, out: Path) -> None:
    labels = ["Appium UI", "mitm 网络", "Frida(可选)"]
    scores = [2.4, 2.2, 0.4]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ["#27ae60", "#2980b9", "#e74c3c"]
    bars = ax.bar(labels, scores, color=colors)
    for b, v in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:g}", ha="center", fontweight="bold")
    ax.set_ylim(0, 3)
    ax.set_ylabel("相对权重（示意）")
    ax.set_title("Lab4-2 动态验证：mitm 网络为主轨，Appium UI 为辅")
    fig.tight_layout()
    fig.savefig(out / "fig_frida_vs_appium.png", dpi=160)
    plt.close(fig)


def build_pptx(
    lab41: list[dict],
    lab42: list[dict],
    nav: list[dict],
    lab43: list[dict],
    fig_dir: Path,
) -> None:
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except ImportError as exc:
        print("python-pptx not installed; skip PPTX:", exc)
        return

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    def title_slide(title: str, sub: str) -> None:
        s = prs.slides.add_slide(prs.slide_layouts[0])
        s.shapes.title.text = title
        s.placeholders[1].text = sub

    def bullet_slide(title: str, bullets: list[str], font_pt: int = 17) -> None:
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = title
        tf = s.placeholders[1].text_frame
        tf.clear()
        for i, line in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.size = Pt(font_pt)

    def pic_slide(title: str, img: Path, cap: str = "") -> None:
        s = prs.slides.add_slide(prs.slide_layouts[5])
        s.shapes.title.text = title
        if img.is_file():
            s.shapes.add_picture(str(img), Inches(0.7), Inches(1.3), width=Inches(11.8))
        if cap:
            box = s.shapes.add_textbox(Inches(0.7), Inches(6.55), Inches(11.8), Inches(0.7))
            box.text_frame.text = cap

    def two_pic_slide(title: str, left: Path, right: Path, cap: str = "") -> None:
        s = prs.slides.add_slide(prs.slide_layouts[5])
        s.shapes.title.text = title
        if left.is_file():
            s.shapes.add_picture(str(left), Inches(0.5), Inches(1.4), width=Inches(5.8))
        if right.is_file():
            s.shapes.add_picture(str(right), Inches(6.8), Inches(1.4), width=Inches(5.8))
        if cap:
            box = s.shapes.add_textbox(Inches(0.5), Inches(6.5), Inches(12.2), Inches(0.7))
            box.text_frame.text = cap

    pass_n = sum(1 for r in lab42 if r["status"] == "pass")
    rev_n = sum(1 for r in lab42 if r["status"] == "review_needed")
    trk_n = sum(1 for r in lab42 if r["trackers"])
    total_paths = sum(r["ok"] + r["fail"] for r in nav)
    total_ok = sum(r["ok"] for r in nav)

    title_slide(
        "Lab4-2 / Lab4-3 动态验证实验汇报",
        f"Android 隐私检测系统 · mitm 网络主轨 + Appium UI\n"
        f"15 App 全量 Lab4-2 · {len(lab43)} App Lab4-3 · {REPORT_DATE}",
    )

    bullet_slide(
        "汇报范围",
        [
            f"Lab4-2 同意权：{len(lab42)} App before/after mitm+UI 对比（apps_full_batch_15）",
            f"Lab4-3 访问权：{len(lab43)} App 副本人工审核（其余待导出副本）",
            "策略升级：mitm-primary 替代 Frida-only；关开关后 Tracker/PII 网络对比",
            "LLM 桥接路径导航、支付宝 H5 特例",
            "Lab4-1 受 mitm 代理影响，本汇报以 Lab4-2/3 为主",
        ],
    )

    bullet_slide(
        "Stage3 + Lab4-2 方法",
        [
            "stage3-dynamic --dynamic-track lab42 --real-run：UI + mitmproxy 同步采集",
            "BEFORE：隐私开关开启 → AFTER：真机手动关闭开关",
            "主判定 track3_network：tracker_hits / pii_key_hits before vs after",
            "辅助 track UI：compare_lab42_ui_evidence（关键词、tap 稳定性）",
            "每 App 跑完自动 adb force-stop，批量 15 App 约 2h",
        ],
        font_pt=16,
    )

    pic_slide("Stage3 动态验证管线（mitm 主轨）", fig_dir / "fig_stage3_pipeline.png")
    pic_slide("Lab4-2 同意权实验管线", fig_dir / "fig_lab42_pipeline.png")

    bullet_slide(
        f"Lab4-2 总览（{len(lab42)} App）",
        [
            f"pass：{pass_n}（Keep、支付宝、韭研公社）",
            f"review_needed：{rev_n}",
            f"关开关后 Tracker 仍传输：{trk_n} App",
            "强案例：钓鱼人(Umeng/Bugly/百度)、好大夫(Umeng×6)、快看漫画、世纪佳缘",
            "正面对照：Keep（关后 PII 网络字段停止）",
        ],
    )

    pic_slide("Lab4-2 判定分布", fig_dir / "fig_lab42_status.png")
    pic_slide("Lab4-2 before/after 网络流量", fig_dir / "fig_lab42_flows.png")
    pic_slide(
        "关开关后 Tracker 仍传输",
        fig_dir / "fig_lab42_tracker_leaks.png",
        "钓鱼人、好大夫、快看漫画、世纪佳缘等为汇报重点",
    )

    lab42_lines = [
        f"{r['app']}: {r['status']} | 流{r['b_flow']}→{r['a_flow']} | "
        f"trk {r['b_trk']}→{r['a_trk']} | "
        + ("; ".join(r["trackers"][:2]) if r["trackers"] else "无 tracker 泄漏")
        for r in lab42
    ]
    for i in range(0, len(lab42_lines), 8):
        bullet_slide(
            f"Lab4-2 逐 App 结果 ({i + 1}–{min(i + 8, len(lab42_lines))})",
            lab42_lines[i : i + 8],
            font_pt=14,
        )

    pic_slide(
        f"政策路径导航成功率（{len(nav)} App，after）",
        fig_dir / "fig_lab42_nav_success.png",
        f"合计 {total_ok}/{total_paths} 条路径成功",
    )

    pic_slide("Keep 隐私设置页", fig_dir / "fig_keep_privacy.png")
    pic_slide("钓鱼人 Lab4-2 after 截图", fig_dir / "fig_diaoyuren_network.png", "网络证据：Umeng/Bugly/百度广告关后仍传")

    bullet_slide(
        "支付宝 H5 特例（Appium 受限）",
        [
            "设置/用户保护中心在 Nebula/XRiver WebView，page_source 常无 text",
            "tap[隐私设置]=miss；UI 导航成功率低于其他 App",
            "本批 Lab4-2 网络 pass（1→1 条流，无 tracker）；证据偏弱",
        ],
        font_pt=16,
    )

    two_pic_slide(
        "支付宝：设置页 vs 用户保护中心",
        fig_dir / "fig_alipay_setting.png",
        fig_dir / "fig_alipay_user_center.png",
    )

    pic_slide("Lab4-3 访问权实验管线", fig_dir / "fig_lab43_pipeline.png")

    bullet_slide(
        f"Lab4-3 本批（{len(lab43)}/{len(LAB42_APPS)} App 有报告）",
        [
            "流程：政策 declared_pii + 用户导出副本 JSON → manual_copy_audit 人工审核",
            "自动覆盖率在 Frida/mitm 未产出 runtime_observed 时为 N/A",
            "待补：杉果游戏、小黑盒、韭研公社、咪噜游戏、钓鱼人、快看漫画 副本导出",
        ]
        + [
            f"{r['app']}: 声明{r['declared']}类 · 副本字段{r['copy_fields']} · 自动匹配{r['auto_matched']}（{r['auto_field_labels']}）"
            for r in lab43[:6]
        ],
        font_pt=14,
    )

    if len(lab43) > 6:
        bullet_slide(
            "Lab4-3 逐 App（续）",
            [
                f"{r['app']}: 声明{r['declared']} · 匹配{r['auto_matched']}/{r['copy_fields']} · {r['auto_field_labels']}"
                for r in lab43[6:]
            ],
            font_pt=14,
        )

    pic_slide("Lab4-3 副本自动匹配", fig_dir / "fig_lab43_copy_vs_declared.png")

    if lab41:
        pic_slide("Lab4-1 知情权（参考；mitm 可能抬高观测）", fig_dir / "fig_lab41_focus.png")

    pic_slide("验证手段权重", fig_dir / "fig_frida_vs_appium.png")

    bullet_slide(
        "总结",
        [
            f"完成 15 App Lab4-2 mitm before/after 全量实验",
            f"{trk_n} App 关隐私开关后 Tracker 仍传输 — 同意权违规风险",
            f"{pass_n} App 网络 track pass（含低流量弱证据）",
            f"Lab4-3 已有 {len(lab43)} 份报告，副本审核需继续扩展",
            "Lab4-1 受 mitm 拦截影响，汇报以 Lab4-2/3 为主",
        ],
    )

    title_slide("感谢聆听", "Q & A")

    out_path = PPT / "Lab4-2-3动态验证汇报.pptx"
    prs.save(str(out_path))
    print("Wrote", out_path)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    copy_screenshots()
    lab41 = load_lab41_rows()
    lab42 = load_lab42_rows()
    nav = load_nav_rows("stage3_lab42_after")
    lab43 = load_lab43_rows()
    plt = _configure_matplotlib()
    plot_stage3_pipeline(plt, FIG)
    plot_lab42_pipeline(plt, FIG)
    plot_lab43_pipeline(plt, FIG)
    plot_lab42_status(lab42, plt, FIG)
    plot_lab42_flows(lab42, plt, FIG)
    plot_lab42_tracker_leaks(lab42, plt, FIG)
    plot_nav_success(nav, plt, FIG)
    if lab41:
        plot_lab41_focus(lab41, plt, FIG)
    plot_lab43_coverage(lab43, plt, FIG)
    plot_lab43_manual_results(lab43, plt, FIG)
    plot_verification_stack(plt, FIG)
    print("Wrote figures to", FIG)
    print(f"Lab42 rows: {len(lab42)}, Lab43 rows: {len(lab43)}")
    build_pptx(lab41, lab42, nav, lab43, FIG)


if __name__ == "__main__":
    main()
