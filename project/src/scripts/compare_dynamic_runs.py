#!/usr/bin/env python3
"""Compare two stage3 output trees (e.g. consent_before vs stage3_dynamic)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
APPS = """Keep
小红书
58同城
支付宝
LOFTER
智联招聘
好大夫医生版
小荷健康
东方财富
世纪佳缘""".strip().splitlines()


def load(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def metrics(d: dict) -> dict:
    ui = d.get("ui_result") or {}
    s = d.get("summary") or {}
    steps = ui.get("steps") or []
    taps_ok = sum(1 for x in steps if isinstance(x, str) and re.search(r"tap\[.+\]=ok", x))
    taps_fail = sum(1 for x in steps if isinstance(x, str) and re.search(r"tap\[.+\]=fail", x))
    nav = ui.get("navigation_paths") or []
    kw = ui.get("keyword_checks") or []
    kw_m = sum(1 for r in kw if isinstance(r, dict) and r.get("matched"))
    return {
        "screenshots": s.get("screenshots", len(ui.get("screenshots") or [])),
        "ui_steps": s.get("ui_steps", len(steps)),
        "keyword_match": s.get("keyword_match_count", kw_m),
        "keyword_total": s.get("keyword_checks", len(kw)),
        "taps_ok": taps_ok,
        "taps_fail": taps_fail,
        "nav_paths": len(nav),
        "nav_preview": [" > ".join(p) for p in nav[:3]],
        "frida_runtime": s.get("frida_runtime_events", 0),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    old_root = _SRC / "output" / "dynamic"
    new_root = _SRC / "output" / "consent_before"
    label_old, label_new = "dynamic", "consent_before"
    if len(sys.argv) >= 3:
        old_root = _SRC / sys.argv[1]
        new_root = _SRC / sys.argv[2]
        label_old, label_new = sys.argv[1], sys.argv[2]

    print(f"Compare: {label_old}  vs  {label_new}\n")
    print(f"{'App':<12} {'kw':>8} {'->':>3} {'kw':>8} {'tap_ok':>7} {'->':>3} {'tap_ok':>7} {'shots':>6} {'->':>3} {'shots':>6}  paths_changed")
    print("-" * 95)
    for app in APPS:
        o = metrics(load(old_root / app / "dynamic_evidence.json"))
        n = metrics(load(new_root / app / "dynamic_evidence.json"))
        if not o and not n:
            print(f"{app:<12} (both missing)")
            continue
        paths_chg = o.get("nav_preview") != n.get("nav_preview")
        kw_o, kw_n = o.get("keyword_match", "-"), n.get("keyword_match", "-")
        t_o, t_n = o.get("taps_ok", "-"), n.get("taps_ok", "-")
        s_o, s_n = o.get("screenshots", "-"), n.get("screenshots", "-")
        flag = "YES" if paths_chg else ""
        delta = ""
        if isinstance(kw_o, int) and isinstance(kw_n, int) and kw_n < kw_o:
            delta = " WORSE"
        print(f"{app:<12} {kw_o:>8} {'->':>3} {kw_n:>8} {t_o:>7} {'->':>3} {t_n:>7} {s_o:>6} {'->':>3} {s_n:>6}  {flag}{delta}")

    print("\n--- Detail: navigation_paths diff ---")
    for app in APPS:
        o = load(old_root / app / "dynamic_evidence.json")
        n = load(new_root / app / "dynamic_evidence.json")
        op = (o.get("ui_result") or {}).get("navigation_paths") or []
        np_ = (n.get("ui_result") or {}).get("navigation_paths") or []
        if op != np_:
            print(f"\n[{app}]")
            print(f"  OLD: {op}")
            print(f"  NEW: {np_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
