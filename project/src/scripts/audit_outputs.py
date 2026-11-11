#!/usr/bin/env python3
"""Audit Disclosure audit/2/3 outputs against reporting requirements."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

APPS = [
    "Keep", "小红书", "58同城", "支付宝", "LOFTER",
    "智联招聘", "好大夫医生版", "小荷健康", "东方财富", "世纪佳缘",
]


def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    print("=== DYNAMIC_CAPTURE DYNAMIC (真机) ===")
    for app in APPS:
        p = SRC / "output/dynamic" / app / "dynamic_evidence.json"
        d = load_json(p)
        if not d:
            print(f"  {app}: MISSING")
            continue
        fr = d.get("frida_result") or {}
        summ = d.get("summary") or {}
        nav = d.get("navigation") or {}
        paths = nav.get("paths", []) if isinstance(nav, dict) else []
        path_labels = [p.get("label", p.get("name", "")) for p in paths if isinstance(p, dict)]
        shots = len(d.get("screenshots") or [])
        java_active = summ.get("java_active_count", fr.get("java_active_count", 0))
        events = fr.get("events") or []
        runtime = summ.get("frida_runtime_events")
        if runtime is None:
            from privacy_audit.analyzers.dynamic.hook_loader import is_frida_runtime_api
            runtime = sum(1 for e in events if isinstance(e, dict) and is_frida_runtime_api(str(e.get("api_name", ""))))
        dl_path = any("个人信息" in str(x) or "下载" in str(x) for x in path_labels)
        print(f"  {app}: screenshots={shots} nav_paths={len(paths)} dl_path={dl_path} java_active={java_active} runtime={runtime}")

    print("\n=== LAB4-1 ===")
    for app in APPS:
        p = SRC / "output/disclosure" / f"{app}_disclosure_report.json"
        d = load_json(p)
        if not d:
            print(f"  {app}: MISSING")
            continue
        s = d.get("summary") or {}
        verdicts = d.get("verdicts") or []
        counts: dict[str, int] = {}
        for v in verdicts:
            st = v.get("status", "?")
            counts[st] = counts.get(st, 0) + 1
        ev = d.get("evidence") or {}
        obs = ev.get("observed_pii") or d.get("observed_pii") or []
        prov = ev.get("provenance") or d.get("provenance") or {}
        sources = prov.get("sources") if isinstance(prov, dict) else []
        kb = ev.get("knowledge_base") or {}
        feasible = len(kb.get("feasible_evidence") or []) if isinstance(kb, dict) else 0
        print(f"  {app}: declared={s.get('declared_count','?')} observed={len(obs)} feasible_kb={feasible} sources={len(sources) if isinstance(sources,list) else '?'} verdicts={counts}")

    print("\n=== LAB4-2 ===")
    for app in APPS:
        p = SRC / "output/consent" / f"{app}_consent_result.json"
        d = load_json(p)
        if not d:
            print(f"  {app}: MISSING")
            continue
        s = d.get("summary") or {}
        same = Path(d.get("before_off_file", "")).resolve() == Path(d.get("after_off_file", "")).resolve() if d.get("before_off_file") and d.get("after_off_file") else False
        print(f"  {app}: status={d.get('status')} before_hits={s.get('before_runtime_hits')} after_hits={s.get('after_runtime_hits')} same_file={same}")

    print("\n=== LAB4-3 (latest reports) ===")
    for app in APPS:
        p = SRC / "output/access" / f"{app}_access_result.json"
        d = load_json(p)
        if not d:
            print(f"  {app}: MISSING")
            continue
        s = d.get("summary") or {}
        keys = d.get("copy_keys_preview") or []
        real_copy = [k for k in keys if k not in ("app_name", "case_id", "channel", "status") and not k.startswith("avg_") and not k.startswith("results")]
        user_norm = (SRC / "input/user_copy_normalized" / f"{app}.json").exists()
        print(f"  {app}: copy_keys={d.get('copy_key_count')} real_copy_preview={real_copy[:6]} runtime_obs={s.get('runtime_observed_count')} coverage={s.get('coverage_rate')} user_copy={user_norm}")

    print("\n=== EVIDENCE BUNDLES ===")
    for app in APPS:
        p = SRC / "output/evidence" / f"{app}.json"
        d = load_json(p)
        if not d:
            print(f"  {app}: MISSING bundle")
            continue
        obs = d.get("observed_pii") or []
        static = d.get("static_evidence") or {}
        dynamic = d.get("dynamic_evidence") or {}
        print(f"  {app}: observed={len(obs)} has_static={bool(static)} has_dynamic={bool(dynamic)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
