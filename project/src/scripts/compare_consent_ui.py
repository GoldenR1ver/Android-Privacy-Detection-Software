#!/usr/bin/env python3
"""
Consent audit UI-only consent comparison (no Frida runtime required).

Compares before/after dynamic_evidence.json using Appium signals:
  - keyword_match_count / keyword_checks (page_source probes)
  - navigation tap steps (tap[...]=ok vs fail)
  - toggle-related tokens (切换基本版, 隐私设置, 撤回授权, etc.)

Usage (from project/src):
  python scripts/compare_consent_ui.py --app-name 小荷健康 \\
    --before-file output/consent_before/小荷健康/dynamic_evidence.json \\
    --after-file output/consent_after/小荷健康/dynamic_evidence.json \\
    --output-dir output/consent_ui

  python scripts/compare_consent_ui.py --batch \\
    --before-dir output/consent_before \\
    --after-dir output/consent_after \\
    --apps-file input/apps_full_batch_15.txt \\
    --output-dir output/consent_ui
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_TOGGLE_TOKENS = (
    "切换基本版",
    "基本版",
    "隐私设置",
    "撤回授权",
    "管理设备隐私授权",
    "第三方数据合作",
    "个性化",
    "推荐",
)


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ui(data: dict[str, Any]) -> dict[str, Any]:
    ui = data.get("ui_result")
    return ui if isinstance(ui, dict) else {}


def _tap_outcomes(steps: list[Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for step in steps:
        if not isinstance(step, str):
            continue
        m = re.match(r"^tap\[(.+)\]=(.+)$", step.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _toggle_keyword_rows(keyword_checks: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in keyword_checks:
        if not isinstance(row, dict):
            continue
        token = str(row.get("step_token", ""))
        checks = row.get("checks") if isinstance(row.get("checks"), dict) else {}
        if any(t in token for t in _TOGGLE_TOKENS):
            rows.append(row)
            continue
        if any(checks.get(t) for t in _TOGGLE_TOKENS if t in checks):
            rows.append(row)
    return rows


def compare_ui_evidence(
    app_name: str,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    toggle_name: str = "privacy_switch",
) -> dict[str, Any]:
    b_ui = _ui(before)
    a_ui = _ui(after)
    b_sum = before.get("summary") if isinstance(before.get("summary"), dict) else {}
    a_sum = after.get("summary") if isinstance(after.get("summary"), dict) else {}

    b_kw = int(b_sum.get("keyword_match_count", 0))
    a_kw = int(a_sum.get("keyword_match_count", 0))
    b_taps = _tap_outcomes(b_ui.get("steps") or [])
    a_taps = _tap_outcomes(a_ui.get("steps") or [])
    b_toggle_rows = _toggle_keyword_rows(b_ui.get("keyword_checks") or [])
    a_toggle_rows = _toggle_keyword_rows(a_ui.get("keyword_checks") or [])

    tap_tokens = sorted(set(b_taps) | set(a_taps))
    tap_comparisons: list[dict[str, Any]] = []
    tap_regressions = 0
    for tok in tap_tokens:
        b_stat = b_taps.get(tok, "missing")
        a_stat = a_taps.get(tok, "missing")
        if b_stat == "ok" and a_stat != "ok":
            verdict = "path_broken_after_toggle"
            tap_regressions += 1
        elif b_stat != "ok" and a_stat == "ok":
            verdict = "path_improved_after_toggle"
        elif b_stat == a_stat == "ok":
            verdict = "path_stable"
        else:
            verdict = "path_uncertain"
        tap_comparisons.append(
            {
                "step_token": tok,
                "before": b_stat,
                "after": a_stat,
                "verdict": verdict,
            }
        )

    toggle_checks: list[dict[str, Any]] = []
    toggle_changed = 0
    for label, rows in (("before", b_toggle_rows), ("after", a_toggle_rows)):
        for row in rows:
            toggle_checks.append(
                {
                    "phase": label,
                    "path_index": row.get("path_index"),
                    "step_token": row.get("step_token"),
                    "matched": row.get("matched"),
                    "checks": row.get("checks", {}),
                }
            )
    # Heuristic: any toggle-row matched flag differs
    b_m = {str(r.get("step_token")): bool(r.get("matched")) for r in b_toggle_rows}
    a_m = {str(r.get("step_token")): bool(r.get("matched")) for r in a_toggle_rows}
    for tok in sorted(set(b_m) | set(a_m)):
        if b_m.get(tok) != a_m.get(tok):
            toggle_changed += 1

    ui_delta = a_kw - b_kw
    if tap_regressions == 0 and (ui_delta != 0 or toggle_changed > 0):
        status = "ui_change_observed"
    elif tap_regressions > 0:
        status = "review_needed"
    else:
        status = "no_ui_delta"

    return {
        "app_name": app_name,
        "lab": "4-2",
        "method": "ui_only_no_frida",
        "toggle_name": toggle_name,
        "status": status,
        "summary": {
            "before_keyword_match_count": b_kw,
            "after_keyword_match_count": a_kw,
            "keyword_match_delta": ui_delta,
            "before_screenshots": int(b_sum.get("screenshots", len(b_ui.get("screenshots") or []))),
            "after_screenshots": int(a_sum.get("screenshots", len(a_ui.get("screenshots") or []))),
            "tap_step_count_before": len(b_taps),
            "tap_step_count_after": len(a_taps),
            "tap_regressions_after_toggle": tap_regressions,
            "toggle_keyword_rows_before": len(b_toggle_rows),
            "toggle_keyword_rows_after": len(a_toggle_rows),
            "toggle_row_match_changes": toggle_changed,
        },
        "tap_comparisons": tap_comparisons,
        "toggle_keyword_checks": toggle_checks,
        "interpretation_hint": (
            "UI-only Consent audit: compare navigation reachability and page_source keywords "
            "before vs after manually disabling consent/basic-mode switch. "
            "Does not prove API-level collection stopped; pair with screenshots in report."
        ),
    }


def _load_apps(path: Path) -> list[str]:
    apps: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            apps.append(line)
    return apps


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Consent audit UI-only before/after comparison")
    parser.add_argument("--app-name", default="")
    parser.add_argument("--before-file", type=Path)
    parser.add_argument("--after-file", type=Path)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--before-dir", type=Path, default=Path("output/consent_before"))
    parser.add_argument("--after-dir", type=Path, default=Path("output/consent_after"))
    parser.add_argument("--apps-file", type=Path, default=Path("input/apps_full_batch_15.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/consent_ui"))
    parser.add_argument("--toggle-name", default="privacy_switch")
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir.is_absolute() else _SRC / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        apps_path = args.apps_file if args.apps_file.is_absolute() else _SRC / args.apps_file
        before_root = args.before_dir if args.before_dir.is_absolute() else _SRC / args.before_dir
        after_root = args.after_dir if args.after_dir.is_absolute() else _SRC / args.after_dir
        apps = _load_apps(apps_path)
        results: list[dict[str, Any]] = []
        for app in apps:
            b_path = before_root / app / "dynamic_evidence.json"
            a_path = after_root / app / "dynamic_evidence.json"
            if not b_path.exists() or not a_path.exists():
                print(f"SKIP {app}: missing before={b_path.exists()} after={a_path.exists()}")
                continue
            rep = compare_ui_evidence(app, _load(b_path), _load(a_path), toggle_name=args.toggle_name)
            out = out_dir / f"{app}_consent_ui_result.json"
            out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            rep["report_path"] = str(out)
            results.append(rep)
            print(f"OK {app} status={rep['status']} kw_delta={rep['summary']['keyword_match_delta']}")
        summary = {
            "apps": len(results),
            "output_dir": str(out_dir),
            "results": [{k: r.get(k) for k in ("app_name", "status", "report_path")} for r in results],
        }
        (out_dir / "batch_consent_ui_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 0 if results else 1

    if not args.app_name or not args.before_file or not args.after_file:
        parser.error("single-app mode requires --app-name --before-file --after-file (or use --batch)")
    b_path = args.before_file if args.before_file.is_absolute() else _SRC / args.before_file
    a_path = args.after_file if args.after_file.is_absolute() else _SRC / args.after_file
    rep = compare_ui_evidence(args.app_name, _load(b_path), _load(a_path), toggle_name=args.toggle_name)
    out = out_dir / f"{args.app_name}_consent_ui_result.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    rep["report_path"] = str(out)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
