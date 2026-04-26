from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SHOTS_SCHEMA_VERSION = 1

DEFAULT_SHOTS_PATH = Path(__file__).resolve().parent / "ref" / "shots.json"


def default_shots_document() -> Dict[str, Any]:
    return {
        "schema_version": SHOTS_SCHEMA_VERSION,
        "description": "句级 pii_related 人工参考例（few-shot）；可由 review_bundle 提取或手改。",
        "updated_at_utc": None,
        "shots": [],
    }


def load_shots_json(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    读取 shots 文件。支持顶层为对象（含 shots 数组）或纯数组（兼容）。
    返回 (完整文档对象用于写回, shots 列表)。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        doc = default_shots_document()
        doc["shots"] = raw
        return doc, raw
    if isinstance(raw, dict):
        shots = raw.get("shots")
        if not isinstance(shots, list):
            shots = []
        base = default_shots_document()
        base.update(raw)
        base["shots"] = shots
        return base, shots
    raise ValueError("shots.json 顶层必须是对象或数组")


def normalize_shot(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """单条 shot：必须含 text 与布尔 related。"""
    text = str(row.get("text", "")).strip()
    rel = row.get("related")
    if not text or not isinstance(rel, bool):
        return None
    out: Dict[str, Any] = {
        "id": str(row.get("id", "")).strip() or None,
        "text": text,
        "related": rel,
    }
    rh = row.get("reason_hint") or row.get("reason")
    if rh is not None:
        out["reason_hint"] = str(rh).strip()
    src = row.get("source")
    if src:
        out["source"] = str(src)
    dk = row.get("disagreement_kind")
    if dk:
        out["disagreement_kind"] = str(dk)
    out = {k: v for k, v in out.items() if v is not None}
    return out


def shots_for_prompt(doc_shots: List[Dict[str, Any]], *, max_n: int) -> List[Dict[str, Any]]:
    """去无效项并截断长度，供 LLM few-shot。"""
    out: List[Dict[str, Any]] = []
    for row in doc_shots:
        n = normalize_shot(row)
        if n:
            out.append(n)
        if max_n > 0 and len(out) >= max_n:
            break
    return out


def _bundle_items(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = bundle.get("items")
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def extract_shots_from_review_bundle(
    bundle: Dict[str, Any],
    *,
    mode: str = "ai_true_human_false",
) -> List[Dict[str, Any]]:
    """
    从 review_bundle 提取用于 few-shot 的样例。
    mode:
      - ai_true_human_false: 模型判相关、人工判不相关（偏严口径）
      - ai_false_human_true: 模型不相关、人工相关
      - all_disagree: 上述两种分歧并集
    """
    out: List[Dict[str, Any]] = []
    for it in _bundle_items(bundle):
        ai = it.get("ai") or {}
        hum = it.get("human") or {}
        a_pr = ai.get("pii_related")
        h_pr = hum.get("pii_related")
        if not isinstance(a_pr, bool) or not isinstance(h_pr, bool):
            continue
        if a_pr == h_pr:
            continue
        if mode == "ai_true_human_false" and not (a_pr is True and h_pr is False):
            continue
        if mode == "ai_false_human_true" and not (a_pr is False and h_pr is True):
            continue
        text = str(it.get("text", "")).strip()
        if not text:
            continue
        notes = str(hum.get("notes", "")).strip()
        kind = "ai_true_human_false" if a_pr and not h_pr else "ai_false_human_true"
        reason = notes or (
            "人工与模型分歧：按个人信息处理行为相关性的更严口径，此句判为不相关。"
            if h_pr is False
            else "人工与模型分歧：此句实际涉及个人信息处理与利用。"
        )
        sid = str(it.get("id", "")).strip()
        shot: Dict[str, Any] = {
            "id": sid or None,
            "text": text,
            "related": h_pr,
            "reason_hint": reason,
            "source": "review_bundle",
            "disagreement_kind": kind,
        }
        out.append({k: v for k, v in shot.items() if v is not None})
    return out


def merge_shots_into_document(
    doc: Dict[str, Any],
    new_rows: List[Dict[str, Any]],
    *,
    replace_same_id: bool = True,
) -> Dict[str, Any]:
    """按 id 去重合并；无 id 的按 text+related 去重。"""
    existing: List[Dict[str, Any]] = list(doc.get("shots") or [])
    if not isinstance(existing, list):
        existing = []

    def key(r: Dict[str, Any]) -> str:
        rid = str(r.get("id", "")).strip()
        if rid:
            return f"id:{rid}"
        t = str(r.get("text", "")).strip()
        rel = r.get("related")
        return f"tx:{t}\x00{rel!r}"

    seen = {key(r) for r in existing}
    merged = list(existing)
    for r in new_rows:
        n = normalize_shot(r)
        if not n:
            continue
        k = key(n)
        if k in seen:
            if replace_same_id and k.startswith("id:"):
                merged = [x for x in merged if key(x) != k]
                merged.append(n)
                seen.add(k)
            continue
        merged.append(n)
        seen.add(k)
    doc = dict(doc)
    doc["schema_version"] = SHOTS_SCHEMA_VERSION
    doc["shots"] = merged
    doc["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return doc


def save_shots_document(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def load_shots_for_classify(path: Optional[Path], *, max_n: int) -> List[Dict[str, Any]]:
    """供 pipeline 调用：路径缺省或文件不存在则返回空列表。"""
    if path is None:
        return []
    p = path.expanduser()
    if not p.is_file():
        return []
    _, shots = load_shots_json(p)
    return shots_for_prompt(shots, max_n=max_n)


def cmd_extract(args: argparse.Namespace) -> None:
    bundle_path: Path = args.bundle
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    extracted = extract_shots_from_review_bundle(bundle, mode=args.mode)
    out_path: Path = args.out
    if args.merge and out_path.is_file():
        doc, _ = load_shots_json(out_path)
    else:
        doc = default_shots_document()
        if args.merge and not out_path.is_file():
            doc["note"] = "merge requested but file missing; created new"
    doc = merge_shots_into_document(doc, extracted, replace_same_id=True)
    if args.note:
        doc["note"] = str(args.note)
    save_shots_document(out_path, doc)
    print(
        json.dumps(
            {
                "wrote": str(out_path),
                "extracted": len(extracted),
                "total_shots": len(doc.get("shots") or []),
                "mode": args.mode,
            },
            ensure_ascii=False,
        )
    )


def cmd_validate(args: argparse.Namespace) -> None:
    path = args.path
    doc, shots = load_shots_json(path)
    errs: List[str] = []
    if doc.get("schema_version") != SHOTS_SCHEMA_VERSION:
        errs.append(f"schema_version 建议为 {SHOTS_SCHEMA_VERSION}")
    for i, row in enumerate(shots):
        if normalize_shot(row) is None:
            errs.append(f"shots[{i}] 缺少有效 text 或 related 非布尔")
    print(json.dumps({"ok": len(errs) == 0, "errors": errs, "rows": len(shots)}, ensure_ascii=False))
    if errs:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PII 句级 few-shot：从 review_bundle 提取写入 ref/shots.json，供 classify 使用。",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", help="从 review_bundle 提取分歧样例并写入 shots.json")
    p_ex.add_argument("--bundle", type=Path, required=True)
    p_ex.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SHOTS_PATH,
        help=f"输出路径（默认 {DEFAULT_SHOTS_PATH}）",
    )
    p_ex.add_argument(
        "--mode",
        choices=("ai_true_human_false", "ai_false_human_true", "all_disagree"),
        default="ai_true_human_false",
        help="提取哪类人机分歧（默认：模型相关、人工不相关）",
    )
    p_ex.add_argument(
        "--merge",
        action="store_true",
        help="与已有 shots.json 按 id 合并（同 id 覆盖）",
    )
    p_ex.add_argument("--note", default="", help="写入文档级 note")
    p_ex.set_defaults(func=cmd_extract)

    p_val = sub.add_parser("validate", help="校验 shots.json")
    p_val.add_argument("--path", type=Path, default=DEFAULT_SHOTS_PATH)
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
