from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from labeling_queue import load_jsonl

SCHEMA_VERSION = 1


def stable_item_id(doc_id: str, sent_index: int) -> str:
    return f"{doc_id}#{int(sent_index)}"


def _empty_human() -> Dict[str, Any]:
    return {
        "pii_related": None,
        "notes": "",
        "reviewed_at": None,
        "reviewer_id": "",
    }


def item_from_labeling_row(row: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = str(row.get("doc_id", ""))
    idx = int(row.get("sent_index", 0))
    return {
        "id": stable_item_id(doc_id, idx),
        "doc_id": doc_id,
        "sent_index": idx,
        "text": row.get("text", ""),
        "keyword_hint": row.get("keyword_hint"),
        "labeling_queue_rank": row.get("labeling_queue_rank"),
        "labeling_priority_score": row.get("labeling_priority_score"),
        "ai": {
            "pii_related": row.get("pii_related"),
            "confidence": row.get("confidence"),
            "raw_model_output": row.get("raw_model_output", "") or "",
        },
        "human": _empty_human(),
    }


def build_bundle(
    rows: List[Dict[str, Any]],
    *,
    source_path: str,
    note: str = "",
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "privacy_sentence_review_bundle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_jsonl": source_path,
        "note": note,
        "items": [item_from_labeling_row(r) for r in rows],
    }


def save_bundle(path: Path, bundle: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")


def load_bundle(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_bundle(bundle: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if bundle.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version 应为 {SCHEMA_VERSION}")
    items = bundle.get("items")
    if not isinstance(items, list):
        errors.append("items 必须是数组")
        return False, errors
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            errors.append(f"items[{i}] 不是对象")
            continue
        hid = it.get("id")
        hum = it.get("human")
        if hum is None or not isinstance(hum, dict):
            errors.append(f"items[{i}] 缺少 human 对象")
            continue
        pr = hum.get("pii_related")
        reviewed_at = hum.get("reviewed_at")
        if pr is not None and not isinstance(pr, bool):
            errors.append(f"items[{i}] human.pii_related 必须是 true/false 或 null（当前: {pr!r}）")
        if reviewed_at and pr is None:
            errors.append(f"items[{i}] 已填 reviewed_at 但 pii_related 仍为 null")
    return len(errors) == 0, errors


def bundle_stats(bundle: Dict[str, Any]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = bundle.get("items") or []
    n = len(items)
    human_done = 0
    agree = disagree = unknown = 0
    ai_pos_hum_neg = ai_neg_hum_pos = 0
    for it in items:
        ai_pr = (it.get("ai") or {}).get("pii_related")
        hum = it.get("human") or {}
        h_pr = hum.get("pii_related")
        if h_pr is None:
            continue
        human_done += 1
        if ai_pr is True and h_pr is True:
            agree += 1
        elif ai_pr is False and h_pr is False:
            agree += 1
        elif ai_pr is None:
            unknown += 1
        else:
            disagree += 1
        if ai_pr is True and h_pr is False:
            ai_pos_hum_neg += 1
        if ai_pr is False and h_pr is True:
            ai_neg_hum_pos += 1
    return {
        "total_items": n,
        "human_labeled": human_done,
        "human_remaining": n - human_done,
        "pairwise_agree_when_ai_bool": agree,
        "pairwise_disagree_when_ai_bool": disagree,
        "ai_was_null": unknown,
        "ai_true_human_false": ai_pos_hum_neg,
        "ai_false_human_true": ai_neg_hum_pos,
    }


def export_split(bundle: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = bundle.get("items") or []
    sentences = []
    ai_only = []
    human_only = []
    for it in items:
        sid = it.get("id")
        sentences.append(
            {
                "id": sid,
                "doc_id": it.get("doc_id"),
                "sent_index": it.get("sent_index"),
                "text": it.get("text"),
                "keyword_hint": it.get("keyword_hint"),
                "labeling_queue_rank": it.get("labeling_queue_rank"),
                "labeling_priority_score": it.get("labeling_priority_score"),
            }
        )
        ai_only.append(
            {
                "id": sid,
                "ai": it.get("ai"),
            }
        )
        human_only.append(
            {
                "id": sid,
                "human": it.get("human") or _empty_human(),
            }
        )
    meta = {
        "schema_version": bundle.get("schema_version"),
        "kind": "split_export",
        "parent_created_at_utc": bundle.get("created_at_utc"),
        "source_jsonl": bundle.get("source_jsonl"),
    }
    paths = {
        "meta": out_dir / "review_meta.json",
        "sentences": out_dir / "sentences_for_review.json",
        "ai_evaluations": out_dir / "ai_evaluations.json",
        "human_evaluations": out_dir / "human_evaluations.json",
    }
    paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["sentences"].write_text(
        json.dumps(sentences, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["ai_evaluations"].write_text(
        json.dumps(ai_only, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["human_evaluations"].write_text(
        json.dumps(human_only, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {k: str(v) for k, v in paths.items()}


def merge_human_from_file(bundle: Dict[str, Any], human_path: Path) -> Dict[str, Any]:
    raw = json.loads(human_path.read_text(encoding="utf-8"))
    by_id: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, list):
        for row in raw:
            if isinstance(row, dict) and row.get("id"):
                by_id[str(row["id"])] = row.get("human") or {}
    elif isinstance(raw, dict):
        if "items" in raw and isinstance(raw["items"], dict):
            by_id = {str(k): v for k, v in raw["items"].items()}
        elif "items" in raw and isinstance(raw["items"], list):
            for row in raw["items"]:
                if isinstance(row, dict) and row.get("id"):
                    by_id[str(row["id"])] = row.get("human") or {}
    out = json.loads(json.dumps(bundle))
    for it in out.get("items") or []:
        sid = str(it.get("id", ""))
        if sid in by_id:
            patch = by_id[sid]
            base = dict(it.get("human") or _empty_human())
            for k, v in patch.items():
                if k in ("pii_related", "notes", "reviewed_at", "reviewer_id"):
                    base[k] = v
            it["human"] = base
    out["merged_human_from"] = str(human_path)
    out["merged_at_utc"] = datetime.now(timezone.utc).isoformat()
    return out


def cmd_init(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.from_jsonl)
    bundle = build_bundle(rows, source_path=str(args.from_jsonl.resolve()), note=args.note or "")
    save_bundle(args.out, bundle)
    print(json.dumps({"wrote": str(args.out), "items": len(rows)}, ensure_ascii=False))


def cmd_validate(args: argparse.Namespace) -> None:
    bundle = load_bundle(args.bundle)
    ok, errs = validate_bundle(bundle)
    print(json.dumps({"ok": ok, "errors": errs}, ensure_ascii=False))
    if not ok:
        raise SystemExit(1)


def cmd_stats(args: argparse.Namespace) -> None:
    bundle = load_bundle(args.bundle)
    s = bundle_stats(bundle)
    print(json.dumps(s, ensure_ascii=False))


def cmd_split(args: argparse.Namespace) -> None:
    bundle = load_bundle(args.bundle)
    paths = export_split(bundle, args.out_dir)
    print(json.dumps({"wrote": paths}, ensure_ascii=False))


def cmd_merge(args: argparse.Namespace) -> None:
    bundle = load_bundle(args.bundle)
    merged = merge_human_from_file(bundle, args.human)
    out = args.out or args.bundle
    save_bundle(Path(out), merged)
    print(json.dumps({"wrote": str(out), "items": len(merged.get("items", []))}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="本地 JSON：送标句 + AI 判断 + 人工标注（review_bundle）。",
        epilog=(
            "人工评价：用 init 生成 review_bundle.json 后，仅编辑各条 human 字段："
            "pii_related 填 true/false；可选 notes、reviewer_id；完成一条后填 reviewed_at（ISO8601）。"
            "也可用 split 导出 human_evaluations.json 单独编辑，再 merge 回总包。"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="从 for_labeling.jsonl / sentences.jsonl 生成总包")
    p_init.add_argument("--from-jsonl", type=Path, required=True)
    p_init.add_argument("--out", type=Path, required=True, help="如 review_bundle.json")
    p_init.add_argument("--note", default="")
    p_init.set_defaults(func=cmd_init)

    p_val = sub.add_parser("validate", help="校验 human 字段类型与一致性")
    p_val.add_argument("--bundle", type=Path, required=True)
    p_val.set_defaults(func=cmd_validate)

    p_st = sub.add_parser("stats", help="统计已标数量及与 AI 一致/分歧")
    p_st.add_argument("--bundle", type=Path, required=True)
    p_st.set_defaults(func=cmd_stats)

    p_sp = sub.add_parser("split", help="拆成 sentences / ai / human 三个 JSON")
    p_sp.add_argument("--bundle", type=Path, required=True)
    p_sp.add_argument("--out-dir", type=Path, required=True)
    p_sp.set_defaults(func=cmd_split)

    p_mg = sub.add_parser("merge", help="把 human_evaluations.json 合并回总包")
    p_mg.add_argument("--bundle", type=Path, required=True)
    p_mg.add_argument("--human", type=Path, required=True)
    p_mg.add_argument("--out", type=Path, default=None, help="默认覆盖 --bundle")
    p_mg.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
