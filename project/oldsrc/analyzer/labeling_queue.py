from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def joint_labeling_score(row: Dict[str, Any]) -> int:
    """
    Higher = higher priority for human labeling.
    Combines pii_related (model) and keyword_hint (rule): both True ranks highest.
    Encoding: pii True=2, False=1, null=0; joint = 2*pii_score + kh (0/1).
    """
    pr = row.get("pii_related")
    if pr is True:
        ps = 2
    elif pr is False:
        ps = 1
    else:
        ps = 0
    kh = 1 if row.get("keyword_hint") else 0
    return ps * 2 + kh


def _confidence_uncertainty(row: Dict[str, Any]) -> float:
    """Lower confidence => more valuable to label (tiebreaker)."""
    c = row.get("confidence")
    if c is None:
        return 0.5
    try:
        return float(c)
    except (TypeError, ValueError):
        return 0.5


def labeling_sort_key(row: Dict[str, Any]) -> Tuple:
    """
    Sort ascending: first rows are highest priority for 送标.
    Primary: joint_labeling_score descending (via negation).
    Secondary: lower model confidence first (uncertain cases).
    Stable: doc_id, sent_index.
    """
    return (
        -joint_labeling_score(row),
        _confidence_uncertainty(row),
        str(row.get("doc_id", "")),
        int(row.get("sent_index", 0)),
    )


def sort_rows_for_labeling(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=labeling_sort_key)


def labeling_export_rows(
    rows: List[Dict[str, Any]],
    *,
    limit: Optional[int] = None,
    include_meta: bool = True,
) -> List[Dict[str, Any]]:
    """Same ordering as for_labeling.jsonl; optional top-N slice."""
    ordered = sort_rows_for_labeling(rows)
    if limit is not None and limit > 0:
        ordered = ordered[:limit]
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(ordered):
        base = {k: v for k, v in r.items() if not str(k).startswith("_")}
        if include_meta:
            base["labeling_queue_rank"] = i
            base["labeling_priority_score"] = joint_labeling_score(r)
        out.append(base)
    return out


def write_labeling_jsonl(
    rows: List[Dict[str, Any]],
    path: Path,
    *,
    limit: Optional[int] = None,
    include_meta: bool = True,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    export_rows = labeling_export_rows(rows, limit=limit, include_meta=include_meta)
    with path.open("w", encoding="utf-8") as f:
        for base in export_rows:
            f.write(json.dumps(base, ensure_ascii=False) + "\n")
    return len(export_rows)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sort sentences.jsonl by pii_related + keyword_hint for labeling export.",
    )
    parser.add_argument("--input", type=Path, required=True, help="sentences.jsonl or all_sentences.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="e.g. for_labeling.jsonl")
    parser.add_argument("--top-n", type=int, default=0, help="Write only first N rows after sort (0=all)")
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    lim = args.top_n if args.top_n and args.top_n > 0 else None
    n = write_labeling_jsonl(rows, args.output, limit=lim, include_meta=True)
    print(
        json.dumps(
            {"wrote": str(args.output), "rows": n, "input_rows": len(rows)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
