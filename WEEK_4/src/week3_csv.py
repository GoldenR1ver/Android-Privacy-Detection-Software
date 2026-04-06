from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

# Aligns with WEEK_3/src/2-2/data/*.csv column names and typical empty Data Safety cell.
WEEK3_FIELDNAMES = [
    "app_id",
    "app_pkg",
    "app_name",
    "category_id",
    "data_safety_content",
    "privacy_policy_content",
    "result",
]

EMPTY_DATA_SAFETY_CONTENT = (
    "{'data_shared': [], 'data_collected': [], 'security_practices': []}"
)


def write_week3_sentence_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    *,
    app_pkg: str,
    app_name: str,
    category_id: int,
    app_id_start: int,
) -> None:
    """
    One CSV row per sentence. privacy_policy_content is the sentence text.
    app_id = app_id_start + sent_index (unique within this export).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=WEEK3_FIELDNAMES,
            quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        for row in rows:
            idx = int(row["sent_index"])
            app_id = row.get("_export_app_id", app_id_start + idx)
            w.writerow(
                {
                    "app_id": app_id,
                    "app_pkg": row.get("export_app_pkg", app_pkg),
                    "app_name": row.get("export_app_name", app_name),
                    "category_id": row.get("export_category_id", category_id),
                    "data_safety_content": EMPTY_DATA_SAFETY_CONTENT,
                    "privacy_policy_content": row["text"],
                    "result": row.get("result", "") or "",
                }
            )


def compute_privacy_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "total_sentences": 0,
            "pii_related": {
                "true": 0,
                "false": 0,
                "null": 0,
                "ratio_of_total": None,
                "ratio_of_labeled": None,
            },
            "keyword_hint": {
                "true": 0,
                "false": 0,
                "ratio_of_total": None,
            },
        }

    p_true = p_false = p_null = 0
    k_true = 0
    for r in rows:
        pr = r.get("pii_related")
        if pr is True:
            p_true += 1
        elif pr is False:
            p_false += 1
        else:
            p_null += 1
        if r.get("keyword_hint"):
            k_true += 1

    labeled = p_true + p_false
    ratio_labeled = (p_true / labeled) if labeled else None
    ratio_total = (p_true / total) if p_null == 0 and labeled == total else None
    if p_null > 0 and labeled > 0:
        ratio_total_partial_note = (
            "pii_related_ratio_of_total is None when any sentence is unclassified; "
            "use ratio_of_labeled or run classify on all sentences."
        )
    else:
        ratio_total_partial_note = None

    return {
        "total_sentences": total,
        "pii_related": {
            "true": p_true,
            "false": p_false,
            "null": p_null,
            "ratio_of_total": ratio_total,
            "ratio_of_labeled": ratio_labeled,
            "note": ratio_total_partial_note,
        },
        "keyword_hint": {
            "true": k_true,
            "false": total - k_true,
            "ratio_of_total": k_true / total,
        },
    }
