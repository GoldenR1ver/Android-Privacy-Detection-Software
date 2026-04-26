from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Literal

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


def build_audit_ds_from_cluster_peers(
    rows: List[Dict[str, Any]],
    index: int,
    *,
    max_chars: int = 12000,
    max_peers: int = 50,
    separator: str = "\n【同簇句】\n",
) -> str:
    """
    Synthetic Data Safety text: concatenate peer sentences in the same HDBSCAN cluster
    (excluding the current sentence). Privacy policy side stays the single current sentence.

    Noise points (cluster_id < 0): use a sample of other sentences from the same document
    as weak context so the model can still judge incomplete vs incorrect.
    """
    row = rows[index]
    cid_raw = row.get("cluster_id", -1)
    try:
        cid = int(cid_raw)
    except (TypeError, ValueError):
        cid = -1

    if cid < 0:
        others = [
            str(r.get("text", "")).strip()
            for i, r in enumerate(rows)
            if i != index and str(r.get("text", "")).strip()
        ]
        sample = others[: min(20, len(others))]
        body = separator.join(sample) if sample else "（同文档无其它句）"
        header = (
            "【Data Safety 侧说明】本句在句向量聚类中为噪声点（未归入稠密簇）。"
            "下列为同文档中抽取的若干其它句子，仅作弱参照，不等价于应用商店 Data Safety 官方字段。\n\n"
        )
        return (header + body)[:max_chars]

    peers = [
        str(r.get("text", "")).strip()
        for i, r in enumerate(rows)
        if i != index and int(r.get("cluster_id", -999)) == cid and str(r.get("text", "")).strip()
    ]
    if not peers:
        return (
            "【Data Safety 侧说明】同簇除当前句外无其它同伴句；簇内披露视为极简。"
        )[:max_chars]

    parts: List[str] = []
    used = 0
    for t in peers[:max_peers]:
        step = len(t) + len(separator)
        if used + step > max_chars:
            break
        parts.append(t)
        used += step
    body = separator.join(parts)
    header = (
        f"【Data Safety 侧：簇内语义聚合】以下为与当前 Privacy Policy 句同属 HDBSCAN 簇 #{cid} 的"
        "其它句子汇总（不含当前句）。请将其视作「簇内数据实践摘要 / 简化披露」，"
        "与右侧「当前单句 Privacy Policy」比对，判断 incorrect / incomplete / inconsistent。\n\n"
    )
    return (header + body)[:max_chars]


def write_week3_sentence_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    *,
    app_pkg: str,
    app_name: str,
    category_id: int,
    app_id_start: int,
    ds_mode: Literal["empty", "cluster_peers"] = "empty",
    cluster_ds_max_chars: int = 12000,
    cluster_ds_max_peers: int = 50,
) -> None:
    """
    One CSV row per sentence. privacy_policy_content is the sentence text.
    app_id = app_id_start + sent_index (unique within this export).

    ds_mode:
      empty: fixed EMPTY_DATA_SAFETY_CONTENT (legacy; incomplete/inconsistent rarely fire).
      cluster_peers: rows must have been augmented with cluster_id from sentence_cluster.run_clustering.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if ds_mode == "cluster_peers":
        for i, r in enumerate(rows):
            if "cluster_id" not in r:
                raise ValueError(
                    "ds_mode=cluster_peers requires each row to contain 'cluster_id' "
                    "(run sentence_cluster.run_clustering before export)."
                )
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=WEEK3_FIELDNAMES,
            quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        for i, row in enumerate(rows):
            idx = int(row["sent_index"])
            app_id = row.get("_export_app_id", app_id_start + idx)
            if ds_mode == "empty":
                ds_val: str = EMPTY_DATA_SAFETY_CONTENT
            else:
                ds_val = build_audit_ds_from_cluster_peers(
                    rows,
                    i,
                    max_chars=cluster_ds_max_chars,
                    max_peers=cluster_ds_max_peers,
                )
            w.writerow(
                {
                    "app_id": app_id,
                    "app_pkg": row.get("export_app_pkg", app_pkg),
                    "app_name": row.get("export_app_name", app_name),
                    "category_id": row.get("export_category_id", category_id),
                    "data_safety_content": ds_val,
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
