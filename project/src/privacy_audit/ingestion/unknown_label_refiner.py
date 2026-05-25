"""Cluster unknown labels and suggest mapping expansions."""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

from privacy_audit.ingestion.legacy_bridge import load_legacy_label_mapping


def _normalize_head(label: str) -> str:
    return re.split(r"[；;。]", label, maxsplit=1)[0].strip()


def _keyword_rules() -> list[tuple[list[str], str, float]]:
    return [
        (["手机号", "手机号码", "短信验证码", "登录", "账号"], "phone_number", 0.95),
        (["实名", "身份", "身份证"], "identity_info", 0.95),
        (["未成年人", "儿童"], "minor_info", 0.95),
        (["生物", "指纹", "人脸"], "biometric_info", 0.95),
        (["位置", "定位", "gps", "地理"], "location", 0.95),
        (["ip"], "ip_address", 0.9),
        (["mac"], "mac_address", 0.9),
        (["通讯录", "联系人"], "contact_list", 0.95),
        (["剪贴板", "复制", "粘贴"], "clipboard", 0.9),
        (["日志"], "log_info", 0.9),
        (["权限"], "permission_info", 0.9),
        (["sdk", "第三方", "技术组件", "合作方"], "sdk_info", 0.85),
        (["个性化", "广告", "推荐", "画像", "偏好"], "profile_info", 0.85),
        (["安全", "投诉", "响应", "处理时限"], "security_info", 0.8),
        (["查阅", "更正", "删除", "权利", "导出"], "rights_info", 0.8),
        (["用户信息", "个人信息", "收集"], "account_info", 0.75),
    ]


def infer_pii_type(label: str) -> tuple[str | None, float]:
    """Infer pii_type from label text and return confidence."""
    text = label.lower()
    best_type: str | None = None
    best_score = 0.0
    for keys, pii_type, score in _keyword_rules():
        if any(key in text for key in keys):
            if score > best_score:
                best_type = pii_type
                best_score = score
    return best_type, best_score


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    compact = "".join(ch for ch in text if not ch.isspace())
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def _text_similarity(a: str, b: str) -> float:
    """Hybrid similarity for Chinese policy labels."""
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    a2 = _char_ngrams(a, n=2)
    b2 = _char_ngrams(b, n=2)
    inter = len(a2 & b2)
    union = len(a2 | b2) or 1
    jaccard = inter / union
    a1 = set(a)
    b1 = set(b)
    char_overlap = len(a1 & b1) / (len(a1 | b1) or 1)
    return 0.55 * seq + 0.35 * jaccard + 0.10 * char_overlap


def _nearest_neighbor(
    label: str,
    mapping: dict[str, list[str]],
) -> tuple[str | None, float, str]:
    best_type: str | None = None
    best_score = 0.0
    best_label = ""
    for pii_type, labels in mapping.items():
        for known_label in labels:
            score = _text_similarity(label, known_label)
            if score > best_score:
                best_score = score
                best_type = pii_type
                best_label = known_label
    return best_type, best_score, best_label


def cluster_unknown_labels(unknown_labels: dict[str, int]) -> list[dict[str, Any]]:
    """Cluster unknown labels by normalized head and inferred pii_type."""
    buckets: dict[str, dict[str, Any]] = {}
    for raw_label, count in unknown_labels.items():
        head = _normalize_head(raw_label)
        inferred_type, conf = infer_pii_type(head)
        key = f"{inferred_type or 'unmapped'}::{head}"
        if key not in buckets:
            buckets[key] = {
                "cluster_head": head,
                "suggested_pii_type": inferred_type,
                "confidence": conf,
                "total_count": 0,
                "labels": [],
            }
        buckets[key]["total_count"] += int(count)
        buckets[key]["labels"].append({"label": raw_label, "count": count})

    clusters = sorted(
        buckets.values(),
        key=lambda item: (item["total_count"], item["confidence"]),
        reverse=True,
    )
    return clusters


def _dump_mapping_yaml(mapping: dict[str, list[str]], path: Path) -> None:
    lines: list[str] = [
        "# Map legacy human review labels to normalized pii_type.",
        "# Used in stage0 -> stage1 bridge importer.",
        "",
    ]
    for pii_type in sorted(mapping.keys()):
        labels = sorted(set(mapping[pii_type]))
        safe = [f"\"{label.replace('\"', '\\\"')}\"" for label in labels]
        lines.append(f"{pii_type}:")
        lines.append(f"  labels: [{', '.join(safe)}]")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def refine_unknown_labels(
    bridge_output_file: Path,
    mapping_file: Path,
    clusters_output_file: Path,
    suggestions_output_file: Path,
    min_confidence: float = 0.8,
    use_similarity: bool = True,
    similarity_threshold: float = 0.52,
) -> dict[str, Any]:
    """Generate clusters/suggestions and semi-auto expand mapping file."""
    bundle = json.loads(bridge_output_file.read_text(encoding="utf-8"))
    unknown_labels = bundle.get("summary", {}).get("unknown_labels", {})
    clusters = cluster_unknown_labels(unknown_labels)
    mapping = load_legacy_label_mapping(mapping_file)

    suggestions: dict[str, list[str]] = {}
    for cluster in clusters:
        pii_type = cluster.get("suggested_pii_type")
        confidence = float(cluster.get("confidence", 0.0))
        neighbor_label = ""
        if (not pii_type or confidence < min_confidence) and use_similarity:
            nn_type, nn_score, nn_label = _nearest_neighbor(
                str(cluster.get("cluster_head", "")),
                mapping,
            )
            if nn_type and nn_score >= similarity_threshold:
                pii_type = nn_type
                confidence = nn_score
                neighbor_label = nn_label
        if not pii_type or confidence < min_confidence:
            continue
        head = str(cluster.get("cluster_head", "")).strip()
        if not head:
            continue
        suggestions.setdefault(pii_type, []).append(head)
        if neighbor_label:
            cluster["nearest_neighbor"] = {
                "label": neighbor_label,
                "score": round(confidence, 4),
            }

    added: dict[str, list[str]] = {}
    for pii_type, labels in suggestions.items():
        bucket = mapping.setdefault(pii_type, [])
        for label in labels:
            if label not in bucket:
                bucket.append(label)
                added.setdefault(pii_type, []).append(label)

    _dump_mapping_yaml(mapping, mapping_file)
    clusters_output_file.parent.mkdir(parents=True, exist_ok=True)
    suggestions_output_file.parent.mkdir(parents=True, exist_ok=True)

    clusters_output_file.write_text(
        json.dumps({"clusters": clusters}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    suggestions_output_file.write_text(
        json.dumps(
            {
                "min_confidence": min_confidence,
                "use_similarity": use_similarity,
                "similarity_threshold": similarity_threshold,
                "suggestions": suggestions,
                "applied_to_mapping": added,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "unknown_count": len(unknown_labels),
        "cluster_count": len(clusters),
        "suggested_types": len(suggestions),
        "applied_labels": sum(len(v) for v in added.values()),
        "use_similarity": use_similarity,
        "similarity_threshold": similarity_threshold,
        "clusters_output_file": str(clusters_output_file),
        "suggestions_output_file": str(suggestions_output_file),
        "mapping_file": str(mapping_file),
    }

