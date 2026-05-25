"""Stage0 -> Stage1 bridge for legacy outputs."""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any


def _parse_inline_list(value: str) -> list[str]:
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return []
    body = value[1:-1].strip()
    if not body:
        return []
    return [token.strip().strip("\"'") for token in body.split(",") if token.strip()]


def load_legacy_label_mapping(mapping_file: Path) -> dict[str, list[str]]:
    """Load mapping file: pii_type -> labels list."""
    lines = mapping_file.read_text(encoding="utf-8").splitlines()
    mapping: dict[str, list[str]] = {}
    current_key: str | None = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not raw.startswith(" ") and line.endswith(":"):
            current_key = line[:-1].strip()
            mapping[current_key] = []
            continue
        if current_key is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() == "labels":
            mapping[current_key] = _parse_inline_list(value.strip())
    return mapping


def build_label_to_pii_map(mapping: dict[str, list[str]]) -> dict[str, str]:
    """Flatten mapping into legacy_label -> pii_type."""
    flat: dict[str, str] = {}
    for pii_type, labels in mapping.items():
        for label in labels:
            flat[label] = pii_type
    return flat


def _normalize_label_head(label: str) -> str:
    return re.split(r"[；;。]", label, maxsplit=1)[0].strip()


def _lookup_mapped_pii(label: str, label_to_pii: dict[str, str]) -> str | None:
    if label in label_to_pii:
        return label_to_pii[label]
    head = _normalize_label_head(label)
    if head in label_to_pii:
        return label_to_pii[head]
    for mapped_label, pii_type in label_to_pii.items():
        if mapped_label and label.startswith(mapped_label):
            return pii_type
    return None


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    compact = "".join(ch for ch in text if not ch.isspace())
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def _similarity(a: str, b: str) -> float:
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    a2 = _char_ngrams(a, n=2)
    b2 = _char_ngrams(b, n=2)
    inter = len(a2 & b2)
    union = len(a2 | b2) or 1
    jaccard = inter / union
    return 0.65 * seq + 0.35 * jaccard


def _nearest_mapped_pii(
    label: str,
    label_to_pii: dict[str, str],
    threshold: float = 0.30,
) -> tuple[str | None, float, str]:
    head = _normalize_label_head(label)
    best_type: str | None = None
    best_score = 0.0
    best_label = ""
    for mapped_label, pii_type in label_to_pii.items():
        score = _similarity(head, mapped_label)
        if score > best_score:
            best_score = score
            best_type = pii_type
            best_label = mapped_label
    if best_type and best_score >= threshold:
        return best_type, best_score, best_label
    return None, best_score, best_label


def _infer_pii_from_label_text(label: str) -> str | None:
    text = label.lower()
    keyword_rules: list[tuple[list[str], str]] = [
        (["imei", "android id", "oaid", "设备"], "imei"),
        (["位置", "定位", "gps", "经纬度"], "location"),
        (["ip"], "ip_address"),
        (["mac"], "mac_address"),
        (["通讯录", "联系人"], "contact_list"),
        (["剪贴板", "复制", "粘贴"], "clipboard"),
        (["手机号", "手机号码", "短信验证码", "账号", "登录"], "phone_number"),
        (["实名", "身份", "身份证"], "identity_info"),
        (["未成年人", "儿童"], "minor_info"),
    ]
    for keys, pii_type in keyword_rules:
        if any(key in text for key in keys):
            return pii_type
    return None


def import_claims_from_review_bundle(
    review_bundle_file: Path,
    mapping_file: Path,
    output_file: Path | None = None,
    nn_threshold: float = 0.30,
) -> dict[str, Any]:
    """Convert legacy human-labeled bundle into normalized claim records."""
    bundle = json.loads(review_bundle_file.read_text(encoding="utf-8"))
    mapping = load_legacy_label_mapping(mapping_file)
    label_to_pii = build_label_to_pii_map(mapping)

    claims: list[dict[str, Any]] = []
    unknown_labels: dict[str, int] = {}
    skipped_non_pii = 0
    mapped_by_counts: dict[str, int] = {}
    for item in bundle.get("items", []):
        human = item.get("human", {})
        if not bool(human.get("pii_related", False)):
            continue
        label = str(human.get("notes", "")).strip()
        mapped = _lookup_mapped_pii(label, label_to_pii)
        mapped_by = "direct_or_prefix"
        nn_score = 0.0
        nn_label = ""
        if not mapped:
            mapped = _infer_pii_from_label_text(label)
            mapped_by = "keyword_infer"
        if not mapped:
            mapped, nn_score, nn_label = _nearest_mapped_pii(
                label,
                label_to_pii,
                threshold=nn_threshold,
            )
            if mapped:
                mapped_by = "nearest_neighbor"
        if mapped == "skip_non_pii":
            skipped_non_pii += 1
            continue
        if not mapped:
            unknown_labels[label] = unknown_labels.get(label, 0) + 1
            continue
        claims.append(
            {
                "claim_id": item.get("id", ""),
                "app_name": item.get("doc_id", ""),
                "right_type": "informed",
                "pii_type": mapped,
                "declared": True,
                "claim_text": item.get("text", ""),
                "policy_doc_id": item.get("doc_id", ""),
                "sentence_index": item.get("sent_index"),
                "confidence": 1.0,
                "metadata": {
                    "legacy_label": label,
                    "source": "review_bundle.human",
                    "mapped_by": mapped_by,
                    "nearest_label": nn_label,
                    "nearest_score": round(nn_score, 4),
                },
            }
        )
        mapped_by_counts[mapped_by] = mapped_by_counts.get(mapped_by, 0) + 1

    result = {
        "schema_version": 1,
        "source": str(review_bundle_file),
        "nn_threshold": nn_threshold,
        "claims": claims,
        "summary": {
            "total_items": len(bundle.get("items", [])),
            "imported_claims": len(claims),
            "skipped_non_pii": skipped_non_pii,
            "mapped_by_counts": mapped_by_counts,
            "unknown_labels": unknown_labels,
        },
    }
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result

