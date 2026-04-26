from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from keyword_hint import keyword_hint
from llm_related import classify_sentence
from sentence_split import split_policy_text


def build_rows_for_text(
    raw: str,
    doc_id: str,
    *,
    mode: str,
    provider: str,
    ollama_model: str,
    ollama_base_url: str,
    deepseek_api_key: str,
    deepseek_model: str,
    deepseek_base_url: str,
    timeout_sec: int,
    limit: Optional[int],
    max_chars: int,
    pii_shots: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    sentences = split_policy_text(raw, max_chars=max_chars)
    rows: List[Dict[str, Any]] = []
    num_classified = 0

    for idx, sent in enumerate(sentences):
        row: Dict[str, Any] = {
            "doc_id": doc_id,
            "sent_index": idx,
            "text": sent,
            "pii_related": None,
            "confidence": None,
            "keyword_hint": bool(keyword_hint(sent)),
            "raw_model_output": "",
        }

        if mode == "classify":
            if limit is not None and idx >= limit:
                rows.append(row)
                continue
            res = classify_sentence(
                sent,
                provider=provider,
                ollama_model=ollama_model,
                ollama_base_url=ollama_base_url,
                deepseek_api_key=deepseek_api_key,
                deepseek_model=deepseek_model,
                deepseek_base_url=deepseek_base_url,
                timeout_sec=timeout_sec,
                pii_shots=pii_shots,
            )
            row["pii_related"] = res.related
            row["confidence"] = res.confidence
            row["raw_model_output"] = res.raw
            num_classified += 1

        rows.append(row)

    return rows, num_classified
