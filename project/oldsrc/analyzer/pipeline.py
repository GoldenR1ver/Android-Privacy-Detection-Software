from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    log_callback: Optional[Callable[[str], None]] = None,
    log_every: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    sentences = split_policy_text(raw, max_chars=max_chars)
    if log_callback:
        log_callback(
            f"[prepare] split doc={doc_id!r}: raw_chars={len(raw)}, "
            f"sentences={len(sentences)}, mode={mode}, limit={limit or 'all'}"
        )
    rows: List[Dict[str, Any]] = []
    num_classified = 0
    log_every = max(0, int(log_every or 0))

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
                if log_callback and idx == limit:
                    log_callback(f"[prepare] classify limit reached at sentence_index={idx}; remaining rows split only")
                rows.append(row)
                continue
            should_log = bool(log_callback) and (log_every <= 1 or idx == 0 or (idx + 1) % log_every == 0)
            t0 = time.perf_counter()
            if should_log and log_callback:
                preview = sent.replace("\n", " ")[:80]
                log_callback(
                    f"[prepare] classify request doc={doc_id!r} "
                    f"sentence={idx + 1}/{len(sentences)} chars={len(sent)} text={preview!r}"
                )
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
            elapsed = time.perf_counter() - t0
            row["pii_related"] = res.related
            row["confidence"] = res.confidence
            row["raw_model_output"] = res.raw
            num_classified += 1
            if should_log and log_callback:
                raw_preview = (res.raw or "").replace("\n", " ")[:120]
                log_callback(
                    f"[prepare] classify done doc={doc_id!r} "
                    f"sentence={idx + 1}/{len(sentences)} elapsed={elapsed:.2f}s "
                    f"related={res.related} confidence={res.confidence} raw={raw_preview!r}"
                )

        rows.append(row)

    if log_callback:
        log_callback(
            f"[prepare] build rows completed doc={doc_id!r}: rows={len(rows)}, classified={num_classified}"
        )
    return rows, num_classified
