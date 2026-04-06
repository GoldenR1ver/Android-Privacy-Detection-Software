from __future__ import annotations

import re
from typing import List


# Chinese + common English end punctuation; keep enumerated clauses together when possible.
_SPLIT_PATTERN = re.compile(
    r"(?<=[。！？；\n])|(?<=[.!?])\s+"
)


def split_policy_text(text: str, max_chars: int = 0) -> List[str]:
    """
    Split privacy policy text into sentence-like units for annotation.
    max_chars: if > 0, merge adjacent short fragments so no unit exceeds this length
    (best-effort; does not break Chinese characters).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    parts = [p.strip() for p in _SPLIT_PATTERN.split(text) if p and p.strip()]
    units: List[str] = list(parts)
    if max_chars > 0:
        merged: List[str] = []
        buf = ""
        for p in parts:
            if not buf:
                buf = p
                continue
            if len(buf) + len(p) <= max_chars:
                buf = f"{buf}{p}"
            else:
                merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)
        units = merged

    out: List[str] = []
    for s in units:
        for line in s.split("\n"):
            line = line.strip()
            if line:
                out.append(line)
    return out
