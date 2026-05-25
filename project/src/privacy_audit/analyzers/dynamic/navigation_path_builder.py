"""Policy navigation path evidence: token extraction and sidebar insertion rules."""

from __future__ import annotations

import re
from typing import Any

SIDEBAR_TOKEN = "侧边栏"

# UI / policy alias for the same drawer or overflow menu control.
SIDEBAR_ALIASES: frozenset[str] = frozenset({SIDEBAR_TOKEN, "更多"})

# Visible labels to tap when executing a sidebar step (first match wins).
SIDEBAR_TAP_LABELS: tuple[str, ...] = ("更多", "菜单", SIDEBAR_TOKEN)

# Policy wording that once drove proactive sidebar injection (kept for hint scoring only).
# Corner markers like 左上角/右上角 are ambiguous; sidebar is opened at runtime on tap miss.
SIDEBAR_TRIGGER_MARKERS: tuple[str, ...] = ("左上角", "右上角", "界面内", "更多")
SIDEBAR_RUNTIME_REASON = "runtime_fallback_only"

_ME_TAB_NAMES = frozenset({"我", "我的", "个人", "主页", "用户", "个人主页", "用户主页"})


def hint_requires_sidebar(hint: str) -> bool:
    return any(marker in hint for marker in SIDEBAR_TRIGGER_MARKERS)


def marker_in_hint(hint: str) -> str:
    for marker in SIDEBAR_TRIGGER_MARKERS:
        if marker in hint:
            return marker
    return ""


def is_me_tab_token(token: str) -> bool:
    text = token.strip().replace("「", "").replace("」", "")
    return text in _ME_TAB_NAMES


def _normalize_token(token: str) -> str:
    return token.strip().replace("「", "").replace("」", "")


def is_sidebar_token(token: str) -> bool:
    return _normalize_token(token) in SIDEBAR_ALIASES


def path_has_sidebar_step(path: list[str]) -> bool:
    return any(is_sidebar_token(tok) for tok in path)


def canonicalize_sidebar_aliases(path: list[str]) -> list[str]:
    """Normalize 更多 → 侧边栏 in path evidence for a single canonical step name."""
    canonical: list[str] = []
    for tok in path:
        if _normalize_token(tok) == "更多":
            if not canonical or canonical[-1] != SIDEBAR_TOKEN:
                canonical.append(SIDEBAR_TOKEN)
            continue
        canonical.append(tok)
    return canonical


def _hint_has_marker_between(hint: str, before_tok: str, after_tok: str) -> bool:
    """True when hint places a corner/inline marker between two path tokens."""
    if not hint or not before_tok or not after_tok:
        return False
    bt = re.escape(before_tok)
    at = re.escape(after_tok)
    for marker in SIDEBAR_TRIGGER_MARKERS:
        if marker not in hint:
            continue
        pattern = rf"{bt}.{{0,120}}{re.escape(marker)}.{{0,120}}{at}"
        if re.search(pattern, hint, flags=re.DOTALL):
            return True
    return False


def _path_has_me_then_settings(path: list[str]) -> bool:
    for idx in range(1, len(path)):
        if path[idx] == "设置" and is_me_tab_token(path[idx - 1]):
            return True
    return False


def find_sidebar_insert_index(path: list[str], source_hint: str = "") -> int | None:
    """
    Index at which to insert 侧边栏 (insert before path[index]).

    Prefers hint marker between tokens; falls back to 我/我的 -> 设置.
    """
    if not path or path_has_sidebar_step(path):
        return None

    if source_hint:
        for idx in range(1, len(path)):
            if _hint_has_marker_between(source_hint, path[idx - 1], path[idx]):
                return idx

    for idx in range(1, len(path)):
        if path[idx] == "设置" and is_me_tab_token(path[idx - 1]):
            return idx

    if source_hint and hint_requires_sidebar(source_hint):
        for idx, tok in enumerate(path):
            if tok == "设置":
                return idx

    return None


def strip_sidebar_steps(path: list[str]) -> list[str]:
    """Remove 侧边栏/更多 steps from a path (legacy bundles / dedupe)."""
    return [tok for tok in path if not is_sidebar_token(tok)]


def path_dedupe_key(path: list[str]) -> tuple[str, ...]:
    """Dedupe keys ignore optional sidebar steps so A→B and A→侧边栏→B collapse."""
    return tuple(strip_sidebar_steps(path))


def inject_sidebar_into_path(
    path: list[str],
    source_hint: str = "",
) -> tuple[list[str], dict[str, Any]]:
    """
  Record policy path evidence without inserting 侧边栏.

  Sidebar / drawer is opened only at Appium runtime when a step misses or times out.
  """
    record: dict[str, Any] = {
        "source_hint": source_hint[:200] if source_hint else "",
        "original_path": list(path),
        "sidebar_injected": False,
        "sidebar_reason": SIDEBAR_RUNTIME_REASON,
        "sidebar_insert_index": None,
        "policy_corner_marker": marker_in_hint(source_hint) if source_hint else "",
    }
    if not path:
        return path, record

    # Keep 更多 in path if policy quoted it; do not rewrite to 侧边栏 for execution.
    cleaned = strip_sidebar_steps(path)
    record["canonical_path"] = list(cleaned)
    record["would_have_inserted_at"] = find_sidebar_insert_index(cleaned, source_hint)
    return cleaned, record


def build_navigation_path_evidence(
    raw_paths: list[list[str]],
    policy_hints: list[str],
    max_paths: int = 2,
) -> dict[str, Any]:
    """
    Build navigation_paths and per-path evidence records for dynamic_evidence.json.

    Each raw path is paired with the first matching hint (by shared tokens) or the first hint.
    """
    records: list[dict[str, Any]] = []
    expanded_paths: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    for path_idx, raw in enumerate(raw_paths[:max_paths]):
        hint = _match_hint_for_path(raw, policy_hints)
        expanded, record = inject_sidebar_into_path(raw, hint)
        record["path_index"] = path_idx + 1
        record["matched_hint"] = hint[:200] if hint else ""
        record["expanded_path"] = expanded
        key = path_dedupe_key(expanded)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
        expanded_paths.append(expanded)

    return {
        "navigation_paths": expanded_paths,
        "navigation_path_records": records,
    }


def _match_hint_for_path(path: list[str], policy_hints: list[str]) -> str:
    if not policy_hints:
        return ""
    best_hint = policy_hints[0]
    best_score = -1
    for hint in policy_hints:
        score = sum(1 for tok in path if tok and tok in hint)
        if score > best_score:
            best_score = score
            best_hint = hint
    return best_hint
