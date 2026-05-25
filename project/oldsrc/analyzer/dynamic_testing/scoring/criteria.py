"""Usability scoring criteria and decision tables.

Four dimensions, each scored 1-5:

  Findability    — how many taps/scrolls to locate the feature
  Functionality  — does the feature work as privacy policy describes
  Efficiency     — how many steps to complete the action
  Consistency    — do UI labels match policy path_text
"""

from __future__ import annotations


# ── Findability scoring (可寻性) ──────────────────────────────────────────

def score_findability(
    found: bool,
    taps_to_reach: int = 99,
    needed_scroll: bool = False,
) -> int:
    """Score findability from 1-5.

    5: Entry on first screen (≤1 tap, no scroll)
    4: Found within 1 scroll + 1 tap
    3: Found within 2 scrolls + 2 taps
    2: Found in deep menu (3+ taps) or via search
    1: Not found at all
    """
    if not found:
        return 1
    if taps_to_reach <= 1 and not needed_scroll:
        return 5
    if taps_to_reach <= 2 and needed_scroll:
        return 4
    if taps_to_reach <= 3:
        return 3
    if taps_to_reach <= 5:
        return 2
    return 1


# ── Functionality scoring (功能性) ────────────────────────────────────────

def score_functionality(
    feature_works: bool,
    partial_work: bool = False,
    crashes: bool = False,
) -> int:
    """Score functionality from 1-5.

    5: Feature works fully as policy describes
    4: Feature works with minor UI differences
    3: Feature partially works (some options missing)
    2: Feature exists but non-functional (button does nothing)
    1: Feature does not exist or crashes
    """
    if crashes:
        return 1
    if feature_works:
        return 5
    if partial_work:
        return 3
    return 1 if not feature_works else 2


# ── Efficiency scoring (效率) ─────────────────────────────────────────────

def score_efficiency(
    tap_count: int,
    has_redirect: bool = False,
    has_external: bool = False,
) -> int:
    """Score efficiency from 1-5.

    5: 1-2 steps from main entry
    4: 3-4 steps
    3: 5-7 steps or involves switching to web
    2: 8+ steps or requires external communication
    1: Impossible via app UI
    """
    if has_external:
        return 2
    if has_redirect:
        return 3
    if tap_count <= 2:
        return 5
    if tap_count <= 4:
        return 4
    if tap_count <= 7:
        return 3
    if tap_count <= 10:
        return 2
    return 1


# ── Consistency scoring (一致性) ──────────────────────────────────────────

def score_consistency(
    policy_path: str,
    actual_path: str,
    labels_match: bool = False,
    partial_match: bool = False,
) -> int:
    """Score consistency from 1-5.

    5: UI path labels match policy path_text exactly
    4: Labels differ slightly but same meaning
    3: Different labels but same page reached
    2: Different page reached than policy described
    1: Path cannot be mapped to UI at all
    """
    if labels_match:
        return 5
    if partial_match:
        return 3
    # Default: moderate consistency if the path_text was parseable
    if policy_path and len(policy_path) > 5:
        return 3
    return 1


# ── Composite score ───────────────────────────────────────────────────────

def composite_score(
    findability: int,
    functionality: int,
    efficiency: int,
    consistency: int,
    weights: dict | None = None,
) -> float:
    """Compute weighted composite usability score, normalized to 0-100."""
    if weights is None:
        from ..config import SCORE_WEIGHTS as weights

    weighted = (
        findability * weights["findability"]
        + functionality * weights["functionality"]
        + efficiency * weights["efficiency"]
        + consistency * weights["consistency"]
    )
    return round(weighted / 5.0 * 100.0, 1)
