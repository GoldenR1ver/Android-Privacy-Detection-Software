"""LLM verifier replacement: path suggester and explainer.

This module does NOT produce pass/fail verdicts.
It only provides:
- candidate path suggestions
- natural-language explanation from collected evidence
"""

from __future__ import annotations

from typing import Any


def build_path_suggestions(
    policy_text: str,
    candidate_pii: list[str],
    ui_steps: list[str],
) -> list[str]:
    """Build deterministic path suggestions from policy text and ui traces."""
    suggestions: list[str] = []
    lines = [line.strip() for line in policy_text.splitlines() if line.strip()]
    for line in lines:
        if len(suggestions) >= 8:
            break
        if "设置" in line and ("隐私" in line or "账号" in line or "权限" in line):
            suggestions.append(line)
            continue
        if "->" in line or "→" in line:
            suggestions.append(line)
    if ui_steps:
        suggestions.append(f"UI observed steps: {' | '.join(ui_steps[:3])}")
    for pii_type in candidate_pii[:3]:
        suggestions.append(f"Check path for pii_type={pii_type}")
    # Keep stable order and unique entries.
    uniq: list[str] = []
    for text in suggestions:
        if text not in uniq:
            uniq.append(text)
    return uniq[:10]


def build_evidence_explanation(
    app_name: str,
    candidate_pii: list[str],
    ui_result: dict[str, Any],
    frida_result: dict[str, Any],
) -> str:
    """Generate explanation text from dynamic evidence collection."""
    ui_steps = len(ui_result.get("steps", []))
    screenshots = len(ui_result.get("screenshots", []))
    frida_events = len(frida_result.get("events", []))
    registered = len(frida_result.get("registered_hooks", {}))
    return (
        f"Collected dynamic evidence for {app_name}: "
        f"candidate_pii={len(candidate_pii)}, ui_steps={ui_steps}, "
        f"screenshots={screenshots}, frida_registered_types={registered}, "
        f"frida_events={frida_events}. "
        "This explanation is descriptive only and does not produce verdict."
    )

