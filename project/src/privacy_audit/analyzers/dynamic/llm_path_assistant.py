"""LLM-assisted navigation: suggest intermediate menus from on-screen labels.

Uses DeepSeek (OpenAI-compatible API). Does NOT produce pass/fail verdicts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any

from privacy_audit.utils.ui_hierarchy import UiElement, parse_page_source

logger = logging.getLogger("privacy_audit.llm_path")

DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"

_SYSTEM_PROMPT = (
    "You assist Android Appium UI navigation for privacy-policy compliance checks. "
    "The policy names a settings entry that is NOT visible on the current screen. "
    "Pick ONE label from the provided on-screen list that is most likely an "
    "intermediate submenu leading to the target. "
    "The target may appear under different polarity (policy 不X vs switch label X). "
    "Return JSON only: "
    '{"bridge":"<label copied exactly from on-screen list or null>","reason":"..."}'
)


def deepseek_api_config() -> dict[str, str]:
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    return {
        "api_key": api_key,
        "base_url": os.environ.get("DEEPSEEK_API_BASE", DEFAULT_DEEPSEEK_BASE).strip()
        or DEFAULT_DEEPSEEK_BASE,
        "model": os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL).strip()
        or DEFAULT_DEEPSEEK_MODEL,
    }


def llm_nav_available() -> bool:
    return bool(deepseek_api_config()["api_key"])


def extract_visible_nav_labels(
    page_source: str,
    *,
    max_labels: int = 80,
) -> list[str]:
    """Collect distinct on-screen text/content-desc labels for LLM context."""
    elements = parse_page_source(page_source)
    if not elements:
        return []

    ranked: list[tuple[int, str]] = []
    seen: set[str] = set()
    for el in elements:
        if not el.displayed:
            continue
        label = _element_nav_label(el)
        if not label or label in seen:
            continue
        seen.add(label)
        priority = 0
        if el.clickable:
            priority += 2
        if el.enabled:
            priority += 1
        ranked.append((priority, label))

    ranked.sort(key=lambda item: (-item[0], len(item[1])))
    return [label for _, label in ranked[:max_labels]]


def _element_nav_label(el: UiElement) -> str:
    for raw in (el.text, el.content_desc):
        text = (raw or "").strip()
        if not text or len(text) > 40:
            continue
        if _looks_like_noise_label(text):
            continue
        return text
    return ""


def _looks_like_noise_label(text: str) -> bool:
    if text.isdigit():
        return True
    if re.fullmatch(r"[\d:./\\-]+", text):
        return True
    if len(text) == 1 and not ("\u4e00" <= text <= "\u9fff"):
        return True
    return False


def _deepseek_chat_json(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    model: str,
    base_url: str,
    timeout_sec: float = 20.0,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return json.loads(content)
    if isinstance(content, dict):
        return content
    raise ValueError("Unexpected LLM response content")


def _resolve_bridge_label(
    bridge: str,
    visible_labels: list[str],
) -> str | None:
    bridge = bridge.strip()
    if not bridge:
        return None
    if bridge in visible_labels:
        return bridge
    for label in visible_labels:
        if bridge in label or label in bridge:
            return label
    return None


def suggest_intermediate_bridge(
    target_token: str,
    target_aliases: list[str],
    path_prefix: list[str],
    visible_labels: list[str],
    *,
    policy_hint: str = "",
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_sec: float = 20.0,
) -> dict[str, Any]:
    """
    Ask DeepSeek which on-screen label is the best intermediate menu for target_token.

    Returns {"bridge": str|None, "reason": str, "error": str|None}.
    """
    cfg = deepseek_api_config()
    key = cfg["api_key"] if api_key is None else api_key.strip()
    if not key:
        return {
            "bridge": None,
            "reason": "",
            "error": "DEEPSEEK_API_KEY not set",
        }
    if not visible_labels:
        return {
            "bridge": None,
            "reason": "",
            "error": "no visible labels",
        }

    user_prompt = {
        "target_token": target_token,
        "target_aliases": target_aliases[:12],
        "path_so_far": path_prefix[-6:],
        "visible_labels": visible_labels,
        "policy_hint": policy_hint[:500],
    }
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(user_prompt, ensure_ascii=False),
        },
    ]
    try:
        parsed = _deepseek_chat_json(
            messages,
            api_key=key,
            model=model or cfg["model"],
            base_url=base_url or cfg["base_url"],
            timeout_sec=timeout_sec,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("DeepSeek bridge suggestion failed: %s", exc)
        return {"bridge": None, "reason": "", "error": str(exc)}

    raw_bridge = parsed.get("bridge")
    reason = str(parsed.get("reason", "") or "")
    if raw_bridge in (None, "", "null"):
        return {"bridge": None, "reason": reason, "error": None}
    resolved = _resolve_bridge_label(str(raw_bridge), visible_labels)
    if not resolved:
        return {
            "bridge": None,
            "reason": reason,
            "error": f"bridge not on screen: {raw_bridge}",
        }
    return {"bridge": resolved, "reason": reason, "error": None}


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
