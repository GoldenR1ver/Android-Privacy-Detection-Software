from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


SYSTEM_PROMPT = (
    "你是中文隐私合规文本分析助手。判断给定句子是否描述或涉及"
    "「个人信息的处理与利用」（含收集、使用、存储、共享、转让、委托处理、"
    "披露、跨境传输、安全保护、用户权利响应、未成年人保护等）。"
    "若句子仅为目录、标题口号、与具体处理行为无关的法律套话，判为不相关。"
)


USER_TEMPLATE = """请只输出一个 JSON 对象，不要其它文字。字段如下：
- "related": 布尔，true 表示与个人信息处理与利用相关，否则 false
- "confidence": 0 到 1 之间的小数，表示你对该判断的把握
- "reason": 字符串，一句话说明理由（中文）

句子：
{sentence}
"""


@dataclass
class ClassifyResult:
    related: Optional[bool]
    confidence: Optional[float]
    reason: str
    raw: str


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if not m:
        m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def classify_mock(sentence: str) -> ClassifyResult:
    """Deterministic stub: related if sentence mentions 个人信息 or 收集/使用."""
    related = any(
        k in sentence
        for k in ("个人信息", "个人敏感信息", "收集", "使用", "共享", "处理")
    )
    return ClassifyResult(
        related=related,
        confidence=0.5,
        reason="mock：基于关键词规则的占位判断",
        raw=json.dumps(
            {"related": related, "confidence": 0.5, "reason": "mock"},
            ensure_ascii=False,
        ),
    )


def classify_ollama(
    sentence: str,
    model: str,
    base_url: str = "http://127.0.0.1:11434/api/chat",
    timeout_sec: int = 120,
) -> ClassifyResult:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(sentence=sentence)},
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return ClassifyResult(
            related=None,
            confidence=None,
            reason="",
            raw=f'{{"error": "{e!s}"}}',
        )

    message = (body.get("message") or {})
    content = (message.get("content") or "").strip()
    return _result_from_parsed(content, _extract_json_object(content))


def _result_from_parsed(content: str, parsed: Optional[Dict[str, Any]]) -> ClassifyResult:
    if not parsed:
        return ClassifyResult(
            related=None,
            confidence=None,
            reason="",
            raw=content,
        )
    rel = parsed.get("related")
    conf = parsed.get("confidence")
    reason = str(parsed.get("reason", ""))
    related_bool: Optional[bool] = rel if isinstance(rel, bool) else None
    try:
        conf_f = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf_f = None
    return ClassifyResult(
        related=related_bool,
        confidence=conf_f,
        reason=reason,
        raw=content,
    )


def classify_deepseek(
    sentence: str,
    api_key: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com/chat/completions",
    timeout_sec: int = 120,
) -> ClassifyResult:
    key = api_key.strip()
    if not key:
        return ClassifyResult(
            related=None,
            confidence=None,
            reason="",
            raw='{"error": "empty API key"}',
        )
    try:
        key.encode("latin-1")
    except UnicodeEncodeError:
        return ClassifyResult(
            related=None,
            confidence=None,
            reason="",
            raw='{"error": "API key must be ASCII"}',
        )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(sentence=sentence)},
        ],
        "temperature": 0,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return ClassifyResult(
            related=None,
            confidence=None,
            reason="",
            raw=json.dumps({"error": str(e)}, ensure_ascii=False),
        )
    try:
        obj = json.loads(raw)
        content = (
            obj.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        return ClassifyResult(
            related=None,
            confidence=None,
            reason="",
            raw=json.dumps({"error": str(e)}, ensure_ascii=False),
        )
    parsed = _extract_json_object(content)
    return _result_from_parsed(content, parsed)


def classify_sentence(
    sentence: str,
    provider: str,
    ollama_model: str,
    ollama_base_url: str,
    deepseek_api_key: str = "",
    deepseek_model: str = "deepseek-chat",
    deepseek_base_url: str = "https://api.deepseek.com/chat/completions",
    timeout_sec: int = 120,
) -> ClassifyResult:
    if provider == "mock":
        return classify_mock(sentence)
    if provider == "ollama":
        return classify_ollama(
            sentence, model=ollama_model, base_url=ollama_base_url, timeout_sec=timeout_sec
        )
    if provider == "deepseek":
        return classify_deepseek(
            sentence,
            api_key=deepseek_api_key,
            model=deepseek_model,
            base_url=deepseek_base_url,
            timeout_sec=timeout_sec,
        )
    raise ValueError(f"Unknown provider: {provider}")
