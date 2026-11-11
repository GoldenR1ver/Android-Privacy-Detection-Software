"""Tests for DeepSeek LLM navigation helpers."""

from __future__ import annotations

from privacy_audit.analyzers.dynamic.llm_path_assistant import (
    _resolve_bridge_label,
    extract_visible_nav_labels,
    suggest_intermediate_bridge,
)


_SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy>
  <node class="android.widget.TextView" text="隐私设置" displayed="true" clickable="false"/>
  <node class="android.widget.TextView" text="找到我的方式" displayed="true" clickable="true"/>
  <node class="android.widget.TextView" text="把我推荐给可能认识的人" displayed="true" clickable="false"/>
</hierarchy>
"""


def test_extract_visible_nav_labels_from_xml() -> None:
    labels = extract_visible_nav_labels(_SAMPLE_XML)
    assert "找到我的方式" in labels
    assert "隐私设置" in labels


def test_resolve_bridge_label_exact_and_partial() -> None:
    labels = ["找到我的方式", "隐私设置"]
    assert _resolve_bridge_label("找到我的方式", labels) == "找到我的方式"
    assert _resolve_bridge_label("找到我的", labels) == "找到我的方式"


def test_suggest_intermediate_bridge_without_api_key() -> None:
    result = suggest_intermediate_bridge(
        "不把我推荐给可能认识的人",
        ["把我推荐给可能认识的人"],
        ["我", "设置", "隐私设置"],
        ["找到我的方式", "隐私设置"],
        api_key="",
    )
    assert result["bridge"] is None
    assert "DEEPSEEK_API_KEY" in str(result.get("error", ""))
