"""Load and query structured legal knowledge base JSON files."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def default_kb_root() -> Path:
    return Path(__file__).resolve().parent / "knowledge_base"


class KnowledgeBaseStore:
    """In-memory view over categories, aliases, android rules, and audit bridge."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_kb_root()).resolve()
        self.categories_doc = self._read("categories.json")
        self.aliases_doc = self._read("aliases.json")
        self.sensitivity_doc = self._read("sensitivity.json")
        self.android_doc = self._read("android_mapping.json")
        self.risk_doc = self._read("risk_rules.json")
        self.bridge_doc = self._read("audit_taxonomy_bridge.json")

        self._alias_index: dict[str, list[str]] = self._build_alias_index()
        self._item_by_id: dict[str, dict[str, Any]] = self._build_item_index()
        self._audit_to_legal: dict[str, list[str]] = dict(
            self.bridge_doc.get("audit_to_legal", {})
        )
        self._legal_to_audit: dict[str, list[str]] = dict(
            self.bridge_doc.get("legal_to_audit", {})
        )

    def _read(self, name: str) -> dict[str, Any]:
        path = self.root / name
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_alias_index(self) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
        index = self.categories_doc.get("index", {}).get("by_alias", {})
        if isinstance(index, dict):
            merged.update({str(k): list(v) for k, v in index.items()})
        aliases = self.aliases_doc.get("aliases", {})
        if isinstance(aliases, dict):
            for key, value in aliases.items():
                merged.setdefault(str(key), [])
                for item_id in value:
                    if item_id not in merged[str(key)]:
                        merged[str(key)].append(item_id)
        return merged

    def _build_item_index(self) -> dict[str, dict[str, Any]]:
        items: dict[str, dict[str, Any]] = {}
        for category in self.categories_doc.get("categories", []):
            for item in category.get("items", []):
                item_id = str(item.get("id", ""))
                if item_id:
                    items[item_id] = {
                        **item,
                        "category_id": category.get("id"),
                        "category_sensitive": category.get("sensitive", False),
                    }
        return items

    def match_text_to_legal_items(self, text: str, min_alias_len: int = 2) -> list[str]:
        """Match free text to legal item ids via alias index (longest token first)."""
        if not text or not text.strip():
            return []
        lowered = text.lower()
        matched: set[str] = set()
        for alias, item_ids in sorted(self._alias_index.items(), key=lambda x: -len(x[0])):
            alias = str(alias).strip()
            if len(alias) < min_alias_len:
                continue
            if alias.lower() in lowered:
                matched.update(item_ids)
        return sorted(matched)

    def legal_items_to_audit_types(self, item_ids: list[str]) -> list[str]:
        out: set[str] = set()
        for item_id in item_ids:
            for audit_type in self._legal_to_audit.get(item_id, []):
                out.add(audit_type)
        return sorted(out)

    def audit_types_to_legal_items(self, audit_types: list[str]) -> list[str]:
        out: set[str] = set()
        for audit_type in audit_types:
            out.update(self._audit_to_legal.get(audit_type, []))
        return sorted(out)

    def item_label(self, item_id: str) -> str:
        item = self._item_by_id.get(item_id, {})
        return str(item.get("label", item_id))

    def item_sensitivity(self, item_id: str) -> dict[str, Any]:
        item = self._item_by_id.get(item_id, {})
        return dict(item.get("sensitivity", {}))

    def risk_bucket_for_item(self, item_id: str) -> str:
        buckets = self.risk_doc.get("buckets", {})
        for bucket, ids in buckets.items():
            if item_id in ids:
                return str(bucket)
        sens = self.item_sensitivity(item_id).get("normalized")
        if sens == "sensitive":
            return "HIGH"
        if sens == "general":
            return "MEDIUM"
        return "LOW"

    def android_rules(self) -> list[dict[str, Any]]:
        rules = self.android_doc.get("rules", [])
        return list(rules) if isinstance(rules, list) else []

    @property
    def alias_index(self) -> dict[str, list[str]]:
        return self._alias_index

    @property
    def item_by_id(self) -> dict[str, dict[str, Any]]:
        return self._item_by_id


@lru_cache(maxsize=4)
def get_kb_store(root: str | None = None) -> KnowledgeBaseStore:
    path = Path(root).resolve() if root else default_kb_root()
    return KnowledgeBaseStore(path)
