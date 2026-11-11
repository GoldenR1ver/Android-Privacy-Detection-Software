"""Persist learned intermediate navigation bridges (policy token -> UI submenu)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("privacy_audit.navigation_bridge")

DEFAULT_BRIDGE_FILE = Path("input/navigation_bridges.json")

# Shipped defaults; file entries extend/override per token.
_BUILTIN_BRIDGES: dict[str, tuple[str, ...]] = {
    "不把我推荐给可能认识的人": ("找到我的方式",),
    "把我推荐给可能认识的人": ("找到我的方式",),
    "用户保护中心": ("设置",),
}


class NavigationBridgeStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_BRIDGE_FILE
        self._bridges: dict[str, list[str]] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._bridges = {
            token: list(labels)
            for token, labels in _BUILTIN_BRIDGES.items()
        }
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                file_bridges = raw.get("bridges", raw)
                if isinstance(file_bridges, dict):
                    for token, labels in file_bridges.items():
                        key = str(token).strip()
                        if not key:
                            continue
                        if isinstance(labels, str):
                            seq = [labels]
                        elif isinstance(labels, (list, tuple)):
                            seq = [str(x).strip() for x in labels if str(x).strip()]
                        else:
                            continue
                        if seq:
                            self._bridges[key] = seq
            except Exception as exc:
                logger.warning("Could not load navigation bridges from %s: %s", self.path, exc)
        self._loaded = True

    def bridges_for(self, token: str) -> tuple[str, ...]:
        self.load()
        norm = token.strip()
        if norm in self._bridges:
            return tuple(self._bridges[norm])
        return ()

    def add_bridge(self, target_token: str, bridge_label: str) -> bool:
        self.load()
        target = target_token.strip()
        bridge = bridge_label.strip()
        if not target or not bridge:
            return False
        current = list(self._bridges.get(target, ()))
        if bridge in current:
            return False
        current.append(bridge)
        self._bridges[target] = current
        self._persist()
        logger.info("Learned navigation bridge: %s -> %s", target, bridge)
        return True

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": 1,
            "bridges": {k: v for k, v in sorted(self._bridges.items())},
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def snapshot(self) -> dict[str, list[str]]:
        self.load()
        return {k: list(v) for k, v in self._bridges.items()}
