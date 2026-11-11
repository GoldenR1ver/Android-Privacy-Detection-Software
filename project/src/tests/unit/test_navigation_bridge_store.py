"""Tests for navigation bridge persistence."""

from __future__ import annotations

from pathlib import Path

from privacy_audit.analyzers.dynamic.navigation_bridge_store import NavigationBridgeStore


def test_bridge_store_persists_learned_bridge(tmp_path: Path) -> None:
    path = tmp_path / "bridges.json"
    store = NavigationBridgeStore(path)
    store.load()
    assert store.bridges_for("测试目标") == ()
    assert store.add_bridge("测试目标", "中间菜单")
    assert store.bridges_for("测试目标") == ("中间菜单",)

    reloaded = NavigationBridgeStore(path)
    reloaded.load()
    assert reloaded.bridges_for("测试目标") == ("中间菜单",)
