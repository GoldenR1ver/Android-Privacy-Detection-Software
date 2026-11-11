"""R4: tier sync and dynamic-track auto batch helpers."""

from __future__ import annotations

import json
from pathlib import Path

from privacy_audit.workflows.capture_track import resolve_capture_tracks
from privacy_audit.workflows.tier_sync import apply_probe_to_tier_entry, sync_frida_tier_from_probe_dir
from privacy_audit.analyzers.dynamic.frida_runner import should_enable_frida_for_consent


def test_apply_probe_never_ready() -> None:
    entry = apply_probe_to_tier_entry({"java_bridge": "never_ready", "success": False})
    assert entry["tier"] == "network_static_only"
    assert entry["enable_frida_consent"] is False
    assert "mitm" in entry["preferred_dynamic_tracks"]


def test_should_enable_frida_respects_network_static_only(tmp_path: Path) -> None:
    tier_file = tmp_path / "frida_tier.json"
    tier_file.write_text(
        json.dumps(
            {
                "defaults": {},
                "apps": {
                    "钓鱼人": {
                        "tier": "network_static_only",
                        "java_bridge": "never_ready",
                        "enable_frida_consent": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert should_enable_frida_for_consent("钓鱼人", tier_file) is False


def test_sync_frida_tier_from_probe_dir(tmp_path: Path) -> None:
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "diaoyuren.json").write_text(
        json.dumps({"java_bridge": "never_ready", "success": False}),
        encoding="utf-8",
    )
    tier_file = tmp_path / "frida_tier.json"
    tier_file.write_text(
        json.dumps({"defaults": {}, "apps": {"钓鱼人": {"tier": "unknown"}}}),
        encoding="utf-8",
    )
    result = sync_frida_tier_from_probe_dir(tier_file, probe_dir, write=True)
    assert "钓鱼人" in result["updated_apps"]
    data = json.loads(tier_file.read_text(encoding="utf-8"))
    assert data["apps"]["钓鱼人"]["java_bridge"] == "never_ready"


def test_resolve_auto_uses_tier_file(tmp_path: Path) -> None:
    tier_file = tmp_path / "frida_tier.json"
    tier_file.write_text(
        json.dumps(
            {
                "defaults": {},
                "apps": {
                    "钓鱼人": {
                        "tier": "network_static_only",
                        "java_bridge": "never_ready",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    plan = resolve_capture_tracks(
        "钓鱼人",
        dynamic_track="auto",
        probe_dir=tmp_path / "missing",
        frida_tier_file=tier_file,
    )
    assert plan["enable_frida"] is False
    assert plan["enable_mitm"] is True
    assert plan["mitm_primary"] is True
    assert plan["dynamic_track_resolved"] == "mitm"


