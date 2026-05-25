"""Frida attach candidate ranking."""

from privacy_audit.analyzers.dynamic.frida_runner import _resolve_attach_candidates


class _Proc:
    def __init__(self, pid: int, name: str, identifier: str) -> None:
        self.pid = pid
        self.name = name
        self.identifier = identifier


class _Device:
    def __init__(self, processes: list[_Proc]) -> None:
        self._processes = processes

    def enumerate_processes(self):
        return self._processes


def test_skips_display_name_process() -> None:
    device = _Device(
        [
            _Proc(100, "Keep", ""),
            _Proc(200, "com.gotokeep.keep", ""),
            _Proc(300, "com.gotokeep.keep:core", ""),
        ]
    )
    ranked = _resolve_attach_candidates(device, "com.gotokeep.keep", "Keep")
    pids = [row[1] for row in ranked]
    assert 300 not in pids
    assert pids[0] == 100
    assert 200 in pids
