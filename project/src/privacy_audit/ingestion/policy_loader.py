"""Policy loader placeholder."""

from pathlib import Path


def load_policy_text(path: Path) -> str:
    """Read policy text file as UTF-8."""
    return path.read_text(encoding="utf-8")
