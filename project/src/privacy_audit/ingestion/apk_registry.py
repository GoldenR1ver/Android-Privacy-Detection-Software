"""APK registry placeholder."""

from pathlib import Path


def load_apk_list(apk_dir: Path) -> list[Path]:
    """Return APK files in a directory."""
    return sorted(apk_dir.glob("*.apk"))
