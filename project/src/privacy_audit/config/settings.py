"""Runtime settings placeholders."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
