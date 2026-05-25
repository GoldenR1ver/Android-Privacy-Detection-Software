"""Adapter interface placeholders."""

from typing import Protocol


class Analyzer(Protocol):
    def run(self) -> dict:
        """Run analysis and return structured result."""
