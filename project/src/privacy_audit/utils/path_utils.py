"""Path helpers: provenance paths are always under output/."""

from __future__ import annotations

from pathlib import Path


def provenance_base() -> Path:
    """CLI working root: project/src."""
    return Path(__file__).resolve().parents[2]


def output_root() -> Path:
    return provenance_base() / "output"


def as_output_path(path: Path | str | None) -> str:
    """
    Return a forward-slash path starting with output/ for files under src/output.

    Returns empty string if the path is outside output/ (use canonical provenance helpers).
    """
    if path is None:
        return ""
    text = str(path).strip()
    if not text:
        return ""
    anchor = provenance_base().resolve()
    out_root = output_root().resolve()
    target = Path(text)
    if not target.is_absolute():
        target = (anchor / target).resolve()
    else:
        target = target.resolve()
    try:
        rel = target.relative_to(out_root)
    except ValueError:
        return ""
    return f"output/{rel.as_posix()}"


def output_provenance(*parts: str) -> str:
    """Build output/... provenance path from segments (no leading slashes)."""
    cleaned = [part.strip("/\\") for part in parts if part and part.strip("/\\")]
    if not cleaned:
        return "output"
    return "output/" + "/".join(cleaned)


def relativize_provenance_map(provenance: dict[str, str]) -> dict[str, str]:
    """Normalize provenance values to output/ form."""
    normalized: dict[str, str] = {}
    for key, value in provenance.items():
        if not value:
            continue
        if value.startswith("output/"):
            normalized[key] = value.replace("\\", "/")
            continue
        rel = as_output_path(value)
        normalized[key] = rel if rel else value.replace("\\", "/")
    return normalized
