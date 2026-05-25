"""Ingest PDFs from reference/法律参考文件/ into input.md appendix."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

APPENDIX_HEADER = "## 附录：法律参考文件自动摘录"
PDF_SECTION_RE = re.compile(
    r"^###\s+来源：(.+?)\s*$",
    re.MULTILINE,
)


def default_reference_dir() -> Path:
    """project/reference/法律参考文件 (sibling of project/src)."""
    return Path(__file__).resolve().parents[2] / "reference" / "法律参考文件"


def default_input_md() -> Path:
    return Path(__file__).resolve().parent / "input.md"


def _extract_pdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF ingestion requires pypdf. Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(text.strip())
    return "\n\n".join(chunks)


def _normalize_excerpt(text: str, max_chars: int = 12000) -> str:
    cleaned = re.sub(r"\r\n?", "\n", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n\n…（摘录已截断）"
    return cleaned


def _strip_existing_appendix(content: str) -> str:
    idx = content.find(APPENDIX_HEADER)
    if idx == -1:
        return content.rstrip() + "\n"
    return content[:idx].rstrip() + "\n"


def build_appendix(pdf_dir: Path) -> tuple[str, list[dict[str, str]]]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        return "", []

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "",
        APPENDIX_HEADER,
        "",
        f"> 由 `legal_pdf_ingest` 于 {ts} UTC 从 `{pdf_dir.name}/` 生成，供人工对照；",
        "> **不自动改写** 上文 A–H 矩阵表格。更新矩阵请人工编辑后运行 `kb-build`。",
        "",
    ]
    ingested: list[dict[str, str]] = []

    for pdf in pdfs:
        excerpt = _normalize_excerpt(_extract_pdf_text(pdf))
        rel = pdf.name
        lines.extend(
            [
                f"### 来源：{rel}",
                "",
                excerpt or "（未能提取文本）",
                "",
            ]
        )
        ingested.append({"file": rel, "chars": str(len(excerpt))})

    return "\n".join(lines), ingested


def ingest_reference_pdfs(
    *,
    reference_dir: Path | None = None,
    input_md: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    ref_dir = (reference_dir or default_reference_dir()).resolve()
    input_path = (input_md or default_input_md()).resolve()

    if not ref_dir.is_dir():
        return {
            "status": "skipped",
            "reason": f"reference directory not found: {ref_dir}",
            "reference_dir": str(ref_dir),
            "ingested": [],
        }

    pdfs = list(ref_dir.glob("*.pdf"))
    if not pdfs:
        return {
            "status": "skipped",
            "reason": "no PDF files in reference directory",
            "reference_dir": str(ref_dir),
            "ingested": [],
            "hint": f"Place PIPL/CCPA/GDPR PDFs under {ref_dir}",
        }

    appendix, ingested = build_appendix(ref_dir)
    base = _strip_existing_appendix(input_path.read_text(encoding="utf-8"))
    updated = base + appendix

    if not dry_run:
        input_path.write_text(updated, encoding="utf-8")

    return {
        "status": "ok",
        "reference_dir": str(ref_dir),
        "input_md": str(input_path),
        "pdf_count": len(pdfs),
        "ingested": ingested,
        "dry_run": dry_run,
    }
