"""PDF text extraction using pdfplumber.

Extracts text and tables page-by-page from digital PDFs.
Does NOT perform OCR — returns a fallback message for scanned PDFs.
"""

from __future__ import annotations

from io import BytesIO

import structlog

log = structlog.get_logger()

# Minimum characters to consider a PDF as having extractable text
MIN_TEXT_THRESHOLD = 50


def extract_pdf_text(
    file_bytes: bytes,
    filename: str | None = None,
) -> dict:
    """Extract text and tables from a PDF file.

    Args:
        file_bytes: Raw PDF bytes.
        filename: Original filename for metadata.

    Returns:
        dict with keys: filename, page_count, character_count,
        has_text, content (or error).
    """
    import pdfplumber

    fname = filename or "upload.pdf"
    pages_text: list[str] = []
    page_count = 0

    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            log.info("pdf_opened", filename=fname, pages=page_count)

            for i, page in enumerate(pdf.pages, start=1):
                parts: list[str] = [f"--- Page {i} ---"]

                # Extract main text
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(text.strip())

                # Extract tables
                try:
                    tables = page.extract_tables() or []
                    for t_idx, table in enumerate(tables, start=1):
                        if not table:
                            continue
                        formatted = _format_table(table, t_idx)
                        if formatted:
                            parts.append(formatted)
                except Exception:
                    # Table extraction can fail on complex layouts — skip
                    pass

                pages_text.append("\n".join(parts))

    except Exception as e:
        log.error("pdf_extraction_failed", filename=fname, error=str(e)[:200])
        return {
            "filename": fname,
            "page_count": 0,
            "character_count": 0,
            "has_text": False,
            "error": f"Failed to process PDF: {str(e)[:200]}",
        }

    full_text = "\n\n".join(pages_text)
    char_count = len(full_text)

    # Check if the PDF had meaningful text
    if char_count < MIN_TEXT_THRESHOLD:
        log.info("pdf_no_text", filename=fname, chars=char_count)
        return {
            "filename": fname,
            "page_count": page_count,
            "character_count": char_count,
            "has_text": False,
            "error": (
                "This PDF appears to be scanned. "
                "OCR support is not enabled yet."
            ),
        }

    log.info("pdf_extracted", filename=fname, pages=page_count, chars=char_count)
    return {
        "filename": fname,
        "page_count": page_count,
        "character_count": char_count,
        "has_text": True,
        "content": full_text,
    }


def _format_table(table: list[list], table_idx: int) -> str | None:
    """Format a pdfplumber table as readable plain text."""
    if not table or len(table) < 2:
        return None

    rows: list[str] = []
    for row in table:
        cells = [str(cell).strip() if cell else "" for cell in row]
        rows.append(" | ".join(cells))

    if not any(r.strip(" |") for r in rows):
        return None

    return f"\n[Table {table_idx}]\n" + "\n".join(rows)
