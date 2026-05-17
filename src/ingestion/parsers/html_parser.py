import re
from pathlib import Path

import structlog

from src.models.document import DocumentChunk, Modality

log = structlog.get_logger()


class HTMLParser:
    def parse(self, file_path: str, document_id: str) -> list[DocumentChunk]:
        path = Path(file_path)
        if not path.exists():
            log.error("file not found", path=file_path)
            return []

        raw_html = path.read_text(errors="replace")
        text = self._strip_html(raw_html)
        tables = self._extract_tables(raw_html)

        chunks: list[DocumentChunk] = []

        if text.strip():
            chunks.append(
                DocumentChunk(
                    id=f"{document_id}_html_text",
                    document_id=document_id,
                    content=text,
                    modality=Modality.TEXT,
                    chunk_index=0,
                    metadata={"source": "html_text"},
                )
            )

        for i, table_text in enumerate(tables):
            if table_text.strip():
                chunks.append(
                    DocumentChunk(
                        id=f"{document_id}_html_table_{i}",
                        document_id=document_id,
                        content=table_text,
                        modality=Modality.TABLE,
                        chunk_index=len(chunks),
                        metadata={"source": "html_table", "table_index": i},
                    )
                )

        log.info("parsed html", path=file_path, text_len=len(text), tables=len(tables))
        return chunks

    @staticmethod
    def _strip_html(html: str) -> str:
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        html = re.sub(r"<[^>]+>", " ", html)
        html = re.sub(r"&nbsp;", " ", html)
        html = re.sub(r"&amp;", "&", html)
        html = re.sub(r"&lt;", "<", html)
        html = re.sub(r"&gt;", ">", html)
        html = re.sub(r"&#\d+;", "", html)
        html = re.sub(r"\s+", " ", html)
        return html.strip()

    @staticmethod
    def _extract_tables(html: str) -> list[str]:
        table_pattern = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
        tables: list[str] = []

        for match in table_pattern.finditer(html):
            table_html = match.group(1)
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE)

            parsed_rows: list[list[str]] = []
            for row_html in rows:
                cells = re.findall(
                    r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>", row_html, re.DOTALL | re.IGNORECASE
                )
                cleaned = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
                cleaned = [re.sub(r"\s+", " ", c) for c in cleaned]
                if any(c for c in cleaned):
                    parsed_rows.append(cleaned)

            if len(parsed_rows) >= 2:
                md = _rows_to_markdown(parsed_rows)
                if len(md) > 50:
                    tables.append(md)

        return tables


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    max_cols = max(len(r) for r in rows)
    header = rows[0] + [""] * (max_cols - len(rows[0]))
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")
    for row in rows[1:]:
        padded = row + [""] * (max_cols - len(row))
        lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(lines)
