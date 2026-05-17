from pathlib import Path

import pdfplumber
import structlog

from src.models.document import DocumentChunk, Modality

log = structlog.get_logger()


class PDFParser:
    def parse(self, file_path: str, document_id: str) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        path = Path(file_path)

        if not path.exists():
            log.error("file not found", path=file_path)
            return chunks

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    chunks.append(
                        DocumentChunk(
                            id=f"{document_id}_p{page_num}_text",
                            document_id=document_id,
                            content=text,
                            modality=Modality.TEXT,
                            chunk_index=len(chunks),
                            page_number=page_num,
                            metadata={"source": "pdf_text"},
                        )
                    )

                for table_idx, table in enumerate(page.extract_tables()):
                    if not table:
                        continue
                    table_text = self._table_to_markdown(table)
                    if table_text.strip():
                        chunks.append(
                            DocumentChunk(
                                id=f"{document_id}_p{page_num}_t{table_idx}",
                                document_id=document_id,
                                content=table_text,
                                modality=Modality.TABLE,
                                chunk_index=len(chunks),
                                page_number=page_num,
                                metadata={"source": "pdf_table", "table_index": table_idx},
                            )
                        )

        log.info("parsed pdf", path=file_path, chunks=len(chunks))
        return chunks

    @staticmethod
    def _table_to_markdown(table: list[list[str | None]]) -> str:
        if not table:
            return ""

        cleaned = [[cell or "" for cell in row] for row in table]
        header = cleaned[0]
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for row in cleaned[1:]:
            padded = row + [""] * (len(header) - len(row)) if len(row) < len(header) else row
            lines.append("| " + " | ".join(padded[: len(header)]) + " |")
        return "\n".join(lines)
