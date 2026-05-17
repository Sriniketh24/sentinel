import uuid
from datetime import datetime
from pathlib import Path

import httpx
import structlog

from src.config import get_settings
from src.models.document import Document, DocumentType

log = structlog.get_logger()

EDGAR_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"

FORM_TYPE_MAP = {
    "10-K": DocumentType.SEC_10K,
    "10-Q": DocumentType.SEC_10Q,
    "8-K": DocumentType.SEC_8K,
}


class EdgarClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._headers = {"User-Agent": settings.sec_edgar_user_agent}
        self._data_dir = Path("data/raw/edgar")
        self._data_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_filings(
        self,
        ticker: str,
        form_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[Document]:
        if form_types is None:
            form_types = ["10-K", "10-Q"]

        cik = await self._resolve_cik(ticker)
        if not cik:
            log.warning("could not resolve CIK", ticker=ticker)
            return []

        documents: list[Document] = []
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            url = EDGAR_SUBMISSIONS.format(cik=cik.zfill(10))
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accession_numbers = recent.get("accessionNumber", [])
            filing_dates = recent.get("filingDate", [])
            primary_docs = recent.get("primaryDocument", [])

            count = 0
            for i, form in enumerate(forms):
                if count >= limit:
                    break
                if form not in form_types:
                    continue

                accession = accession_numbers[i].replace("-", "")
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_docs[i]}"
                )

                file_path = await self._download_filing(client, doc_url, ticker, form, i)

                documents.append(
                    Document(
                        id=str(uuid.uuid4()),
                        source_url=doc_url,
                        file_path=str(file_path) if file_path else "",
                        doc_type=FORM_TYPE_MAP.get(form, DocumentType.UNKNOWN),
                        company_ticker=ticker.upper(),
                        company_name=data.get("name", ""),
                        filing_date=datetime.fromisoformat(filing_dates[i]),
                        metadata={"accession_number": accession_numbers[i], "form": form},
                    )
                )
                count += 1

        log.info("fetched filings", ticker=ticker, count=len(documents))
        return documents

    async def _resolve_cik(self, ticker: str) -> str | None:
        url = "https://www.sec.gov/files/company_tickers.json"
        async with httpx.AsyncClient(headers=self._headers, timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            tickers = resp.json()

        for entry in tickers.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"])
        return None

    async def _download_filing(
        self,
        client: httpx.AsyncClient,
        url: str,
        ticker: str,
        form: str,
        index: int,
    ) -> Path | None:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            suffix = ".htm" if "htm" in url else ".txt"
            path = self._data_dir / f"{ticker}_{form}_{index}{suffix}"
            path.write_bytes(resp.content)
            return path
        except httpx.HTTPError:
            log.error("failed to download filing", url=url)
            return None
