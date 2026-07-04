import os

import bs4
from langchain_core.documents import Document

from logging_config import get_logger

logger = get_logger(__name__)

SEC_EDGAR_FILINGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "sec-edgar-filings"
)


def get_docs():
    logger.info("Scanning %s for filing documents", SEC_EDGAR_FILINGS_DIR)
    documents = []
    for root, _, files in os.walk(SEC_EDGAR_FILINGS_DIR):
        path_parts = root.split(os.sep)
        company = path_parts[-3] if len(path_parts) >= 3 else "unknown"
        year = path_parts[-1] if len(path_parts) >= 1 else "unknown"
        year = f"20{year.split('-')[1]}" if "-" in year else "unknown"
        filing_type = path_parts[-2] if len(path_parts) >= 2 else "unknown"
        for file in files:
            if not file.startswith("primary-document"):
                continue
            file_path = os.path.join(root, file)
            logger.debug(
                "Loading %s (company=%s, year=%s, filing_type=%s)",
                file_path,
                company,
                year,
                filing_type,
            )
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            text = bs4.BeautifulSoup(raw, "html.parser").get_text(separator="\n")
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "company": company,
                        "year": year,
                        "filing_type": filing_type,
                    },
                )
            )
    if not documents:
        logger.warning("No primary-document files found under %s", SEC_EDGAR_FILINGS_DIR)
    else:
        logger.info("Loaded %d filing document(s)", len(documents))
    return documents
