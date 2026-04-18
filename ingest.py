import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings

from clean_files import extract_text_from_raw, split_by_items

RAW_DIR = "sec-edgar-filings"
PERSIST_DIR = "chroma_finance_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_filings(raw_dir: str) -> list[tuple[Document, str]]:
    """Returns list of (Document, pre_norm_text) tuples.

    pre_norm_text retains newlines needed by split_by_items for MULTILINE regex matching.
    """
    docs = []
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            extracted, filing_meta, pre_norm = extract_text_from_raw(raw)
            if not extracted.strip():
                continue
            parts = path.split(os.sep)
            meta = {
                "source": path,
                "ticker": parts[-4] if len(parts) >= 4 else "unknown",
                "form_type": parts[-3] if len(parts) >= 3 else "unknown",
                "accession": parts[-2] if len(parts) >= 2 else "unknown",
                **filing_meta,
            }
            docs.append((Document(page_content=extracted, metadata=meta), pre_norm))
    return docs


def chunk_documents(docs: list[tuple[Document, str]]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for doc, pre_norm in docs:
        # split_by_items requires pre_norm (newlines intact) for MULTILINE Item header matching
        sections = split_by_items(pre_norm)
        source_sections = sections if sections else [doc.page_content]
        for i, section in enumerate(source_sections):
            for j, chunk_text in enumerate(splitter.split_text(section)):
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={**doc.metadata, "section_index": i, "chunk_index": j},
                ))
    return chunks


if __name__ == "__main__":
    print("[1/3] Loading and cleaning filings...")
    docs = load_filings(RAW_DIR)
    if not docs:
        print(f"No .txt files found in '{RAW_DIR}'. Run get_files.py first.")
        raise SystemExit(1)
    print(f"      {len(docs)} filing(s) loaded")

    print("[2/3] Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"      {len(chunks)} chunks produced")

    print("[3/3] Building embeddings and persisting to ChromaDB...")
    if os.path.exists(PERSIST_DIR):
        print(f"      Removing existing store at '{PERSIST_DIR}'...")
        shutil.rmtree(PERSIST_DIR)
    print("      This may take several minutes on first run.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"\n[+] Done. {len(chunks)} chunks stored in '{PERSIST_DIR}'.")
    print("    Run 'python query.py' to start asking questions.")
