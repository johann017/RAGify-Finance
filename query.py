import os
import re
from typing import Optional

import torch
from transformers import pipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import datetime

PERSIST_DIR = "chroma_finance_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "microsoft/Phi-3.5-mini-instruct"
TOP_K = 5  # chunks for single-company queries
TOP_K_MULTI = 3  # chunks per company for multi-company queries
MAX_NEW_TOKENS = 768

SYSTEM_PROMPT = (
    f"Today's date is {datetime.date.today().strftime('%B %d, %Y')}. "
    "You are a financial analyst assistant with expertise in interpreting SEC 10-K filings. "
    "The source documents are annual 10-K filings only — never reference 10-Q or any other form type. "
    "Answer questions using ONLY the provided filing excerpts as your source of truth. "
    "Include a summary section which summarizes key points for each company mentioned in the question, and a details section with more in-depth analysis. "
    "Structure every response as:\n"
    "**Summary** — one to five sentences which should cover key points for each company mentioned in the question (cover every company, no exceptions)\n"
    "**Details** — a short paragraph per company with key figures, citing ticker, filing date, and excerpt. Include references and quotes from the excerpts, citing them properly.\n"
    "Be concise. If a company has no data in the excerpts, state that explicitly in its bullet/paragraph."
)

# Maps company names/aliases to the ticker stored in ChromaDB metadata
TICKER_ALIASES: dict[str, str] = {
    "apple": "AAPL",
    "aapl": "AAPL",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "googl": "GOOGL",
}


def detect_years_back(question: str) -> Optional[int]:
    """Return number of years back requested, e.g. 'last 2 years' → 2."""
    m = re.search(r"last\s+(\d+)\s+year", question, re.IGNORECASE)
    return int(m.group(1)) if m else None


def date_cutoff(years_back: int) -> str:
    """Return YYYYMMDD string for (today - years_back years)."""
    cutoff = datetime.date.today().replace(year=datetime.date.today().year - years_back)
    return cutoff.strftime("%Y%m%d")


def detect_tickers(question: str) -> list[str]:
    """Return tickers mentioned in the question, in order of appearance."""
    q = question.lower()
    seen: set[str] = set()
    result: list[str] = []
    for alias, ticker in TICKER_ALIASES.items():
        if alias in q and ticker not in seen:
            seen.add(ticker)
            result.append(ticker)
    return result


def load_generator():
    if torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device_map = {"": "mps"}
        torch_dtype = torch.float16
    else:
        device_map = {"": "cpu"}
        torch_dtype = torch.float32

    print(f"Loading {GENERATION_MODEL}  (first run downloads ~7 GB)...")
    return pipeline(
        "text-generation",
        model=GENERATION_MODEL,
        dtype=torch_dtype,
        device_map=device_map,
    )


def load_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def _filter_by_cutoff(docs: list, cutoff: str) -> list:
    """Keep only docs whose filed_date (YYYYMMDD string) >= cutoff."""
    return [d for d in docs if d.metadata.get("filed_date", "") >= cutoff]


def retrieve(vector_db: Chroma, question: str, ticker: Optional[str] = None) -> list:
    years_back = detect_years_back(question)
    cutoff = date_cutoff(years_back) if years_back else None
    # Fetch extra when date-filtering so we still have enough after pruning
    k_single = TOP_K * 3 if cutoff else TOP_K
    k_multi = TOP_K_MULTI * 3 if cutoff else TOP_K_MULTI

    # Explicit ticker prefix (e.g. "AAPL: ...") overrides auto-detection
    if ticker:
        docs = vector_db.as_retriever(
            search_kwargs={"k": k_single, "filter": {"ticker": ticker.upper()}}
        ).invoke(question)
        return _filter_by_cutoff(docs, cutoff)[:TOP_K] if cutoff else docs

    tickers = detect_tickers(question)
    if len(tickers) > 1:
        docs: list = []
        for t in tickers:
            t_docs = vector_db.as_retriever(
                search_kwargs={"k": k_multi, "filter": {"ticker": t}}
            ).invoke(question)
            if cutoff:
                t_docs = _filter_by_cutoff(t_docs, cutoff)[:TOP_K_MULTI]
            docs += t_docs
        return docs

    search_kwargs: dict = {"k": k_single}
    if tickers:
        search_kwargs["filter"] = {"ticker": tickers[0]}
    docs = vector_db.as_retriever(search_kwargs=search_kwargs).invoke(question)
    return _filter_by_cutoff(docs, cutoff)[:TOP_K] if cutoff else docs


def format_context(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        label = f"{m.get('ticker', '?')} | {m.get('form_type', '?')} | filed {m.get('filed_date', '?')}"
        parts.append(f"[Excerpt {i} — {label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def ask(
    generator, vector_db: Chroma, question: str, ticker: Optional[str] = None
) -> None:
    docs = retrieve(vector_db, question, ticker)
    if not docs:
        print("No relevant excerpts found in the database.")
        return

    context = format_context(docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"FILING EXCERPTS:\n{context}\n\nQUESTION: {question}",
        },
    ]

    output = generator(
        messages, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.1
    )
    answer = output[0]["generated_text"][-1]["content"]
    print(f"\nAnswer: {answer}\n")


def parse_input(text: str) -> tuple:
    """Parse optional 'TICKER: question' prefix."""
    if ":" in text:
        prefix, rest = text.split(":", 1)
        prefix = prefix.strip()
        if prefix.isalpha() and len(prefix) <= 5:
            return rest.strip(), prefix.upper()
    return text.strip(), None


def main():
    if not os.path.exists(PERSIST_DIR):
        print(
            f"Vector store not found at '{PERSIST_DIR}'. Run 'python ingest.py' first."
        )
        return

    print("Loading vector store...")
    vector_db = load_vector_store()
    generator = load_generator()

    print("\nRAG Finance Q&A  |  type 'quit' to exit")
    print(
        "Tip: prefix with a ticker to filter, e.g. 'AAPL: what are the main risks?'\n"
    )

    while True:
        try:
            raw = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not raw or raw.lower() in ("quit", "exit", "q"):
            break
        question, ticker = parse_input(raw)
        if not question:
            continue
        ask(generator, vector_db, question, ticker)


if __name__ == "__main__":
    main()
