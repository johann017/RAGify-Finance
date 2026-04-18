import os
from typing import Optional

import torch
from transformers import pipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_finance_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "microsoft/Phi-3.5-mini-instruct"
TOP_K = 6
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a financial analyst assistant with expertise in interpreting SEC filings. "
    "Answer questions using ONLY the provided filing excerpts as your source of truth. "
    "Be precise, cite the source company and filing date when possible, and clearly "
    "state when information is not available in the excerpts."
)


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


def retrieve(vector_db: Chroma, question: str, ticker: Optional[str] = None) -> list:
    search_kwargs: dict = {"k": TOP_K}
    if ticker:
        search_kwargs["filter"] = {"ticker": ticker.upper()}
    return vector_db.as_retriever(search_kwargs=search_kwargs).invoke(question)


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

    output = generator(messages, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
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
