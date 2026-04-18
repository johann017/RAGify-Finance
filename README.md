# RAGify Finance

A local, fully offline question-answering system over SEC 10-K annual filings. You point it at any public company, it downloads their filings, processes them, and lets you ask natural language questions and get answers grounded in the actual documents — not hallucinated from a model's training data.

No cloud APIs, no subscriptions, no data leaving your machine.

---

## What problem does this solve?

Large language models are good at reasoning but they have a knowledge cutoff and they can't know the contents of specific documents. If you ask one "what were Apple's main risk factors last year?" you'll get a plausible-sounding answer that may or may not reflect what Apple actually wrote. There's no way to verify it.

RAG (Retrieval-Augmented Generation) fixes this by splitting the problem in two. First, you build a searchable index of your actual documents. Then at query time, you find the most relevant passages and hand them directly to the model as context. The model now has to answer from *those specific excerpts*, not from memory. You can verify every answer by reading the source.

This project applies that pattern to SEC EDGAR, the US government's public database of every financial filing ever made by a public company. The 10-K is a company's annual report — it covers business overview, risk factors, competitive landscape, financial results, and management's outlook. It's one of the most information-dense documents a company produces, and it's all public.

---

## How it works

The project has two distinct phases that you run separately.

**Ingestion** (run once per batch of filings):

```
SEC EDGAR → raw .txt files → clean & extract → chunk → embed → ChromaDB
```

**Querying** (run whenever you have a question):

```
your question → embed → similarity search → top-6 chunks → Phi-3.5 → answer
```

Here's what each step is actually doing.

### Downloading: `get_files.py`

Uses the `sec-edgar-downloader` library to pull 10-K filings directly from SEC EDGAR's public API. EDGAR requires an identity header (your name and email) to make requests — it's not authentication, just a courtesy so they can contact you if your scraper misbehaves. The files land in `sec-edgar-filings/<TICKER>/10-K/<accession-number>/`.

### Cleaning: `clean_files.py`

Raw SEC filings are genuinely ugly. A single `.txt` file is an SGML bundle that contains the actual 10-K document, multiple attached exhibits, a metadata header, and layers of HTML tags all mixed together. Before any of this can be used for search, it needs to be turned into clean prose.

The cleaning pipeline does the following in order:

1. **Extract SGML metadata** — the header block contains structured fields like the filing date, company name, and CIK (Central Index Key). These get pulled out and stored as metadata on each document, so the retrieval system knows which company and filing year a chunk came from.

2. **Isolate the primary document** — the SGML bundle contains multiple `<DOCUMENT>` blocks. We look for the one tagged `<TYPE>10-K` and discard everything else. Exhibits, signature pages, and ancillary documents would just add noise to the search index.

3. **Strip markup** — the 10-K itself is usually HTML inside the SGML wrapper. BeautifulSoup parses it and extracts readable text, converting the tag soup into plain paragraphs.

4. **Remove boilerplate** — the signature page and exhibit index at the end of every 10-K are legally required boilerplate. They contribute nothing to answering questions, so they get cut.

5. **Remove table noise** — financial statements are full of rows like `"   42,819   38,294   "` that have no labels. A language model can't do anything with unlabeled numbers. We drop any line with no alphabetic characters at all, while keeping rows like `"Net revenue   94,680   91,154"` which have the label that makes the numbers meaningful.

6. **Deduplicate sentences** — SEC risk factor sections are notorious for repeating the same sentence verbatim in multiple places. Duplicates waste embedding space and retrieval slots, so they get removed.

7. **Normalize unicode** — SEC documents contain decades of formatting artifacts: smart quotes, em dashes, ballot boxes from old checkbox fields. These get converted to plain ASCII so the embedding model isn't confused by encoding variations of the same character.

`extract_text_from_raw` returns **two versions** of the cleaned text: a fully normalized flat string (whitespace collapsed, used as the document's stored content) and a **pre-normalized** version that retains newlines. The pre-normalized text is passed to the chunking step and is explained there.

### Chunking: `ingest.py`

You can't embed an entire 10-K as a single unit — it's hundreds of pages, and a single embedding for the whole thing would be too diluted to retrieve anything specific from. The document needs to be broken into focused pieces.

The approach here is two-pass. First, the text is split on SEC Item headers — "Item 1. Business", "Item 1A. Risk Factors", "Item 7. Management's Discussion and Analysis", and so on. These are semantically coherent sections and it makes sense to keep them mostly together. Then within each section, `RecursiveCharacterTextSplitter` breaks things down further into 800-character chunks with 100 characters of overlap.

**Why the pre-normalized text is used for section splitting:** The Item header detector (`ITEM_HEADER` in `clean_files.py`) is a regex that uses `^` to match only at the start of a line — this prevents it from accidentally splitting on inline cross-references like "see Item 1A for more detail." The `^` anchor only works when the text actually contains newlines. The final normalization step collapses all whitespace (including newlines) into single spaces, which destroys line boundaries and breaks the regex. So section splitting must happen *before* final normalization, on the pre-normalized text that still has its newlines intact. After splitting, each section is normalized individually.

The overlap is important. If a key sentence happens to fall right on a chunk boundary, you don't want it split across two chunks where neither half makes sense on its own. The overlap ensures there's always continuity between adjacent chunks.

Every chunk carries metadata: which ticker it came from, the filing date, which Item section it came from, and its position within that section. This is what makes filtered search possible — you can ask "only look at Apple's filings" and the system can honor that.

### Embedding: `ingest.py`

An embedding is a fixed-length list of numbers (a vector) that represents the *meaning* of a piece of text, not the literal words. Two sentences that mean the same thing should produce similar vectors, even if they use completely different words. This is what makes semantic search work — you're not matching keywords, you're matching meaning.

The embedding model used here is `sentence-transformers/all-MiniLM-L6-v2`. It's small (80MB), fast, runs entirely locally, and was specifically trained on semantic similarity tasks. Bigger and more capable embedding models exist, but this one is fast enough that ingesting thousands of chunks doesn't take forever, and the quality is more than sufficient for retrieval from structured corporate filings.

### Vector store: `ingest.py` → ChromaDB

ChromaDB stores each chunk alongside its embedding vector and persists everything to disk. When you search it, you provide a query vector and it returns the chunks whose vectors are most geometrically similar — cosine similarity, specifically.

ChromaDB was chosen because it's embedded (no server to run, no Docker, nothing to manage), stores to disk automatically, and supports metadata filtering out of the box. That last part matters: when you prefix your question with `AAPL:`, the retriever adds `{"ticker": "AAPL"}` to the search and only considers Apple chunks. Without that, a general question about supply chain risks might surface results from multiple companies and be harder to interpret.

### Retrieval and generation: `query.py`

When you ask a question, `query.py` does three things:

1. Embeds your question using the **same model** that was used during ingestion — this is critical, the query vector and the stored chunk vectors need to live in the same vector space to be comparable
2. Asks ChromaDB for the 6 most similar chunks
3. Formats those chunks as labeled context and hands everything to the language model

The language model is `microsoft/Phi-3.5-mini-instruct`, a 3.8B parameter model from Microsoft that runs entirely locally. It's given the retrieved excerpts and told — via the system prompt — to answer strictly from those excerpts, not from anything it may have learned during pre-training. This grounds the answers in the actual filings.

`do_sample=False` means greedy decoding: the model always picks the most likely next token rather than sampling. This makes answers deterministic and factual rather than creative or variable. For a Q&A tool where you want consistent, verifiable answers, that's the right mode.

The hardware detection at startup tries CUDA first (NVIDIA GPU), then MPS (Apple Silicon), then falls back to CPU. You don't have to configure anything — it uses whatever accelerator is available.

---

## Project structure

```
ragify-finance/
├── get_files.py        # Step 1: download 10-K filings from SEC EDGAR
├── clean_files.py      # Helper: text extraction and cleaning logic
├── ingest.py           # Step 2: chunk, embed, and store in ChromaDB
├── query.py            # Step 3: interactive Q&A
├── pyproject.toml      # Dependencies, managed by uv
├── uv.lock             # Exact locked versions for reproducible installs
└── .gitignore
```

The `sec-edgar-filings/` and `chroma_finance_db/` directories are intentionally excluded from git. Filings can be hundreds of megabytes and the vector store is derived data — both can be regenerated from scratch with the two pipeline scripts.

---

## Getting started

You need Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) installed.

```bash
git clone <repo-url>
cd ragify-finance
uv sync
```

**Step 1 — Configure and download filings**

Open `get_files.py` and set your name and email (required by SEC EDGAR for identification), then pick your tickers and how many years back to download:

```python
YOUR_NAME = 
YOUR_EMAIL = 
TICKERS = # company tickers
NUM_FILINGS = 3
```

```bash
uv run python get_files.py
```

**Step 2 — Build the vector store**

This embeds all the downloaded chunks and saves them to ChromaDB. It takes a few minutes depending on how many filings you downloaded.

```bash
uv run python ingest.py
```

**Step 3 — Ask questions**

The first time you run this, it downloads the Phi-3.5-mini model weights (~7 GB) to your HuggingFace cache at `~/.cache/huggingface/`. After that it loads from disk in seconds.

```bash
uv run python query.py
```

Example questions:

```
Question: AAPL: what does Apple say about its competitive position?
Question: NVDA: what are the main supply chain risks?
Question: GOOGL: how does Alphabet describe its AI strategy?
Question: what risks do all three companies have in common?
```

The `TICKER:` prefix filters the retrieval to that company's filings only. Leave it off to search across everything you've ingested.

---

## Design decisions and trade-offs

**Why 10-K filings specifically?**
The 10-K is the most comprehensive and legally accountable document a public company produces. Companies are required by the SEC to be accurate and complete — material omissions are securities fraud. That's a much stronger quality guarantee than press releases, earnings calls, or investor presentations, which are more curated and less candid.

**Why local models instead of an API?**
Financial data is sensitive. Sending SEC filings — even public ones — through a third-party API means that data passes through someone else's infrastructure and may be used for training. Running everything locally means nothing leaves the machine. It's also free after the initial setup: no per-token costs, no rate limits, no API keys to manage.

**Why Phi-3.5-mini over a larger model?**
Phi-3.5-mini (3.8B parameters) was specifically trained to punch above its weight on reasoning and instruction-following tasks. For this use case — reading provided excerpts and answering questions about them — it performs close to models twice its size. If you have a capable GPU and want higher quality, swap `GENERATION_MODEL` in `query.py` to `mistralai/Mistral-7B-Instruct-v0.3`. The rest of the code doesn't change.

**Why ChromaDB over FAISS or Pinecone?**
FAISS is faster but has no built-in persistence or metadata filtering — you'd need to write both yourself, which adds complexity for no benefit at this scale. Pinecone is a managed cloud service, which contradicts the local-first goal. ChromaDB is embedded, handles persistence automatically, and supports filtered search out of the box. For a single-user local tool it's the right fit.

**Why chunk size 800 with 100 overlap?**
800 characters is roughly two or three substantive sentences — large enough to contain a complete thought, small enough to stay topically focused. If chunks are too large, one chunk might cover risk factors and financial results and be a mediocre match for both. If they're too small, you lose the surrounding context that makes a sentence interpretable. 100 characters of overlap (~12% of chunk size) prevents meaningful sentences from being split across chunks without inflating the total count significantly.

**Why uv for package management?**
`uv` resolves and installs dependencies dramatically faster than pip, enforces reproducibility through `uv.lock`, and handles virtual environments automatically. The `uv run` command means you never need to remember to activate an environment — it just works. `uv.lock` is committed to version control so anyone who clones this repo gets the exact same package versions.

---

## Where to go from here

A few natural extensions if you want to take this further:

- **More filing types** — `sec-edgar-downloader` supports 8-K (material events), DEF 14A (proxy statements with executive compensation), S-1 (IPO filings), and many others. The cleaning pipeline would need adjustments for each format.
- **Persistent conversation** — the current query loop is stateless; each question is answered independently with no memory of what came before. Adding conversation history to the model prompt would let you ask follow-up questions naturally.
- **Better metadata filtering** — you could filter by filing year, company sector, or any other structured field. The metadata is already stored on every chunk; it's just a matter of exposing the filters in the query interface.
- **Web interface** — a Gradio or Streamlit wrapper would make this accessible without a terminal, and wouldn't require changing any of the underlying logic.
- **Re-ranking** — after the initial retrieval, a cross-encoder model can re-score the top-k results for relevance before passing them to the generator. This often improves answer quality at the cost of a bit more latency.


Steps:
1.	Raw download
2.	Clean & normalize
3.	Chunking
4.	Embedding
5.	Vector DB
6.	Basic retrieval
7.	Relevance labeling
8.	Retriever training
9.	RAG prompt + LLM
10.	Prompt engineering
11.	Evaluation & metrics
12.	API/UI
13.	Docs + README
14.	Optional: Hybrid search, re‑ranking, fine‑tuning

