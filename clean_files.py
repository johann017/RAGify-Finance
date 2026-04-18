import unicodedata
import re
from bs4 import BeautifulSoup

# Sections that are pure boilerplate / exhibit lists — split text at first match
_BOILERPLATE_PATTERNS = [
    r"^\s*SIGNATURES?\s*$",
    r"^\s*EXHIBIT\s+INDEX\s*$",
    r"^\s*INDEX\s+TO\s+EXHIBITS\s*$",
]

# Matches Item headers only at the start of a line (with optional leading whitespace),
# using a zero-width lookahead so re.split doesn't consume the header text.
# Must be applied to text that still has newlines (before normalize_text) to
# distinguish real section headers from inline cross-references.
ITEM_HEADER = re.compile(r"(?=^\s*ITEM\s+\d+[A-Z]?\.\s+\S)", re.IGNORECASE | re.MULTILINE)


def normalize_text(text: str) -> str:
    """
    Normalize SEC text:
    - Convert unicode to closest ASCII equivalents
    - Remove control characters
    - Collapse whitespace
    """

    # 1) Unicode normalization (NFKD decomposes chars like smart quotes)
    text = unicodedata.normalize("NFKD", text)

    # 2) Replace common non-ASCII symbols with plain text
    replacements = {
        "\u2612": "X",  # ballot box with X
        "\u2610": "",  # empty ballot box
        "\u2014": " - ",  # em dash
        "\u2013": " - ",  # en dash
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u2026": "...",  # ellipsis
        "\u00ae": "(R)",  # registered trademark ®
        "\u2122": "(TM)",  # trademark ™
        "\u2022": "-",  # bullet point •
        "\u20ac": "EUR",  # euro sign €
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)

    # 3) Remove any remaining control chars (non-printable)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

    # 4) Collapse multiple whitespace into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_filing_metadata(raw: str) -> dict:
    """Extract structured metadata from the SEC SGML header block."""
    fields = {
        "conformed_period": r"CONFORMED PERIOD OF REPORT:\s*(\S+)",
        "filed_date": r"FILED AS OF DATE:\s*(\S+)",
        "company_name": r"COMPANY CONFORMED NAME:\s*(.+)",
        "cik": r"CENTRAL INDEX KEY:\s*(\S+)",
        "form_type": r"CONFORMED SUBMISSION TYPE:\s*(\S+)",
    }
    meta = {}
    for key, pattern in fields.items():
        m = re.search(pattern, raw, re.IGNORECASE)
        meta[key] = m.group(1).strip() if m else None
    return meta


def extract_primary_document(raw: str) -> str:
    """
    Extract only the primary 10-K document block from the SGML bundle,
    skipping exhibits, signatures, and other attached documents.
    """
    doc_pattern = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL | re.IGNORECASE)
    for match in doc_pattern.finditer(raw):
        block = match.group(1)
        type_match = re.search(r"<TYPE>([^\n<]+)", block, re.IGNORECASE)
        if type_match and type_match.group(1).strip().upper() in ("10-K", "10-K405"):
            return block
    # Fallback: return the full text if no typed document block found
    return raw


def strip_toc(text: str) -> str:
    """Remove the Table of Contents block (Item entries with trailing page numbers)."""
    toc_pattern = re.compile(
        r"((?:ITEM\s+\d+[A-Z]?\.\s+[^\n]+?\.{3,}\s*\d+\s*\n)+)",
        re.IGNORECASE,
    )
    return toc_pattern.sub("", text)


def strip_boilerplate(text: str) -> str:
    """Truncate text at well-known boilerplate section headers. These sections typically contain no substantive content relevant to financial analysis.

    SIGNATURES — just legal sign-off lines (names, titles, dates of officers signing the document). No financial or business content.
    EXHIBIT INDEX / INDEX TO EXHIBITS — a list of attached file names and exhibit numbers (e.g. "Exhibit 21.1 — Subsidiaries of the Registrant"). The actual exhibit content is in separate <DOCUMENT> blocks, which extract_primary_document already excluded.

    Uses the LAST occurrence of each pattern, since earlier occurrences are often
    TOC or cover-page entries (especially in iXBRL filings) not the real sections.
    """
    for pattern in _BOILERPLATE_PATTERNS:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
        if matches:
            text = text[: matches[-1].start()]
    return text


def remove_table_noise(text: str) -> str:
    """
    Remove lines that are pure formatting artifacts (e.g. page numbers, blank
    separator rows) while preserving financial statement rows which contain
    labeled figures needed to answer performance questions.

    A line is dropped only if it has no alphabetic characters at all — meaning
    it carries zero semantic context (e.g. "   42   ", "......", "- - -").
    Lines like "Net revenue  94,680  91,154" are kept because they have labels.
    """
    lines = text.splitlines()
    cleaned = [line for line in lines if re.search(r"[A-Za-z]", line)]
    return "\n".join(cleaned)


def deduplicate_sentences(text: str) -> str:
    """Remove verbatim duplicate sentences, common in SEC risk factor sections."""
    seen = set()
    result = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        key = sentence.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(sentence)
    return " ".join(result)


def split_by_items(text: str) -> list:
    """
    Split filing text on SEC Item headers (e.g. 'Item 1A. Risk Factors').

    Must receive pre-normalized text (newlines intact). ITEM_HEADER uses ^ with
    re.MULTILINE to anchor matches to line starts — this prevents splitting on
    inline cross-references like "see Item 1A for details". Final normalization
    collapses newlines to spaces, which would break that anchor, so this function
    must run before normalize_text. Each section is deduplicated and normalized here.
    Returns sections with at least 200 characters.
    """
    parts = ITEM_HEADER.split(text)
    sections = []
    for part in parts:
        deduped = deduplicate_sentences(part)
        normalized = normalize_text(deduped)
        if len(normalized) > 200:
            sections.append(normalized)
    return sections


def extract_text_from_raw(raw_str: str) -> tuple:
    """
    Extracts, cleans, and normalizes the narrative text from a raw SEC EDGAR filing.

    Returns:
        (cleaned_text: str, metadata: dict, pre_norm_text: str)
        pre_norm_text retains newlines for accurate section splitting via split_by_items.
    """
    # 1) Pull structured metadata from the SGML header
    metadata = extract_filing_metadata(raw_str)

    # 2) Isolate the primary 10-K document block (skip exhibits)
    primary = extract_primary_document(raw_str)

    # 3) Strip the <TEXT> tag — leave HTML/SGML content intact for BS4 to parse
    text_blocks = re.split(r"<TEXT>", primary, flags=re.IGNORECASE)
    combined = text_blocks[1] if len(text_blocks) > 1 else primary

    # 4) Parse HTML/SGML — BS4 strips all tags and returns text nodes with newlines
    #    between block-level elements, preserving line structure for Item splitting.
    soup = BeautifulSoup(combined, "lxml")
    # Remove iXBRL header block present in modern Inline XBRL filings (2017+).
    # It contains only XBRL metadata (context IDs, dimensions, labels) — no
    # readable narrative — but get_text() would include it as garbage text.
    for tag in soup.find_all(["ix:header", "xbrli:xbrl"]):
        tag.decompose()
    cleaned = soup.get_text(separator="\n")

    # 5) Remove Table of Contents entries (Item lines with trailing dots + page numbers)
    cleaned = strip_toc(cleaned)

    # 6) Remove boilerplate sections (signatures, exhibit index)
    cleaned = strip_boilerplate(cleaned)

    # 7) Remove financial table rows (noise for embeddings)
    cleaned = remove_table_noise(cleaned)

    # 8) Return two versions of the text alongside metadata:
    #    - normalized: whitespace collapsed to single spaces, used as the stored document content.
    #    - pre_norm (cleaned): newlines still intact, passed to split_by_items so its
    #      MULTILINE ^ anchors can detect real line-start Item headers before normalization
    #      destroys the line boundaries. Each section is normalized inside split_by_items.
    return normalize_text(cleaned), metadata, cleaned
