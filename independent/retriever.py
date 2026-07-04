import datetime

from langchain.tools import tool
from langchain.agents import create_agent
from helper import vector_store, model
from logging_config import get_logger

logger = get_logger(__name__)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    logger.info("Retrieving context for query: %r", query)
    retrieved_docs = vector_store.similarity_search(query, k=2)
    logger.info(
        "Retrieved %d chunk(s): %s",
        len(retrieved_docs),
        [doc.metadata.get("source") for doc in retrieved_docs],
    )
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


tools = [retrieve_context]

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
agent = create_agent(model, tools, system_prompt=SYSTEM_PROMPT)
logger.info("Agent created with %d tool(s): %s", len(tools), [t.name for t in tools])
