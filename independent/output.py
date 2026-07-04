from retriever import agent
from logging_config import get_logger

logger = get_logger(__name__)

query = "What are the key cybersecurity risks mentioned in the 10-K filings for Apple (AAPL) and Microsoft (MSFT)? Please provide a summary of the risks for each company, along with any relevant details and excerpts from the filings."

logger.info("Running query: %r", query)

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": query}]},
    version="v3",
)
for kind, item in stream.interleave("messages", "tool_calls"):
    if kind == "messages":
        # Streamed answer tokens are the program's actual output, not a log
        # message, so they go straight to stdout rather than through logger.
        for token in item.text:
            print(token, end="", flush=True)
    elif kind == "tool_calls":
        logger.info("Tool call: %s(%s)", item.tool_name, item.input)
        logger.info("Tool result: %s", item.output)

final_state = stream.output
logger.info("Query complete")
