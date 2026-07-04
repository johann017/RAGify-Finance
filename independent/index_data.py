from load_docs import get_docs
from helper import text_splitter, vector_store
from logging_config import get_logger

logger = get_logger(__name__)

docs = get_docs()
all_splits = text_splitter.split_documents(docs)
logger.info("Split %d document(s) into %d sub-document chunks", len(docs), len(all_splits))

document_ids = vector_store.add_documents(documents=all_splits)
logger.info("Indexed %d chunk(s) into Chroma; first ids: %s", len(document_ids), document_ids[:3])
