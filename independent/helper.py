import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()
logger.info("Loaded environment variables from .env")

CHROMA_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "chroma_10_K_filings"
)
logger.info("Chroma persist directory: %s", CHROMA_PERSIST_DIR)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)

logger.info("Loading embedding model sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = Chroma(
    collection_name="10_K_filings",
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIR,  # Where to save data locally, remove if not necessary
)
logger.info("Connected to Chroma collection '10_K_filings' at %s", CHROMA_PERSIST_DIR)

logger.info("Initializing chat model Qwen/Qwen2.5-7B-Instruct via HuggingFace endpoint")
model = init_chat_model(
    "Qwen/Qwen2.5-7B-Instruct",
    model_provider="huggingface",
    backend="endpoint",  # calls HF's hosted Inference API instead of downloading the model locally
    temperature=0.7,
    max_tokens=1024,
)
