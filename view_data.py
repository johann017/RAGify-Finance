from langchain_chroma import Chroma

# Load the stored Chroma DB
vector_db = Chroma(persist_directory="chroma_finance_db", embedding_function=None)

# Fetch items from the vector store
data = vector_db.get()

ids = data["ids"]
docs = data["documents"]
meta = data["metadatas"]

print(f"Total chunks stored: {len(docs)}\n")

# Print preview of the first few chunks
# for i in range(min(10, len(docs))):
#     print(f"--- CHUNK {i+1} ID: {ids[i]} ---")
#     print("Source:", meta[i])
#     print(docs[i][:400], "...\n")

results = vector_db.as_retriever(
    search_kwargs={"k": 10, "filter": {"ticker": "AAPL"}}
).invoke("cybersecurity risk")

print("Results: ", results)
