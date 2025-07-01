from fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Config
VECTOR_STORE_PATH = "vector_db"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Init MCP
mcp = FastMCP("Vector Search Server")

# Load FAISS + Embeddings
embeddings = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY, model="embed-english-light-v3.0"
)

vectorstore_tables = FAISS.load_local(
    VECTOR_STORE_PATH + "/table_descriptions",
    embeddings,
    allow_dangerous_deserialization=True,
)
vectorstore_data = FAISS.load_local(
    VECTOR_STORE_PATH + "/combined_embeddings",
    embeddings,
    allow_dangerous_deserialization=True,
)


@mcp.tool(
    name="vector_data_search",
    description="Performs vector-based semantic search on indexed documents using FAISS. Include information about solar Energy",
)
def vector_data_search(query: str, top_k: int = 5) -> list:
    print(f"🔍 Searching vector DB for Data: {query}")
    try:
        results = vectorstore_data.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]
    except Exception as e:
        return [f"❌ Error during vector search: {str(e)}"]


@mcp.tool(
    name="vector_table_search",
    description="Performs vector-based semantic search on indexed table description using FAISS",
)
def vector_table_search(query: str, top_k: int = 5) -> list:
    print(f"🔍 Searching vector DB for Tables: {query}")
    try:
        results = vectorstore_tables.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]
    except Exception as e:
        return [f"❌ Error during vector search: {str(e)}"]


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8002)
