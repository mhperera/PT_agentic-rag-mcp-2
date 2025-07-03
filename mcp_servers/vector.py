from fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

from init import embedding_model

load_dotenv()

VECTOR_STORE_PATH = "vector_db"

cohere_api_key = os.getenv("COHERE_API_KEY")
if cohere_api_key:
    os.environ["COHERE_API_KEY"] = cohere_api_key

mcp = FastMCP("Vector Server")

vectorstore = FAISS.load_local(
    VECTOR_STORE_PATH,
    embedding_model,
    allow_dangerous_deserialization=True,
)

vectorstore_tables = FAISS.load_local(
    VECTOR_STORE_PATH + "/table_info",
    embedding_model,
    allow_dangerous_deserialization=True,
)


@mcp.tool(
    name="vector_knowledge_search",
    description="Use this tool to retrieve relevant information from indexed documents using vector-based semantic search. "
    "It includes content on solar energy, database concepts, and table schemas. "
    "Ideal for answering domain-specific or technical knowledge questions before deciding on other actions like SQL query generation.",
)
def vector_knowledge_search(query: str, top_k: int = 5) -> list:
    print(f"🔍 Searching vector DB: {query}")
    try:
        results = vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]
    except Exception as e:
        return [f"❌ Error during vector search: {str(e)}"]


@mcp.tool(
    name="vector_table_search",
    description="Use this tool to understand the structure of the database. "
    "It performs semantic search over vectorized table descriptions to retrieve details "
    "about table names, column types, and relationships. Useful before generating SQL queries.",
)
def vector_table_search(query: str, top_k: int = 5) -> list:
    print(f"🔍 Searching vector DB Tables: {query}")
    try:
        results = vectorstore_tables.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]
    except Exception as e:
        return [f"❌ Error during vector search: {str(e)}"]


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8002)
