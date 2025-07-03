from init import bootstrap
from app.rag_client import rag_cli
import asyncio

if __name__ == "__main__":
    bootstrap()
    asyncio.run(rag_cli())
