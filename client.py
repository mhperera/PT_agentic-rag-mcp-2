from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

import os
import asyncio
from core.tool_loader import load_tool_config
from core.db_connector import init_db_engine
from dotenv import load_dotenv

load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key


def bootstrap():
    """
    Initialize all resources needed before starting the RAG app.
    - Initialize DB connection pool
    - Load schemas or metadata if needed
    """

    try:
        print("\n🕞 Bootstrapping...")
        init_db_engine(pool_size=10, max_overflow=10)
        print("✅ Bootstrap complete.")
    except Exception as e:
        print("❌ Exception:", e)

    # Optional: load schema info here and cache it for tools or LLM prompt context
    # schemas = load_schemas()
    # return schemas


async def main():
    bootstrap()
    tool_config = load_tool_config("config/tool_registry.yaml")
    client = MultiServerMCPClient(tool_config)
    tools = await client.get_tools()

    print("\n📚 Discovered tools:")
    for tool in tools:
        print("-", getattr(tool, "name", repr(tool)))

    if not tools:
        raise RuntimeError("No tools discovered. Check MCP servers and tool config.")

    # llm = ChatGroq(model="qwen-qwq-32b")
    # llm = ChatGroq(model="llama3-8b-8192")
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    agent = create_react_agent(llm, tools)

    print("\n##### RAG APPLICATION #####")
    print("\nEnter your question (or 'exit' to quit)")
    
    try:
        while True:
            
            question = input("\n\nHuman : ").strip()
            if question.lower() == "exit" or question.lower() == "quit":
                print("Exiting...")
                break

            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            print("\n✅ Response:\n" + response["messages"][-1].content)
    except Exception as e:
        print("\n❌ Agent invocation failed", e)


if __name__ == "__main__":
    asyncio.run(main())
