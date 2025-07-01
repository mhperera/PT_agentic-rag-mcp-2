from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

import os
import asyncio
from core.tool_loader import load_tool_config
from core.db_connector import init_db_engine, get_db_schema
from core.vector_db import init_vector_db
from dotenv import load_dotenv

load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

cohere_api_key = os.getenv("COHERE_API_KEY")
if cohere_api_key:
    os.environ["COHERE_API_KEY"] = cohere_api_key


def bootstrap():
    """
    Initialize all resources needed before starting the RAG app.
    - Initialize DB connection
    - Load schemas
    - Load metadata
    if needed
    """

    try:
        print("\n🕞 Bootstrapping...")
        init_db_engine(pool_size=10, max_overflow=10)
        init_vector_db(
            file_paths=[
                "config/table_descriptions.yaml",
                "resources/data.csv",
                "resources/data.txt",
            ],
            cohere_api_key=cohere_api_key,
        )
        get_db_schema()
        print("✅ Bootstrap complete.")
    except Exception as e:
        print("❌ Exception:", e)


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
    llm = ChatGroq(model="llama3-8b-8192")
    # llm = ChatGroq(model="llama-3.3-70b-versatile")

    llm_with_tools = llm.bind_tools(tools)

    ## Define Graph State
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def tool_calling_llm(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph = StateGraph(State)
    graph.add_node("tool_calling_llm", tool_calling_llm)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "tool_calling_llm")
    graph.add_conditional_edges("tool_calling_llm", tools_condition)
    graph.add_edge("tools", "tool_calling_llm")
    agent = graph.compile()

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
        print("\n❌ Agent invocation failed: ", e)


if __name__ == "__main__":
    asyncio.run(main())
