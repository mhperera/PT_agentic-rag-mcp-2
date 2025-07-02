from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langgraph.prebuilt import create_react_agent
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, Messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.messages import AIMessage

import re
import os
import asyncio
from core.tool_loader import load_tool_config
from core.db_connector import init_db_engine, get_db_schema
from core.vector_db import init_vector_db
from core.prompts.classifiers import get_classify_intent_prompt
from core.enums.IntentLabel import IntentLabel
from core.enums.ToolName import ToolName
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

cohere_api_key = os.getenv("COHERE_API_KEY")
if cohere_api_key:
    os.environ["COHERE_API_KEY"] = cohere_api_key

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
if langchain_tracing_v2:
    os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing_v2


llm = ChatGroq(model="qwen-qwq-32b")
# llm = ChatGroq(model="llama3-8b-8192")
# llm = ChatGroq(model="llama-3.3-70b-versatile")
# llm = ChatGroq(model="mistral-saba-24b")
# llm = ChatCohere(
#     model_name="xlarge",
#     temperature=0.5,
#     max_tokens=512,
# )


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
            ],
            cohere_api_key=cohere_api_key,
            output_file="table_descriptions",
        )
        init_vector_db(
            file_paths=[
                "resources/data.csv",
                "resources/data.txt",
            ],
            cohere_api_key=cohere_api_key,
        )
        get_db_schema()
        print("✅ Bootstrap successfully completed.")
    except Exception as e:
        print("❌ Exception:", e)


# Node: Classifier
async def question_classifer(question: str) -> str:
    prompt = get_classify_intent_prompt().format_messages(question=question)
    response = await llm.ainvoke(prompt)
    label = response.content.strip().lower()

    # Remove anything between <think>...</think>, including the tags
    label = re.sub(
        r"<think>.*?</think>", "", response.content.strip().lower(), flags=re.DOTALL
    )

    if IntentLabel.DB_SEARCH.value in label:
        return IntentLabel.DB_SEARCH.value
    elif IntentLabel.VECTOR_SEARCH.value in label:
        return IntentLabel.VECTOR_SEARCH.value
    elif IntentLabel.INTERNET_SEARCH.value in label:
        return IntentLabel.INTERNET_SEARCH.value
    elif IntentLabel.OTHER_TOOL.value in label:
        return IntentLabel.OTHER_TOOL.value
    else:
        return IntentLabel.GENERAL_LLM.value


class State(TypedDict):
    messages: Annotated[list, add_messages]


async def main():

    tool_config = load_tool_config("config/tool_registry.yaml")
    client = MultiServerMCPClient(tool_config)
    tools = await client.get_tools()

    tool_map = {getattr(t, "name", f"unnamed_{i}"): t for i, t in enumerate(tools)}

    print("\n📚 Discovered tools:")
    for tool in tools:
        print("-", getattr(tool, "name", repr(tool)))

    if not tools:
        raise RuntimeError("No tools discovered. Check MCP servers and tool config.")

    llm_with_tools = llm.bind_tools(tools)

    # Node: Call vector_table_search : Dynamic table description selection
    async def node_vector_table_search(state: State):
        question = state["messages"][-1].content
        tool = tool_map[ToolName.VECTOR_TABLE_SEARCH.value]
        output = await tool.ainvoke({"query": question, "top_k": 5})
        return {"table_info": output, "messages": state["messages"]}

    # Node: Call generate_sql_query : Generate the SQL query
    async def node_generate_sql_query(state: State):
        question = state["messages"][-1].content
        schema = get_db_schema()
        table_info = state.get("table_info", [])
        table_desc = "\n".join(table_info)
        input_prompt = f"{schema}\n\n{table_desc}"
        tool = tool_map[ToolName.GENERATE_SQL_QUERY.value]
        output = await tool.ainvoke({"question": question, "schema": input_prompt})
        return {"sql_query": output, "messages": state["messages"]}

    # Node: Call query_database - Execute the SQL Query
    async def node_query_database(state: State):
        query = state.get("sql_query", "")
        tool = tool_map[ToolName.QUERY_DATABASE.value]
        output = await tool.ainvoke({"query": query})
        final_response = str(output)
        state["messages"].append({"role": "assistant", "content": final_response})
        return {"messages": state["messages"]}

    async def node_default_tool_call_llm(state: State):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    # graph = StateGraph(State)
    # graph.add_node("node_default_tool_call_llm", node_default_tool_call_llm)
    # graph.add_node("tools", ToolNode(tools))
    # graph.add_edge(START, "node_default_tool_call_llm")
    # graph.add_conditional_edges("node_default_tool_call_llm", tools_condition)
    # graph.add_edge("tools", "node_default_tool_call_llm")
    # agent = graph.compile()

    # graph = StateGraph(State)
    # graph.add_node("node_classify_intent", node_classify_intent)
    # graph.add_node("tools", ToolNode(tools))
    # graph.add_node("", node_default_tool_call_llm)
    # graph.add_edge(START, "node_default_tool_call_llm")
    # graph.add_conditional_edges("node_default_tool_call_llm", tools_condition)
    # graph.add_edge("tools", "node_default_tool_call_llm")
    # agent = graph.compile()

    def build_db_search_graph():
        graph = StateGraph(State)
        graph.add_node("node_default_tool_call_llm", node_default_tool_call_llm)
        graph.add_node("node_vector_table_search", node_vector_table_search)
        graph.add_node("node_generate_sql_query", node_generate_sql_query)
        graph.add_node("node_query_database", node_query_database)
        graph.set_entry_point("node_vector_table_search")
        graph.add_edge("node_vector_table_search", "node_generate_sql_query")
        graph.add_edge("node_generate_sql_query", "node_query_database")
        graph.add_edge("node_query_database", "node_default_tool_call_llm")
        graph.set_finish_point("node_default_tool_call_llm")
        return graph.compile()

    def build_vector_search_graph():
        graph = StateGraph(State)
        graph.add_node("node_default_tool_call_llm", node_default_tool_call_llm)
        graph.add_node(
            "vector_knowledge_search",
            ToolNode([tool_map[ToolName.VECTOR_KNOWLEDGE_SEARCH.value]]),
        )
        graph.set_entry_point("vector_knowledge_search")
        graph.add_edge("vector_knowledge_search", "node_default_tool_call_llm")
        graph.set_finish_point("node_default_tool_call_llm")
        return graph.compile()

    def build_general_llm_graph():
        graph = StateGraph(State)
        graph.add_node("node_default_tool_call_llm", node_default_tool_call_llm)
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "node_default_tool_call_llm")
        graph.add_conditional_edges("node_default_tool_call_llm", tools_condition)
        graph.add_edge("tools", "node_default_tool_call_llm")
        graph.set_finish_point("node_default_tool_call_llm")
        return graph.compile()

    def build_other_tool_graph():
        graph = StateGraph(State)
        graph.add_node("node_default_tool_call_llm", node_default_tool_call_llm)
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "node_default_tool_call_llm")
        graph.add_conditional_edges("node_default_tool_call_llm", tools_condition)
        graph.add_edge("tools", "node_default_tool_call_llm")
        graph.set_finish_point("node_default_tool_call_llm")
        return graph.compile()

    db_agent = build_db_search_graph()
    vector_agent = build_vector_search_graph()
    llm_agent = build_general_llm_graph()
    other_tool_agent = build_other_tool_graph()

    print("\n##### RAG APPLICATION #####")
    print("\nEnter your question (or 'exit' to quit)")

    try:
        while True:

            question = input("\n\n👩 Human : ").strip()
            if question.lower() == "exit" or question.lower() == "quit":
                print("Exiting...")
                break

            response = None

            classifier = await question_classifer(question) or "N/A"
            print(f"\n[Classifier] Intent detected ::::: {classifier}\n")

            messages = [{"role": "user", "content": question}]

            if classifier == IntentLabel.DB_SEARCH.value:
                response = await db_agent.ainvoke({"messages": messages})
            elif classifier == IntentLabel.VECTOR_SEARCH.value:
                response = await vector_agent.ainvoke({"messages": messages})
            elif classifier == IntentLabel.GENERAL_LLM.value:
                response = await llm_agent.ainvoke({"messages": messages})
            elif classifier == IntentLabel.OTHER_TOOL.value:
                response = await other_tool_agent.ainvoke({"messages": messages})
            else:
                response = await llm_agent.ainvoke({"messages": messages})

            last_message = (
                response["messages"][-1].content
                if isinstance(response["messages"][-1], dict)
                else response["messages"][-1].content
            )
            print("\n🤖 AI:", last_message)
    except Exception as e:
        print("\n❌ Agent invocation failed: ", e)


if __name__ == "__main__":
    bootstrap()
    asyncio.run(main())
