from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from core.enums.ToolName import ToolName
from core.db_connector import get_db_schema


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_agents(llm_with_tools, tools, tool_map) -> dict:

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

    return {
        "db_agent": build_db_search_graph(),
        "vector_agent": build_vector_search_graph(),
        "llm_agent": build_general_llm_graph(),
        "other_tool_agent": build_other_tool_graph(),
    }
