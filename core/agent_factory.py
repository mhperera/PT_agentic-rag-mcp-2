from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from core.enums.ToolName import ToolName
from core.db_connector import get_db_schema_as_ddl
from core.enums.ClassifierLabel import ClassifierLabel
from init import llm
from langsmith import traceable

parser = StrOutputParser()


def build_agents(tools_map) -> dict:

    def general_llm_graph():
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        @traceable(name="Node: General LLM Response")
        async def node_general_llm(state: State):
            response = await llm.ainvoke(state["messages"])
            return {"messages": state["messages"] + [response]}

        graph = StateGraph(State)
        graph.add_node("node_general_llm", node_general_llm)
        graph.set_entry_point("node_general_llm")
        graph.set_finish_point("node_general_llm")
        return graph.compile()

    def db_search_graph():
        class State(TypedDict):
            messages: Annotated[list, add_messages]
            schema: str
            sql_query: str
            result: str

        @traceable(name="Node: Generate SQL Query")
        async def node_generate_sql_query(state: State):
            schema = get_db_schema_as_ddl()
            question = state["messages"][-1].content
            tool = tools_map[ToolName.GENERATE_SQL_QUERY.value]
            sql_query = await tool.ainvoke({"schema": schema, "question": question})
            return {
                "sql_query": sql_query,
                "schema": schema,
                "messages": state["messages"],
            }

        @traceable(name="Node: Execute SQL Query")
        async def node_execute_sql_query(state: State):
            sql_query = state["sql_query"]
            tool = tools_map[ToolName.EXECUTE_SQL_QUERY.value]
            result = await tool.ainvoke({"sql_query": sql_query})
            return {"result": result, "messages": state["messages"]}

        @traceable(name="Node: Rephrase SQL Result")
        async def node_rephrase_result(state: State):
            question = state["messages"][-1].content
            result = state["result"]
            tool = tools_map[ToolName.REPHRASE_RESULT.value]
            final_answer = await tool.ainvoke({"question": question, "result": result})
            new_messages = state["messages"] + [AIMessage(content=final_answer)]
            return {"messages": new_messages}

        graph = StateGraph(State)
        graph.add_node("generate_sql_query", node_generate_sql_query)
        graph.add_node("execute_sql_query", node_execute_sql_query)
        graph.add_node("rephrase_result", node_rephrase_result)
        graph.set_entry_point("generate_sql_query")
        graph.add_edge("generate_sql_query", "execute_sql_query")
        graph.add_edge("execute_sql_query", "rephrase_result")
        graph.set_finish_point("rephrase_result")
        return graph.compile()

    def other_tool_graph():
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        tools = [
            tools_map[ToolName.MATH_ADD.value],
            tools_map[ToolName.MATH_DIVIDE.value],
            tools_map[ToolName.MATH_MULTIPLY.value],
            tools_map[ToolName.GET_WEATHER.value],
        ]

        llm_with_tools = llm.bind_tools(tools)

        @traceable(name="Node: Prepare Tool Question")
        async def node_prepare_question(state: State):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        @traceable(name="Node: Rephrase Tool Result")
        async def node_rephrase_result(state: State):
            question = next(
                (
                    msg.content
                    for msg in state["messages"]
                    if isinstance(msg, HumanMessage)
                ),
                None,
            )
            result = state["messages"][-1].content
            tool = tools_map[ToolName.REPHRASE_RESULT.value]
            final_answer = await tool.ainvoke({"question": question, "result": result})
            new_messages = state["messages"] + [AIMessage(content=final_answer)]
            return {"messages": new_messages}

        graph = StateGraph(State)
        graph.add_node("prepare_question", node_prepare_question)
        graph.add_node("tools", ToolNode(tools))
        graph.add_node("rephrase_result", node_rephrase_result)
        graph.set_entry_point("prepare_question")
        graph.add_conditional_edges("prepare_question", tools_condition)
        graph.add_conditional_edges("tools", tools_condition)
        graph.add_edge("tools", "rephrase_result")
        graph.set_finish_point("rephrase_result")
        return graph.compile()

    return {
        f"{ClassifierLabel.GENERAL_LLM.value}_agent": general_llm_graph(),
        f"{ClassifierLabel.DB_SEARCH.value}_agent": db_search_graph(),
        f"{ClassifierLabel.OTHER_TOOL.value}_agent": other_tool_graph(),
        # f"{ClassifierLabel.VECTOR_SEARCH.value}_agent": vector_search_graph(),
    }
