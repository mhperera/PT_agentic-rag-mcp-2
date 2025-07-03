from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate
from core.db_connector import get_session, get_engine
from sqlalchemy import text

from core.enums.ToolName import ToolName
from core.db_connector import get_db_schema, format_schema, get_db_schema_as_ddl
from core.enums.ClassifierLabel import ClassifierLabel
from init import llm
from config.tool_agent_map import TOOL_AGENT_MAP, get_tools


rephrase_parser = StrOutputParser()
sql_parser = StrOutputParser()

class State(TypedDict):
        messages: Annotated[list, add_messages]
        schema: str
        sql_query: str
        db_result: str
        
def build_agents(tools, tool_map) -> dict:

    # Node: Call vector_table_search : Dynamic table description selection
    async def node_vector_table_search(state: State):
        question = state["messages"][-1].content
        tool = tool_map[ToolName.VECTOR_TABLE_SEARCH.value]
        output = await tool.ainvoke({"query": question, "top_k": 5})
        return {"table_info": output, "messages": state["messages"]}

    # Node: Call generate_sql_query : Generate the SQL query
    async def node_generate_sql_query(state: State):

        sql_prompt = ChatPromptTemplate.from_template(
            """
            You are a SQL expert. Given the database schema and a user question, generate a safe SQL SELECT query.\n
            Schema:{schema}\n
            Question:{question}\n
            SQL Query:
            """
        )
        question = state["messages"][-1].content
        schema = get_db_schema_as_ddl()
        chain = sql_prompt | llm | sql_parser
        sql_query = await chain.ainvoke({"schema": schema, "question": question})
        return {"sql_query": sql_query, "schema": schema, "messages": state["messages"]}

        # question = state["messages"][-1].content
        # schema = format_schema(get_db_schema())
        # tool = tool_map[ToolName.GENERATE_SQL_QUERY.value]
        # response = await tool.ainvoke({"question": question, "schema": schema})
        # return {"sql_query": response, "messages": state["messages"]}

        # question = state["messages"][-1].content
        # schema = get_db_schema()
        # table_info = state.get("table_info", [])
        # table_desc = "\n".join(table_info)
        # input_prompt = f"{schema}\n\n{table_desc}"
        # tool = tool_map[ToolName.GENERATE_SQL_QUERY.value]
        # output = await tool.ainvoke({"question": question, "schema": input_prompt})
        # return {"sql_query": output, "messages": state["messages"]}

    # Node: Call execute_sql_query - Execute the SQL Query
    async def node_execute_sql_query(state: State):
        sql_query = state.get("sql_query", "")
        try:
            session = get_session()
            result = session.execute(text(sql_query)).fetchall()
            # result = session.execute(text(sql_query)).mappings().all()
            result_str = str(result)
        except Exception as e:
            result_str = f"❌ Error: {str(e)}"
        return {"db_result": result_str, "messages": state["messages"]}
        # sql_query = state.get("sql_query", "")
        # tool = tool_map[ToolName.EXECUTE_SQL_QUERY.value]
        # output = await tool.ainvoke({"sql_query": sql_query})
        # final_response = str(output)
        # state["messages"].append({"role": "assistant", "content": final_response})
        # return {"messages": state["messages"]}

    async def node_rephrase_result(state: State):

        rephrase_prompt = ChatPromptTemplate.from_template(
            """
            You are an assistant summarizing llm results for the end users.\n
            User Question:{question}\n
            Raw Data:{data}\n
            Final Answer:
            """
        )

        question = state["messages"][-1].content
        data = state["db_result"]
        chain = rephrase_prompt | llm | rephrase_parser
        response = await chain.ainvoke({"question": question, "data": data})
        return {"messages": state["messages"] + [AIMessage(content=response)]}

    # Node: No tools wil be called
    async def node_general_llm(state: State):
        response = await llm.ainvoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    def general_llm_graph():
        graph = StateGraph(State)
        graph.add_node("node_general_llm", node_general_llm)
        graph.set_entry_point("node_general_llm")
        graph.set_finish_point("node_general_llm")
        return graph.compile()

    def db_search_graph():
        
        # tools_for_this_agent = get_tools(
        #     tool_map=tool_map, label=ClassifierLabel.DB_SEARCH
        # )

        graph = StateGraph(State)
        graph.add_node("node_generate_sql_query", node_generate_sql_query)
        graph.add_node("node_execute_sql_query", node_execute_sql_query)
        graph.add_node("node_rephrase_result", node_rephrase_result)
        graph.set_entry_point("node_generate_sql_query")
        graph.add_edge("node_generate_sql_query", "node_execute_sql_query")
        graph.add_edge("node_execute_sql_query", "node_rephrase_result")
        graph.set_finish_point("node_rephrase_result")
        return graph.compile()

        # graph = StateGraph(State)
        # graph.add_node("node_general_llm", node_general_llm)
        # graph.add_node("node_vector_table_search", node_vector_table_search)
        # graph.add_node("node_generate_sql_query", node_generate_sql_query)
        # graph.add_node("node_execute_sql_query", node_execute_sql_query)
        # graph.set_entry_point("node_vector_table_search")
        # graph.add_edge("node_vector_table_search", "node_generate_sql_query")
        # graph.add_edge("node_generate_sql_query", "node_execute_sql_query")
        # graph.add_edge("node_execute_sql_query", "node_general_llm")
        # graph.set_finish_point("node_general_llm")
        # return graph.compile()

    def vector_search_graph():
        graph = StateGraph(State)
        graph.add_node("node_general_llm", node_general_llm)
        graph.add_node(
            "vector_knowledge_search",
            ToolNode([tool_map[ToolName.VECTOR_KNOWLEDGE_SEARCH.value]]),
        )
        graph.set_entry_point("vector_knowledge_search")
        graph.add_edge("vector_knowledge_search", "node_general_llm")
        graph.set_finish_point("node_general_llm")
        return graph.compile()

    def other_tool_graph():
        graph = StateGraph(State)
        graph.add_node("node_general_llm", node_general_llm)
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "node_general_llm")
        graph.add_conditional_edges("node_general_llm", tools_condition)
        graph.add_edge("tools", "node_general_llm")
        graph.set_finish_point("node_general_llm")
        return graph.compile()

    return {
        f"{ClassifierLabel.GENERAL_LLM.value}_agent": general_llm_graph(),
        f"{ClassifierLabel.DB_SEARCH.value}_agent": db_search_graph(),
        f"{ClassifierLabel.VECTOR_SEARCH.value}_agent": vector_search_graph(),
        f"{ClassifierLabel.OTHER_TOOL.value}_agent": other_tool_graph(),
    }
