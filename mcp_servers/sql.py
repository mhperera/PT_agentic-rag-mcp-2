from fastmcp import FastMCP
from langchain.prompts import ChatPromptTemplate
from core.db_connector import get_session, get_engine
import traceback
from sqlalchemy import text
from langchain_core.output_parsers import StrOutputParser
from init import llm

parser = StrOutputParser()

mcp = FastMCP("SQL Server")


@mcp.tool(
    name="generate_sql_query",
    description="Given a user question, table description and the database schema, return a safe SQL SELECT query to answer it.",
)
async def generate_sql_query_(schema: str, question: str) -> str:
    """ "
    The LLM uses this tool to Generate SQL SELECT query based on the provided database schema and user question.
    """
    try:
        prompt = ChatPromptTemplate.from_template(
            """
            You are a SQL and MySQL expert.\n 
            Given the database schema(Database tables) and a user question, generate a safe SQL SELECT query.\n
            Write ONLY the SQL query (no markdown, no backticks) to answer the following question.\n
            Schema:{schema}\n
            Question:{question}\n
            SQL Query:
            """
        )
        chain = prompt | llm | parser
        return await chain.ainvoke({"schema": schema, "question": question})
    except Exception as e:
        print("❌ Query generation error:", e)
        traceback.print_exc()


@mcp.tool(
    name="execute_sql_query",
    description="Execute given SQL queries on the database and and returns result string",
)
def execute_sql_query(sql_query: str) -> list:
    try:
        session = get_session()
        result = session.execute(text(sql_query)).fetchall()
        return str(result)
    except Exception as e:
        print("❌ Query execution error:", e)
        traceback.print_exc()
    finally:
        session.close()


@mcp.tool(
    name="rephrase_sql_result",
    description="Rephrase raw SQL output into human-readable answer.",
)
async def rephrase_sql_result(question: str, db_result: str) -> str:
    try:
        prompt = ChatPromptTemplate.from_template(
            """
            You are an assistant summarizing llm results for the end users.\n
            User Question:{question}\n
            Raw Data:{db_result}\n
            Final Answer:
        """
        )
        chain = prompt | llm | parser
        return await chain.ainvoke({"question": question, "db_result": db_result})
    except Exception as e:
        print("❌ Query execution error:", e)
        traceback.print_exc()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
