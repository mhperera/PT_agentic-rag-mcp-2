from fastmcp import FastMCP
from core.db_connector import get_session
import traceback
from sqlalchemy import text

mcp = FastMCP("SQL Server")


@mcp.tool(
    name="query_database",
    description="Run SQL SELECT queries on the customer database. Only SELECT allowed.",
)
def query_database(query: str) -> list:

    print("🔥 SQL tool called with raw query param : " + query)

    if not query.strip().lower().startswith("select"):
        return ["Only SELECT queries allowed."]

    try:
        session = get_session()
        result = session.execute(text(query.query))
        rows = result.mappings().all()
        return rows
    except Exception as e:
        print("❌ Query execution error:", e)
        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
