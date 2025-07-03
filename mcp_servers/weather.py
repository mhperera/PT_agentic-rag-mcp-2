from fastmcp import FastMCP
from langsmith import traceable

mcp = FastMCP("Weather Server")


@mcp.tool(name="get_weather", description="Get the weather location")
@traceable(name="Tool: Get Weather")
async def get_weather(location: str) -> str:
    return f"The weather in {location} is mostly sunny with light breeze"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
