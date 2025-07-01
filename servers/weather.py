from fastmcp import FastMCP

mcp = FastMCP("Weather Server")


@mcp.tool(name="get_weather", description="Get the weather location")
async def get_weather(location: str) -> str:
    print("🔥 Tool get_weather called with location : " + location)
    return f"The weather in {location} is mostly sunny with light breeze"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
