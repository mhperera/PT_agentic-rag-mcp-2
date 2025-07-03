from langchain_tavily import TavilySearch
from fastmcp import FastMCP
from langsmith import traceable
from core.env_loader import TAVILY_API_KEY

mcp = FastMCP("Internet Search Server")

tavily_search = TavilySearch(api_key=TAVILY_API_KEY)


@mcp.tool(
    name="browse_tavily",
    description="Use Tavily to browse the internet and answer general knowledge or recent questions.",
)
@traceable(name="Tool: Browse Tavily")
async def browse_tavily(question: str) -> str:
    try:
        print("🔍 Searching Tavily for:", question)
        response = tavily_search.invoke({"query": question})

        if not response or not response.get("results"):
            return "No results found."

        top_result = response["results"][0]
        title = top_result.get("title", "Untitled")
        snippet = top_result.get("content", "No summary available")
        url = top_result.get("url", "")

        return f"**{title}**\n\n{snippet}\n\n🔗 {url}"

    except Exception as e:
        return f"Error searching the web: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8004)
