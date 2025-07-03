from fastmcp import FastMCP
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from init import llm
import traceback

parser = StrOutputParser()

mcp = FastMCP("Rephraser Server")


@mcp.tool(
    name="rephrase_result",
    description="Given a user's question and the raw output from a tool, return a natural, human-friendly response that summarizes the result clearly.",
)
async def rephrase_result(question: str, result: str) -> str:
    try:
        prompt = ChatPromptTemplate.from_template(
            """
            You are an assistant summarizing llm results for the end users.\n
            User Question:{question}\n
            Raw Data:{result}\n
            Final Answer:
        """
        )
        chain = prompt | llm | parser
        return await chain.ainvoke({"question": question, "result": result})
    except Exception as e:
        print("❌ Query execution error:", e)
        traceback.print_exc()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8003)
