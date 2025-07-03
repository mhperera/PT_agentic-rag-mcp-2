from fastmcp import FastMCP
from langsmith import traceable

mcp = FastMCP("Math Server")


@mcp.tool(
    name="math_add",
    description="Adds two numbers and returns the result. Use this for simple or complex addition operations.",
)
@traceable(name="Tool: Add")
def add(a: int, b: int) -> int:
    return a + b


@mcp.tool(
    name="math_multiply",
    description="Multiplies two numbers and returns the result. Use for any arithmetic requiring product calculation.",
)
@traceable(name="Tool: Multiply")
def multiply(a: int, b: int) -> int:
    return a * b


@mcp.tool(
    name="math_divide",
    description="Divides the first number by the second and returns the result. Do not use if the second number is zero.",
)
@traceable(name="Tool: Divide")
def divide(a: int, b: int) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


if __name__ == "__main__":
    mcp.run(transport="stdio")
