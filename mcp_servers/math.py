from fastmcp import FastMCP

mcp = FastMCP("Math Server")


@mcp.tool(
    name="math_add",
    description="Adds two numbers and returns the result. Use this for simple or complex addition operations.",
)
def add(a: int, b: int) -> int:
    print(f"Math add tool called with numbers: {a} and {b}")
    return a + b


@mcp.tool(
    name="math_multiply",
    description="Multiplies two numbers and returns the result. Use for any arithmetic requiring product calculation.",
)
def multiply(a: int, b: int) -> int:
    print(f"Math multiply tool called with numbers: {a} and {b}")
    return a * b


@mcp.tool(
    name="math_divide",
    description="Divides the first number by the second and returns the result. Do not use if the second number is zero.",
)
def divide(a: int, b: int) -> float:
    print(f"Math divide tool called with numbers: {a} and {b}")
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


if __name__ == "__main__":
    mcp.run(transport="stdio")
