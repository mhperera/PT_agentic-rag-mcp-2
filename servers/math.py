from fastmcp import FastMCP

mcp = FastMCP("Math Server")


@mcp.tool(name="add", description="Add two numbers")
def add(a: int, b: int) -> int:
    print("🔥 Math add tool called with numbers : " + a + " and " + b)
    return a + b


@mcp.tool(name="multiply", description="Multiply two numbers")
def multiply(a: int, b: int) -> int:
    print("🔥 Math multiply tool called with numbers : " + a + " and " + b)
    return a * b


@mcp.tool(name="divide", description="Devide first number from the seconds number")
def divide(a: int, b: int) -> float:
    print("🔥 Math divide tool called with numbers : " + a + " and " + b)
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


if __name__ == "__main__":
    mcp.run(transport="stdio")
