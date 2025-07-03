import subprocess

servers = [
    ("Math", "uv run mcp_servers/math.py"),
    ("SQL", "python -m mcp_servers.sql"),
    ("Vector", "python -m mcp_servers.vector"),
    ("Weather", "uv run mcp_servers/weather.py"),
]

print(f"🚀 Starting all MCP servers...")

for name, cmd in servers:
    print(f" - Starting {name} Server...")
    subprocess.Popen(cmd, shell=True)

print(f"✅ Started all servers.")
