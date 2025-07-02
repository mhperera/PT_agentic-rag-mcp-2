import subprocess

servers = [
    ("Math Server", "uv run servers/math_search.py"),
    ("SQL Server", "python -m servers.sql_search"),
    ("Vector Server", "uv run servers/vector_search.py"),
    ("Weather Server", "uv run servers/weather_search.py"),
]

for name, cmd in servers:
    print(f"Starting {name}...")
    subprocess.Popen(cmd, shell=True)
