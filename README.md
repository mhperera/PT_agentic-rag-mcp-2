# Agentic RAG Application

## Project Overview

This project implements a modular Agentic RAG application using multiple MCP tool servers. The architecture separates tool servers and the client application for better scalability and maintainability.

---

## How to Run

### Start the MCP Tool Servers

Run the following command to launch all necessary backend tool servers (SQL, vector DB, math, weather, etc.):

```bash
uv start_servers.py
```

### Start the MCP Tool Servers

Once the servers are up, run the client application:

```bash
uv main.py
```
