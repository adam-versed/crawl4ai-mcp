# Crawl4AI MCP Server

A Model Context Protocol (MCP) server implementation that integrates Crawl4AI with Cursor AI, providing web scraping and crawling capabilities as tools for LLMs in Cursor Composer's agent mode.

## System Requirements

Python 3.10 or higher installed.

## Current Features

- Single page scraping

## Installation

Basic setup instructions also available in the [Official Docs for MCP Server QuickStart](https://modelcontextprotocol.io/quickstart/server#why-claude-for-desktop-and-not-claude-ai).

### Set up your environment

First, let's install `uv` and set up our Python project and environment:

MacOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Make sure to restart your terminal afterwards to ensure that the uv command gets picked up.

After that:

1. Clone the repository

2. Install dependencies using UV:

```bash
# Navigate to the crawl4ai-mcp directory
cd crawl4ai-mcp

# Install dependencies (Only first time)
uv venv
uv sync

# Activate the venv
source .venv/bin/activate

# Run the server
python main.py
```

3. Add to Cursor's MCP Servers or Claude's MCP Servers

You may need to put the full path to the uv executable in the command field. You can get this by running `which uv` on MacOS/Linux or `where uv` on Windows.

```json
{
  "mcpServers": {
    "Crawl4AI": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/PARENT/FOLDER/crawl4ai-mcp",
        "run",
        "main.py"
      ]
    }
  }
}
```
