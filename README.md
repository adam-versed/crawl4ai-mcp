# Crawl4AI MCP Server

A Model Context Protocol (MCP) server implementation that integrates Crawl4AI with Cursor AI, providing web scraping and crawling capabilities as tools for LLMs in Cursor Composer's agent mode.

## System Requirements

Python 3.10 or higher installed.

## Current Features

- Single page scraping
- Website crawling

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

## Tools Provided

This MCP server exposes the following tools to the LLM:

1.  **`scrape_webpage(url: str)`**

    - **Description:** Scrapes the content and metadata from a single webpage using Crawl4AI.
    - **Parameters:**
      - `url` (string, required): The URL of the webpage to scrape.
    - **Returns:** A list containing a `TextContent` object with the scraped content (primarily markdown) as JSON.

2.  **`crawl_website(url: str, crawl_depth: int = 1, max_pages: int = 5)`**
    - **Description:** Crawls a website starting from the given URL up to a specified depth and page limit using Crawl4AI.
    - **Parameters:**
      - `url` (string, required): The starting URL to crawl.
      - `crawl_depth` (integer, optional, default: 1): The maximum depth to crawl relative to the starting URL.
      - `max_pages` (integer, optional, default: 5): The maximum number of pages to scrape during the crawl.
    - **Returns:** A list containing a `TextContent` object with a JSON array of results for the crawled pages (including URL, success status, markdown content, or error).
