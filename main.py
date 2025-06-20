import mcp.types as types
from mcp.server.fastmcp import FastMCP
from tools.scrape import scrape_url
from tools.crawl import crawl_website_async, adaptive_crawl_website_async
from tools.session_manager import cleanup_on_exit
from tools.mcp_sampler import initialise_sampler
import json
import logging
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize FastMCP server
mcp = FastMCP("crawl4ai")

# Initialize MCP sampler for LLM capabilities
initialise_sampler(mcp)

# Register cleanup function
atexit.register(lambda: cleanup_on_exit())


@mcp.tool()
async def scrape_webpage(
    url: str,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Scrape content and metadata from a single webpage using Crawl4AI.

    Args:
        url: The URL of the webpage to scrape

    Returns:
        List containing TextContent with the result as JSON.
    """
    return await scrape_url(url)


@mcp.tool()
async def crawl_website(
    url: str,
    crawl_depth: int = 1,
    max_pages: int = 5,
    query: str = "",
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Crawl a website starting from the given URL up to a specified depth and page limit.

    Args:
        url: The starting URL to crawl.
        crawl_depth: The maximum depth to crawl relative to the starting URL (default: 1).
        max_pages: The maximum number of pages to scrape during the crawl (default: 5).
        query: Optional search query for relevance-based link selection (default: "").

    Returns:
        List containing TextContent with a JSON array of results for crawled pages.
    """
    return await crawl_website_async(url, crawl_depth, max_pages, query)


@mcp.tool()
async def adaptive_crawl_website(
    url: str,
    query: str,
    max_budget: int = 20,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Perform adaptive crawling with LLM-driven decisions and intelligent link selection.
    
    This advanced crawling mode uses AI to make intelligent decisions about which 
    pages to crawl based on relevance to your query. It prioritises high-value 
    content and stops when diminishing returns are detected.

    Args:
        url: The starting URL to crawl.
        query: Search query that drives the crawling decisions and relevance analysis.
        max_budget: Maximum number of pages to crawl (budget limit, default: 20).

    Returns:
        List containing TextContent with a JSON array of intelligently selected crawled pages.
    """
    return await adaptive_crawl_website_async(url, query, max_budget)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
