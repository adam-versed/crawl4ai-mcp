import mcp.types as types
from mcp.server.fastmcp import FastMCP
from tools.scrape import scrape_url

# Initialize FastMCP server
mcp = FastMCP("crawl4ai")


@mcp.tool()
async def scrape_webpage(
    url: str,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Scrape content and metadata from a webpage using Crawl4AI.

    Args:
        url: The URL of the webpage to scrape

    Returns:
        Dictionary with scraped content and metadata
    """
    return await scrape_url(url)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="sse")
    # mcp.run(transport="stdio")  # for testing
