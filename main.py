import mcp.types as types
from mcp.server.fastmcp import FastMCP
from tools.scrape import scrape_url
from tools.crawl import crawl_website_async, adaptive_crawl_website_async
from tools.session_manager import cleanup_on_exit
from tools.mcp_sampler import initialise_sampler, analyse_content_relevance_async
from tools.summarise import summarise_webpage_async, summarise_crawl_results_async, extract_key_insights_async
from tools.research import research_topic_async
from tools.diagnostics import system_diagnostics_async
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


@mcp.tool()
async def summarise_webpage(
    url: str,
    max_tokens: int = 500,
    focus: str = "",
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Scrape and summarise a webpage with intelligent token management.

    Args:
        url: The URL of the webpage to scrape and summarise
        max_tokens: Maximum tokens in the summary (default: 500)
        focus: Optional focus area for summarisation

    Returns:
        List containing TextContent with the summarised result as JSON.
    """
    return await summarise_webpage_async(url, max_tokens, focus)


@mcp.tool()
async def summarise_crawl_results_tool(
    crawl_results: str,
    query: str = "",
    include_page_summaries: bool = True,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Summarise crawl results with optional query focus and individual page summaries.

    Args:
        crawl_results: JSON string of crawl results from crawl_website
        query: Optional query for focused summarisation
        include_page_summaries: Whether to include individual page summaries (default: True)

    Returns:
        List containing TextContent with enhanced crawl results including summaries.
    """
    return await summarise_crawl_results_async(crawl_results, query, include_page_summaries)


@mcp.tool()
async def analyse_content_relevance(
    content: str,
    query: str,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Analyse how relevant content is to a specific query using LLM analysis.

    Args:
        content: The content to analyse for relevance
        query: The query to check relevance against

    Returns:
        List containing TextContent with relevance analysis as JSON.
    """
    return await analyse_content_relevance_async(content, query)


@mcp.tool()
async def extract_key_insights_tool(
    content: str,
    focus_areas: list[str] = [],
    max_tokens: int = 600,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Extract key insights from content with optional focus areas.

    Args:
        content: The content to analyse for insights
        focus_areas: Optional list of specific areas to focus analysis on
        max_tokens: Maximum tokens in the analysis response (default: 600)

    Returns:
        List containing TextContent with key insights analysis as JSON.
    """
    return await extract_key_insights_async(content, focus_areas, max_tokens)


@mcp.tool()
async def research_topic(
    starting_url: str,
    research_query: str,
    max_pages: int = 25,
    depth_strategy: str = "adaptive",
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Conduct comprehensive research on a topic using intelligent crawling and analysis.
    
    This tool combines adaptive crawling, relevance analysis, and intelligent summarisation
    to provide comprehensive research results on any topic. It uses LLM-driven decisions
    to find the most relevant and valuable content.

    Args:
        starting_url: The starting URL for research (should be related to your topic).
        research_query: Detailed research question or topic description.
        max_pages: Maximum number of pages to analyse (default: 25).
        depth_strategy: Crawling strategy - "adaptive" (recommended) or "fixed" (default: "adaptive").

    Returns:
        List containing TextContent with comprehensive research results including analysis and insights.
    """
    return await research_topic_async(starting_url, research_query, max_pages, depth_strategy)


@mcp.tool()
async def system_diagnostics(
    include_cache_stats: bool = True,
    include_error_stats: bool = True,
    include_config_info: bool = False,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Get comprehensive system diagnostics including cache performance, error statistics, and configuration status.
    
    This tool provides detailed information about the crawl4ai-mcp server's operational status,
    performance metrics, and system health. Useful for monitoring and troubleshooting.

    Args:
        include_cache_stats: Include cache performance statistics (default: True).
        include_error_stats: Include error handling statistics (default: True).
        include_config_info: Include configuration information (default: False).

    Returns:
        List containing TextContent with comprehensive system diagnostics as JSON.
    """
    return await system_diagnostics_async(include_cache_stats, include_error_stats, include_config_info)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
