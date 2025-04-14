import asyncio
import mcp.types as types
from typing import Any, List
import json
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlResult
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from .utils import validate_and_normalize_url

CRAWL_TIMEOUT_SECONDS = 300  # Overall timeout for the crawl operation


async def crawl_website_async(url: str, crawl_depth: int, max_pages: int) -> List[Any]:
    """Crawl a website using crawl4ai.

    Args:
        url: The starting URL to crawl.
        crawl_depth: The maximum depth to crawl.
        max_pages: The maximum number of pages to crawl.

    Returns:
        A list containing TextContent objects with the results as JSON.
    """

    normalized_url = validate_and_normalize_url(url)
    if not normalized_url:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "success": False,
                        "url": url,
                        "error": "Invalid URL format",
                    }
                ),
            )
        ]

    try:
        # Use default configurations with minimal customization
        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            ignore_https_errors=True,
            verbose=False,
            extra_args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        # 1. Create the deep crawl strategy with depth and page limits
        crawl_strategy = BFSDeepCrawlStrategy(
            max_depth=crawl_depth, max_pages=max_pages
        )

        # 2. Create the run config, passing the strategy
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            verbose=False,
            page_timeout=30 * 1000,  # 30 seconds per page
            deep_crawl_strategy=crawl_strategy,  # Pass the strategy here
        )

        results_list = []
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # 3. Use arun and wrap in asyncio.wait_for for overall timeout
            crawl_results: List[CrawlResult] = await asyncio.wait_for(
                crawler.arun(
                    url=normalized_url,
                    config=run_config,
                ),
                timeout=CRAWL_TIMEOUT_SECONDS,
            )

            # Process results, checking 'success' attribute
            for result in crawl_results:
                if result.success:  # Check .success instead of .status
                    results_list.append(
                        {
                            "url": result.url,
                            "success": True,
                            "markdown": result.markdown,
                        }
                    )
                else:
                    results_list.append(
                        {
                            "url": result.url,
                            "success": False,
                            "error": result.error,  # Assume .error holds the message
                        }
                    )

            # Return a single TextContent with a JSON array of results
            return [
                types.TextContent(
                    type="text", text=json.dumps({"results": results_list})
                )
            ]

    except asyncio.TimeoutError:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "success": False,
                        "url": normalized_url,
                        "error": f"Crawl operation timed out after {CRAWL_TIMEOUT_SECONDS} seconds.",
                    }
                ),
            )
        ]
    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {"success": False, "url": normalized_url, "error": str(e)}
                ),
            )
        ]
