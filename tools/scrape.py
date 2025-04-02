import asyncio
import mcp.types as types
from typing import Any, List
import json
import re
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig


async def scrape_url(url: str) -> List[Any]:
    """Scrape a webpage using crawl4ai with simple implementation.

    Args:
        url: The URL to scrape

    Returns:
        A list containing TextContent object with the result as JSON
    """

    try:
        # Simple validation for domains/subdomains with http(s)
        url_pattern = re.compile(
            r"^(?:https?://)?(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}(?:/[^/\s]*)*$"
        )

        if not url_pattern.match(url):
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

        # Add https:// if missing
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"

        # Use default configurations with minimal customization
        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            ignore_https_errors=True,
            extra_args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            verbose=True,
            page_timeout=30 * 1000,  # Convert to milliseconds
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await asyncio.wait_for(
                crawler.arun(
                    url=url,
                    config=run_config,
                ),
                timeout=30,
            )

            # Create response in the format requested
            return [
                types.TextContent(
                    type="text", text=json.dumps({"markdown": result.markdown})
                )
            ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"success": False, "url": url, "error": str(e)}),
            )
        ]
