import asyncio
import mcp.types as types
from typing import Any, List
import json
import os
import tempfile
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlResult
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from .utils import validate_and_normalize_url
from .session_manager import session_manager
import logging

CRAWL_TIMEOUT_SECONDS = 300  # Overall timeout for the crawl operation

logger = logging.getLogger(__name__)


async def crawl_website_async(
    url: str, 
    crawl_depth: int = 1, 
    max_pages: int = 5
) -> List[types.TextContent]:
    """
    Crawl a website with session management for stability.
    
    Args:
        url: Starting URL to crawl
        crawl_depth: Maximum depth to crawl (default: 1)
        max_pages: Maximum number of pages to crawl (default: 5)
        
    Returns:
        List containing TextContent with JSON array of crawled pages
    """
    session_id = None
    try:
        # Get crawler and unique session config
        crawler = await session_manager.get_crawler()
        config = await session_manager.get_session_config("crawl")
        session_id = config.session_id
        
        # Update config for crawling
        config.word_threshold = 10
        config.only_text = True
        config.page_timeout = 30000
        
        crawled_pages = []
        urls_to_crawl = [url]
        crawled_urls = set()
        current_depth = 0
        
        logger.info(f"Starting crawl of {url} with depth {crawl_depth} and max pages {max_pages} using session {session_id}")
        
        while urls_to_crawl and len(crawled_pages) < max_pages and current_depth <= crawl_depth:
            current_level_urls = urls_to_crawl.copy()
            urls_to_crawl = []
            
            for current_url in current_level_urls:
                if len(crawled_pages) >= max_pages:
                    break
                    
                if current_url in crawled_urls:
                    continue
                    
                try:
                    # For each URL, create a fresh session config to avoid caching
                    fresh_config = await session_manager.get_session_config("crawl_page")
                    fresh_config.word_threshold = 10
                    fresh_config.only_text = True
                    fresh_config.page_timeout = 30000
                    
                    # Perform the crawl for this URL
                    result = await crawler.arun(url=current_url, config=fresh_config)
                    crawled_urls.add(current_url)
                    
                    if result.success:
                        page_data = {
                            "success": True,
                            "url": result.url,
                            "title": result.metadata.get("title", "") if result.metadata else "",
                            "status_code": result.status_code,
                            "markdown": result.markdown.raw_markdown if result.markdown else "",
                            "cleaned_html": result.cleaned_html[:3000] if result.cleaned_html else "",  # Smaller limit for crawling
                            "depth": current_depth,
                            "links_found": len(result.links.get("internal", [])) if result.links else 0,
                            "metadata": result.metadata if result.metadata else {},
                            "session_id": fresh_config.session_id
                        }
                        
                        # Add internal links for next depth level if we haven't reached max depth
                        if current_depth < crawl_depth and result.links and "internal" in result.links:
                            base_domain = current_url.split('/')[2] if '//' in current_url else current_url.split('/')[0]
                            for link in result.links["internal"][:10]:  # Limit to 10 links per page
                                if base_domain in link and link not in crawled_urls:
                                    urls_to_crawl.append(link)
                        
                        logger.info(f"Successfully crawled {current_url} (depth {current_depth}) with session {fresh_config.session_id}")
                        
                    else:
                        page_data = {
                            "success": False,
                            "url": current_url,
                            "error": result.error_message or "Unknown error occurred",
                            "status_code": result.status_code if hasattr(result, 'status_code') else None,
                            "depth": current_depth,
                            "session_id": fresh_config.session_id
                        }
                        
                        logger.warning(f"Failed to crawl {current_url}: {page_data['error']}")
                    
                    # Mark page session as inactive (don't immediately clean up)
                    await session_manager.cleanup_session(fresh_config.session_id)
                    
                    crawled_pages.append(page_data)
                    
                except Exception as e:
                    logger.error(f"Error crawling {current_url}: {str(e)}")
                    crawled_pages.append({
                        "success": False,
                        "url": current_url,
                        "error": f"Crawling failed: {str(e)}",
                        "error_type": type(e).__name__,
                        "depth": current_depth,
                        "session_id": session_id
                    })
            
            current_depth += 1
        
        # Prepare final response
        response_data = {
            "crawl_summary": {
                "total_pages_crawled": len(crawled_pages),
                "successful_pages": len([p for p in crawled_pages if p.get("success", False)]),
                "failed_pages": len([p for p in crawled_pages if not p.get("success", False)]),
                "max_depth_reached": current_depth - 1,
                "starting_url": url,
                "main_session_id": session_id
            },
            "pages": crawled_pages
        }
        
        logger.info(f"Completed crawl: {response_data['crawl_summary']}")
        
        # Mark main session as inactive (don't immediately clean up)
        if session_id:
            await session_manager.cleanup_session(session_id)
            
        # Periodically clean up old sessions (only when threshold is reached)
        await session_manager.cleanup_old_sessions()
        
        return [types.TextContent(
            type="text",
            text=json.dumps(response_data, indent=2, ensure_ascii=False)
        )]
        
    except Exception as e:
        # Handle session errors
        logger.error(f"Error during crawl of {url}: {str(e)}")
        if session_id:
            await session_manager.handle_session_error(session_id)
        
        error_response = {
            "success": False,
            "starting_url": url,
            "error": f"Website crawling failed: {str(e)}",
            "error_type": type(e).__name__,
            "crawl_summary": {
                "total_pages_crawled": 0,
                "successful_pages": 0,
                "failed_pages": 0,
                "max_depth_reached": 0,
                "starting_url": url,
                "main_session_id": session_id
            },
            "pages": []
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(error_response, indent=2, ensure_ascii=False)
        )]
