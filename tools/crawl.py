import asyncio
import mcp.types as types
from typing import Any, List, Dict, Tuple
import json
import os
import tempfile
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlResult
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from .utils import validate_and_normalize_url
from .session_manager import session_manager
from .link_selector import get_link_selector
from .relevance_scorer import get_relevance_scorer
from .crawl_context import create_crawl_context, CrawlContext, CrawlStatus, PageCrawlResult
from .llm_decisions import get_llm_decision_maker
import logging

CRAWL_TIMEOUT_SECONDS = 300  # Overall timeout for the crawl operation

logger = logging.getLogger(__name__)


def get_session_manager():
    """Get the session manager instance for testing."""
    return session_manager


async def crawl_website_async(
    url: str, 
    crawl_depth: int = 1, 
    max_pages: int = 5,
    query: str = ""
) -> List[types.TextContent]:
    """
    Crawl a website with session management for stability.
    
    Args:
        url: Starting URL to crawl
        crawl_depth: Maximum depth to crawl (default: 1)
        max_pages: Maximum number of pages to crawl (default: 5)
        query: Optional search query for relevance-based crawling
        
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
        
        # Initialize relevance scoring if query provided
        link_selector = get_link_selector() if query else None
        relevance_scorer = get_relevance_scorer() if query else None
        
        logger.info(f"Starting crawl of {url} with depth {crawl_depth} and max pages {max_pages} using session {session_id}")
        if query:
            logger.info(f"Using query-driven crawling with query: '{query}'")
        
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
                        # Calculate relevance score if query provided
                        relevance_data = {}
                        if query and relevance_scorer:
                            try:
                                relevance_analysis = await relevance_scorer.score_page_relevance(
                                    page_content=result.markdown.raw_markdown if result.markdown else "",
                                    query=query,
                                    page_title=result.metadata.get("title", "") if result.metadata else "",
                                    page_url=current_url
                                )
                                relevance_data = {
                                    "relevance_score": relevance_analysis["relevance_score"],
                                    "matches_query": relevance_analysis["matches_query"],
                                    "key_topics": relevance_analysis.get("key_topics", [])
                                }
                            except Exception as e:
                                logger.warning(f"Error calculating relevance for {current_url}: {str(e)}")
                                relevance_data = {"relevance_score": 0.0, "matches_query": False}
                        
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
                            "session_id": fresh_config.session_id,
                            **relevance_data  # Add relevance data if available
                        }
                        
                        # Add internal links for next depth level if we haven't reached max depth
                        if current_depth < crawl_depth and result.links and "internal" in result.links:
                            available_links = result.links["internal"][:20]  # Get more links for analysis
                            
                            if query and link_selector:
                                # Use intelligent link selection
                                try:
                                    selected_links = await link_selector.select_best_links(
                                        links=available_links,
                                        page_content=result.markdown.raw_markdown if result.markdown else "",
                                        query=query,
                                        crawl_context={
                                            "current_depth": current_depth,
                                            "pages_crawled": len(crawled_pages),
                                            "max_pages": max_pages,
                                            "crawled_urls": crawled_urls
                                        },
                                        current_url=current_url
                                    )
                                    
                                    for link in selected_links:
                                        if link not in crawled_urls:
                                            urls_to_crawl.append(link)
                                            
                                    logger.info(f"Selected {len(selected_links)} relevant links from {len(available_links)} available")
                                    
                                except Exception as e:
                                    logger.warning(f"Error in intelligent link selection, falling back to simple method: {str(e)}")
                                    # Fallback to simple selection
                                    base_domain = current_url.split('/')[2] if '//' in current_url else current_url.split('/')[0]
                                    for link in available_links[:10]:
                                        if base_domain in link and link not in crawled_urls:
                                            urls_to_crawl.append(link)
                            else:
                                # Use simple link selection (original logic)
                                base_domain = current_url.split('/')[2] if '//' in current_url else current_url.split('/')[0]
                                for link in available_links[:10]:
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
        
        # Calculate final statistics
        successful_pages = [p for p in crawled_pages if p.get("success", False)]
        avg_relevance = 0.0
        if query and successful_pages:
            relevance_scores = [p.get("relevance_score", 0.0) for p in successful_pages]
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        # Prepare final response
        response_data = {
            "crawl_summary": {
                "total_pages_crawled": len(crawled_pages),
                "successful_pages": len(successful_pages),
                "failed_pages": len([p for p in crawled_pages if not p.get("success", False)]),
                "max_depth_reached": current_depth - 1,
                "starting_url": url,
                "main_session_id": session_id,
                "query": query,
                "average_relevance_score": avg_relevance if query else None,
                "adaptive_mode": bool(query)
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
                "main_session_id": session_id,
                "query": query,
                "adaptive_mode": bool(query)
            },
            "pages": []
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(error_response, indent=2, ensure_ascii=False)
        )]


async def adaptive_crawl_website_async(
    url: str, 
    query: str, 
    max_budget: int = 20
) -> List[types.TextContent]:
    """
    Perform advanced adaptive crawling with comprehensive LLM-driven decisions.
    
    This function uses intelligent context management, LLM decision-making,
    and sophisticated budget tracking to crawl websites more efficiently.
    
    Args:
        url: Starting URL to crawl
        query: Search query that drives the crawling decisions
        max_budget: Maximum number of pages to crawl (budget limit)
        
    Returns:
        List containing TextContent with JSON array of intelligently selected crawled pages
    """
    session_id = None
    crawl_context = None
    
    try:
        # Initialize crawler and session
        crawler = await session_manager.get_crawler()
        config = await session_manager.get_session_config("adaptive_crawl_v2")
        session_id = config.session_id
        
        # Update config for crawling
        config.word_threshold = 10
        config.only_text = True
        config.page_timeout = 30000
        
        # Create crawl context with advanced tracking
        crawl_context = create_crawl_context(
            starting_url=url,
            query=query,
            max_pages=max_budget,
            max_tokens=15000,  # Increased token budget for LLM decisions
            max_time_seconds=600  # 10 minutes max
        )
        
        # Initialize components
        link_selector = get_link_selector()
        relevance_scorer = get_relevance_scorer()
        llm_decision_maker = get_llm_decision_maker()
        
        # Start with the initial URL
        crawl_context.add_pending_urls([(1.0, 0, url)])
        
        logger.info(f"Starting advanced adaptive crawl with context tracking for query: '{query}'")
        
        while True:
            # Use LLM to decide whether to continue crawling
            should_continue, reasoning = await llm_decision_maker.should_continue_crawling(crawl_context)
            
            if not should_continue:
                crawl_context.update_status(CrawlStatus.COMPLETED, reasoning)
                logger.info(f"Stopping crawl: {reasoning}")
                break
            
            # Get next URLs to crawl
            next_urls = crawl_context.prioritise_pending_urls(max_urls=3)
            if not next_urls:
                crawl_context.update_status(CrawlStatus.COMPLETED, "No more URLs to crawl")
                break
            
            # Remove the URLs we're about to crawl from the pending queue
            crawl_context.pending_urls = [
                (priority, depth, url) for priority, depth, url in crawl_context.pending_urls
                if url not in next_urls
            ]
            
            # Crawl the selected URLs
            for current_url in next_urls:
                if current_url in crawl_context.crawled_urls:
                    continue
                
                try:
                    # Create fresh session config
                    fresh_config = await session_manager.get_session_config("adaptive_page_v2")
                    fresh_config.word_threshold = 10
                    fresh_config.only_text = True
                    fresh_config.page_timeout = 30000
                    
                    # Crawl the page
                    result = await crawler.arun(url=current_url, config=fresh_config)
                    
                    # Create page result object
                    if result.success:
                        # Analyse page relevance
                        relevance_analysis = await relevance_scorer.score_page_relevance(
                            page_content=result.markdown.raw_markdown if result.markdown else "",
                            query=query,
                            page_title=result.metadata.get("title", "") if result.metadata else "",
                            page_url=current_url
                        )
                        
                        page_result = PageCrawlResult(
                            url=result.url,
                            success=True,
                            relevance_score=relevance_analysis["relevance_score"],
                            depth=len([u for u in crawl_context.crawled_urls if u.count('/') < current_url.count('/')]),
                            title=result.metadata.get("title", "") if result.metadata else "",
                            key_topics=relevance_analysis.get("key_topics", []),
                            links_found=len(result.links.get("internal", [])) if result.links else 0,
                            content_length=len(result.markdown.raw_markdown) if result.markdown else 0,
                            session_id=fresh_config.session_id
                        )
                        
                        # Process links if the page is relevant
                        if relevance_analysis["matches_query"] and result.links and "internal" in result.links:
                            available_links = result.links["internal"][:30]
                            
                            try:
                                # Analyse links for crawling priority
                                analysed_links = await link_selector.analyse_links_for_crawling(
                                    links=available_links,
                                    page_content=result.markdown.raw_markdown if result.markdown else "",
                                    query=query,
                                    current_url=current_url,
                                    page_title=result.metadata.get("title", "") if result.metadata else ""
                                )
                                
                                # Use LLM to select the best links
                                selected_urls = await llm_decision_maker.select_next_links(
                                    available_links=analysed_links,
                                    context=crawl_context,
                                    max_links=8
                                )
                                
                                # Add selected links to pending queue with calculated priorities
                                new_pending = []
                                for link_url in selected_urls:
                                    # Find the analysis for this URL
                                    link_analysis = next(
                                        (l for l in analysed_links if l["url"] == link_url),
                                        None
                                    )
                                    if link_analysis:
                                        priority = link_analysis.get("relevance_score", 0.5)
                                        depth = page_result.depth + 1
                                        new_pending.append((priority, depth, link_url))
                                
                                crawl_context.add_pending_urls(new_pending)
                                
                                logger.info(f"Added {len(selected_urls)} LLM-selected links from {current_url}")
                                
                            except Exception as e:
                                logger.warning(f"Error in LLM link selection for {current_url}: {str(e)}")
                        
                        logger.info(f"Successfully crawled {current_url} (relevance: {page_result.relevance_score:.2f})")
                        
                    else:
                        page_result = PageCrawlResult(
                            url=current_url,
                            success=False,
                            error=result.error_message or "Unknown error occurred",
                            session_id=fresh_config.session_id
                        )
                        
                        logger.warning(f"Failed to crawl {current_url}: {page_result.error}")
                    
                    # Add page result to context
                    crawl_context.add_crawled_page(page_result)
                    
                    # Clean up session
                    await session_manager.cleanup_session(fresh_config.session_id)
                    
                except Exception as e:
                    logger.error(f"Error in advanced adaptive crawl of {current_url}: {str(e)}")
                    error_page = PageCrawlResult(
                        url=current_url,
                        success=False,
                        error=f"Advanced crawling failed: {str(e)}"
                    )
                    crawl_context.add_crawled_page(error_page)
                
                # Check if we should stop after each page
                if crawl_context.budget.is_any_budget_exceeded():
                    break
        
        # Perform final evaluation using LLM
        final_evaluation = await llm_decision_maker.evaluate_crawl_completion(
            context=crawl_context,
            query=query
        )
        
        # Prepare comprehensive response
        context_summary = crawl_context.get_context_summary()
        successful_pages = [p for p in crawl_context.crawled_pages if p.success]
        
        # Convert PageCrawlResult objects to dictionaries for JSON serialization
        pages_data = []
        for page in crawl_context.crawled_pages:
            page_dict = {
                "success": page.success,
                "url": page.url,
                "relevance_score": page.relevance_score,
                "depth": page.depth,
                "title": page.title,
                "key_topics": page.key_topics,
                "links_found": page.links_found,
                "content_length": page.content_length,
                "session_id": page.session_id,
                "crawl_timestamp": page.crawl_timestamp
            }
            
            if not page.success and page.error:
                page_dict["error"] = page.error
            
            pages_data.append(page_dict)
        
        response_data = {
            "crawl_summary": {
                "starting_url": url,
                "query": query,
                "status": crawl_context.status.value,
                "total_pages_crawled": len(crawl_context.crawled_pages),
                "successful_pages": len(successful_pages),
                "high_relevance_pages": crawl_context.high_relevance_pages,
                "average_relevance_score": crawl_context.get_average_relevance(),
                "crawl_efficiency": crawl_context.get_crawl_efficiency(),
                "topics_discovered": list(crawl_context.topics_discovered),
                "budget_utilisation": context_summary["budget_status"],
                "main_session_id": session_id,
                "adaptive_mode": "advanced_llm_driven"
            },
            "llm_evaluation": final_evaluation,
            "decision_history": crawl_context.decision_history,
            "pages": pages_data
        }
        
        logger.info(f"Completed advanced adaptive crawl with {len(successful_pages)} successful pages")
        
        # Clean up
        if session_id:
            await session_manager.cleanup_session(session_id)
        await session_manager.cleanup_old_sessions()
        
        return [types.TextContent(
            type="text",
            text=json.dumps(response_data, indent=2, ensure_ascii=False)
        )]
        
    except Exception as e:
        logger.error(f"Error during advanced adaptive crawl of {url}: {str(e)}")
        if session_id:
            await session_manager.handle_session_error(session_id)
        
        # Prepare error response with context if available
        error_response = {
            "success": False,
            "starting_url": url,
            "query": query,
            "error": f"Advanced adaptive crawling failed: {str(e)}",
            "error_type": type(e).__name__,
            "crawl_summary": {
                "starting_url": url,
                "query": query,
                "status": "failed",
                "total_pages_crawled": len(crawl_context.crawled_pages) if crawl_context else 0,
                "successful_pages": 0,
                "high_relevance_pages": 0,
                "average_relevance_score": 0.0,
                "crawl_efficiency": 0.0,
                "main_session_id": session_id,
                "adaptive_mode": "advanced_llm_driven"
            },
            "pages": []
        }
        
        # Include partial results if available
        if crawl_context and crawl_context.crawled_pages:
            error_response["partial_results"] = {
                "pages_crawled": len(crawl_context.crawled_pages),
                "context_summary": crawl_context.get_context_summary()
            }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(error_response, indent=2, ensure_ascii=False)
        )]


async def _should_continue_adaptive_crawl(
    relevance_analysis: Dict[str, Any], 
    crawl_context: Dict[str, Any]
) -> bool:
    """
    Determine whether to continue crawling from a given page.
    
    Args:
        relevance_analysis: Analysis of current page relevance
        crawl_context: Current crawling context and statistics
        
    Returns:
        True if crawling should continue from this page
    """
    try:
        # Don't continue if page is not relevant
        if not relevance_analysis.get("matches_query", False):
            return False
        
        # Don't continue if we're running low on budget
        budget_remaining = crawl_context["max_budget"] - crawl_context["pages_crawled"]
        if budget_remaining < 3:
            return False
        
        # Don't continue if we're too deep
        if crawl_context["current_depth"] > 4:
            return False
        
        # Continue if page is highly relevant
        if relevance_analysis.get("relevance_score", 0.0) > 0.7:
            return True
        
        # Continue if page is moderately relevant and we have budget
        if (relevance_analysis.get("relevance_score", 0.0) > 0.4 and 
            budget_remaining > 5):
            return True
        
        # Continue if average relevance is low (we need better content)
        if crawl_context.get("average_relevance", 0.0) < 0.3:
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error in adaptive crawl decision: {str(e)}")
        return False  # Conservative fallback
