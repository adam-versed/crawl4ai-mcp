"""
Smart Summarisation Module for crawl4ai-mcp.

Provides intelligent summarisation capabilities with token management,
using MCP sampling for LLM-powered summarisation.
"""

import json
import logging
from typing import Dict, List, Optional, Any
import mcp.types as types
from .utils import count_tokens, truncate_to_token_limit, chunk_text_by_tokens
from .mcp_sampler import get_sampler
from .scrape import scrape_url

logger = logging.getLogger(__name__)


async def summarise_content(
    content: str, 
    max_input_tokens: int = 8000, 
    max_output_tokens: int = 500,
    focus: Optional[str] = None
) -> str:
    """Summarise content with token management.
    
    Args:
        content: Content to summarise
        max_input_tokens: Maximum tokens to send to LLM
        max_output_tokens: Maximum tokens in summary
        focus: Optional focus area for summarisation
        
    Returns:
        Summary text
    """
    try:
        # Check if content needs truncation
        content_tokens = count_tokens(content)
        logger.info(f"Content has {content_tokens} tokens, limit is {max_input_tokens}")
        
        if content_tokens > max_input_tokens:
            logger.info(f"Truncating content from {content_tokens} to {max_input_tokens} tokens")
            content = truncate_to_token_limit(content, max_input_tokens)
        
        # Get sampler and request summarisation
        sampler = get_sampler()
        summary = await sampler.summarise_text(
            text=content,
            max_output_tokens=max_output_tokens,
            focus=focus
        )
        
        logger.info(f"Generated summary with {count_tokens(summary)} tokens")
        return summary
        
    except Exception as e:
        logger.error(f"Error summarising content: {str(e)}")
        # Fallback to simple truncation if LLM fails
        return truncate_to_token_limit(content, max_output_tokens)


async def summarise_page_for_relevance(
    page_data: Dict[str, Any], 
    query: str, 
    max_tokens: int = 300
) -> Dict[str, Any]:
    """Summarise a page with relevance analysis for a specific query.
    
    Args:
        page_data: Page data dictionary from crawling
        query: Query to analyse relevance against
        max_tokens: Maximum tokens in summary
        
    Returns:
        Enhanced page data with summary and relevance info
    """
    try:
        # Extract content for summarisation
        content_parts = []
        
        if page_data.get("title"):
            content_parts.append(f"Title: {page_data['title']}")
        
        if page_data.get("markdown"):
            content_parts.append(f"Content: {page_data['markdown']}")
        elif page_data.get("cleaned_html"):
            content_parts.append(f"Content: {page_data['cleaned_html']}")
        
        full_content = "\n\n".join(content_parts)
        
        if not full_content.strip():
            logger.warning("No content found for summarisation")
            return {
                **page_data,
                "summary": "No content available for summarisation",
                "relevance_analysis": {
                    "relevance_score": 0.0,
                    "reasoning": "No content to analyse",
                    "key_topics": [],
                    "matches_query": False
                }
            }
        
        # Get sampler for analysis
        sampler = get_sampler()
        
        # Analyse relevance first
        relevance_analysis = await sampler.analyse_relevance(
            content=full_content,
            query=query,
            max_tokens=200
        )
        
        # Generate focused summary
        summary = await summarise_content(
            content=full_content,
            max_input_tokens=6000,  # Leave room for query context
            max_output_tokens=max_tokens,
            focus=f"information relevant to: {query}"
        )
        
        # Enhance page data
        enhanced_page = {
            **page_data,
            "summary": summary,
            "relevance_analysis": relevance_analysis,
            "query_context": query
        }
        
        logger.info(f"Summarised page {page_data.get('url', 'unknown')} with relevance score {relevance_analysis.get('relevance_score', 0.0)}")
        return enhanced_page
        
    except Exception as e:
        logger.error(f"Error summarising page for relevance: {str(e)}")
        return {
            **page_data,
            "summary": "Error generating summary",
            "relevance_analysis": {
                "relevance_score": 0.0,
                "reasoning": f"Error: {str(e)}",
                "key_topics": [],
                "matches_query": False
            }
        }


async def batch_summarise_pages(
    pages: List[Dict[str, Any]], 
    query: str = "",
    max_summary_tokens: int = 200
) -> List[Dict[str, Any]]:
    """Batch summarise multiple pages with optional query focus.
    
    Args:
        pages: List of page data dictionaries
        query: Optional query for relevance-focused summarisation
        max_summary_tokens: Maximum tokens per summary
        
    Returns:
        List of enhanced page data with summaries
    """
    summarised_pages = []
    
    for i, page in enumerate(pages):
        logger.info(f"Summarising page {i+1}/{len(pages)}: {page.get('url', 'unknown')}")
        
        try:
            if query:
                enhanced_page = await summarise_page_for_relevance(
                    page_data=page,
                    query=query,
                    max_tokens=max_summary_tokens
                )
            else:
                # Simple summarisation without query focus
                content_parts = []
                if page.get("title"):
                    content_parts.append(f"Title: {page['title']}")
                if page.get("markdown"):
                    content_parts.append(f"Content: {page['markdown']}")
                elif page.get("cleaned_html"):
                    content_parts.append(f"Content: {page['cleaned_html']}")
                
                full_content = "\n\n".join(content_parts)
                summary = await summarise_content(
                    content=full_content,
                    max_output_tokens=max_summary_tokens
                )
                
                enhanced_page = {
                    **page,
                    "summary": summary
                }
            
            summarised_pages.append(enhanced_page)
            
        except Exception as e:
            logger.error(f"Error summarising page {page.get('url', 'unknown')}: {str(e)}")
            summarised_pages.append({
                **page,
                "summary": "Error generating summary",
                "error": str(e)
            })
    
    return summarised_pages


async def summarise_crawl_results(
    crawl_data: Dict[str, Any], 
    query: str = "",
    include_page_summaries: bool = True
) -> Dict[str, Any]:
    """Summarise entire crawl results with optional query focus.
    
    Args:
        crawl_data: Complete crawl results dictionary
        query: Optional query for focused summarisation
        include_page_summaries: Whether to include individual page summaries
        
    Returns:
        Enhanced crawl data with overall summary
    """
    try:
        pages = crawl_data.get("pages", [])
        crawl_summary = crawl_data.get("crawl_summary", {})
        
        # Summarise individual pages if requested
        if include_page_summaries and pages:
            summarised_pages = await batch_summarise_pages(pages, query)
        else:
            summarised_pages = pages
        
        # Generate overall summary
        successful_pages = [p for p in summarised_pages if p.get("success", False)]
        
        if not successful_pages:
            overall_summary = "No successful pages to summarise."
        else:
            # Combine key information from all pages
            combined_content = []
            
            # Add crawl context
            starting_url = crawl_summary.get("starting_url", "unknown")
            total_pages = crawl_summary.get("total_pages_crawled", 0)
            successful_count = crawl_summary.get("successful_pages", 0)
            
            combined_content.append(f"Crawl Results Summary:")
            combined_content.append(f"Starting URL: {starting_url}")
            combined_content.append(f"Total pages crawled: {total_pages}")
            combined_content.append(f"Successful pages: {successful_count}")
            
            if query:
                combined_content.append(f"Query focus: {query}")
            
            # Add page summaries or content
            for i, page in enumerate(successful_pages[:10]):  # Limit to first 10 pages
                page_info = []
                page_info.append(f"Page {i+1}: {page.get('url', 'unknown')}")
                
                if page.get("title"):
                    page_info.append(f"Title: {page['title']}")
                
                if page.get("summary"):
                    page_info.append(f"Summary: {page['summary']}")
                elif page.get("markdown"):
                    # Use first part of markdown if no summary
                    content = page["markdown"][:500] + "..." if len(page["markdown"]) > 500 else page["markdown"]
                    page_info.append(f"Content: {content}")
                
                if page.get("relevance_analysis", {}).get("relevance_score"):
                    score = page["relevance_analysis"]["relevance_score"]
                    page_info.append(f"Relevance Score: {score:.2f}")
                
                combined_content.append("\n".join(page_info))
            
            full_summary_content = "\n\n".join(combined_content)
            
            # Generate overall summary
            focus_text = f"research on {query}" if query else "general web crawling results"
            overall_summary = await summarise_content(
                content=full_summary_content,
                max_input_tokens=10000,
                max_output_tokens=800,
                focus=f"key findings and insights from {focus_text}"
            )
        
        # Return enhanced crawl data
        return {
            **crawl_data,
            "pages": summarised_pages,
            "overall_summary": overall_summary,
            "query_context": query,
            "summarisation_metadata": {
                "pages_summarised": len(summarised_pages),
                "successful_pages": len(successful_pages),
                "query_focused": bool(query),
                "include_page_summaries": include_page_summaries
            }
        }
        
    except Exception as e:
        logger.error(f"Error summarising crawl results: {str(e)}")
        return {
            **crawl_data,
            "overall_summary": f"Error generating summary: {str(e)}",
            "error": str(e)
        }


async def extract_key_insights(
    content: str, 
    focus_areas: List[str],
    max_tokens: int = 600
) -> Dict[str, Any]:
    """Extract key insights from content focusing on specific areas.
    
    Args:
        content: Content to analyse
        focus_areas: List of areas to focus analysis on
        max_tokens: Maximum tokens in response
        
    Returns:
        Dictionary with key insights
    """
    try:
        sampler = get_sampler()
        
        focus_text = ", ".join(focus_areas) if focus_areas else "general insights"
        
        system_prompt = """You are an expert analyst. Extract key insights and provide a structured analysis.
        Respond with a JSON object containing:
        - key_insights: list of main insights found
        - focus_analysis: analysis for each requested focus area
        - main_themes: list of primary themes
        - actionable_items: list of actionable takeaways
        - confidence_level: float between 0.0 and 1.0 indicating confidence in analysis"""
        
        prompt = f"""Analyse the following content and extract key insights, focusing on: {focus_text}

Content to analyse:
{content}

Provide your analysis as a structured JSON object."""
        
        messages = [{"role": "user", "content": prompt}]
        
        response = await sampler.request_llm_completion(
            messages=messages,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            temperature=0.2
        )
        
        # Parse JSON response
        try:
            insights = json.loads(response)
            return {
                "success": True,
                "insights": insights,
                "focus_areas": focus_areas
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse insights response as JSON: {response}")
            return {
                "success": False,
                "error": "Failed to parse LLM response",
                "raw_response": response,
                "focus_areas": focus_areas
            }
            
    except Exception as e:
        logger.error(f"Error extracting key insights: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "focus_areas": focus_areas
        }


async def summarise_webpage_async(
    url: str,
    max_tokens: int = 500,
    focus: str = "",
) -> List[types.TextContent]:
    """
    Scrape and summarise a webpage with intelligent token management.

    Args:
        url: The URL of the webpage to scrape and summarise
        max_tokens: Maximum tokens in the summary (default: 500)
        focus: Optional focus area for summarisation

    Returns:
        List containing TextContent with the summarised result as JSON.
    """
    try:
        # First scrape the webpage
        scrape_results = await scrape_url(url)
        scrape_data = json.loads(scrape_results[0].text)
        
        if not scrape_data.get("success", False):
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "url": url,
                    "error": "Failed to scrape webpage for summarisation",
                    "scrape_error": scrape_data.get("error", "Unknown error")
                }, indent=2)
            )]
        
        # Extract content for summarisation
        content_parts = []
        if scrape_data.get("title"):
            content_parts.append(f"Title: {scrape_data['title']}")
        if scrape_data.get("markdown"):
            content_parts.append(f"Content: {scrape_data['markdown']}")
        elif scrape_data.get("cleaned_html"):
            content_parts.append(f"Content: {scrape_data['cleaned_html']}")
        
        full_content = "\n\n".join(content_parts)
        
        if not full_content.strip():
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No content available for summarisation"
                }, indent=2)
            )]
        
        # Generate summary
        focus_area = focus if focus.strip() else None
        summary = await summarise_content(
            content=full_content,
            max_output_tokens=max_tokens,
            focus=focus_area
        )
        
        # Prepare response
        response_data = {
            "success": True,
            "url": url,
            "title": scrape_data.get("title", ""),
            "summary": summary,
            "focus": focus_area,
            "original_content_length": len(full_content),
            "summary_length": len(summary),
            "metadata": scrape_data.get("metadata", {})
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(response_data, indent=2, ensure_ascii=False)
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "url": url,
                "error": f"Summarisation failed: {str(e)}",
                "error_type": type(e).__name__
            }, indent=2)
        )]


async def summarise_crawl_results_async(
    crawl_results: str,
    query: str = "",
    include_page_summaries: bool = True,
) -> List[types.TextContent]:
    """
    Summarise crawl results with optional query focus and individual page summaries.

    Args:
        crawl_results: JSON string of crawl results from crawl_website
        query: Optional query for focused summarisation
        include_page_summaries: Whether to include individual page summaries (default: True)

    Returns:
        List containing TextContent with enhanced crawl results including summaries.
    """
    try:
        # Parse crawl results
        crawl_data = json.loads(crawl_results)
        
        # Summarise the crawl results
        enhanced_results = await summarise_crawl_results(
            crawl_data=crawl_data,
            query=query,
            include_page_summaries=include_page_summaries
        )
        
        return [types.TextContent(
            type="text",
            text=json.dumps(enhanced_results, indent=2, ensure_ascii=False)
        )]
        
    except json.JSONDecodeError as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": f"Invalid JSON in crawl_results: {str(e)}",
                "error_type": "JSONDecodeError"
            }, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": f"Summarisation failed: {str(e)}",
                "error_type": type(e).__name__
            }, indent=2)
        )]


async def extract_key_insights_async(
    content: str,
    focus_areas: list[str] = [],
    max_tokens: int = 600,
) -> List[types.TextContent]:
    """
    Extract key insights from content with optional focus areas.

    Args:
        content: The content to analyse for insights
        focus_areas: Optional list of specific areas to focus analysis on
        max_tokens: Maximum tokens in the analysis response (default: 600)

    Returns:
        List containing TextContent with key insights analysis as JSON.
    """
    try:
        insights = await extract_key_insights(
            content=content,
            focus_areas=focus_areas,
            max_tokens=max_tokens
        )
        
        response_data = {
            "success": True,
            "content_length": len(content),
            "focus_areas": focus_areas,
            "insights": insights
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(response_data, indent=2, ensure_ascii=False)
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "content_length": len(content),
                "focus_areas": focus_areas,
                "error": f"Insight extraction failed: {str(e)}",
                "error_type": type(e).__name__
            }, indent=2)
        )] 