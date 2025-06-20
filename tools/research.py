"""
Research Module for crawl4ai-mcp.

Provides comprehensive research capabilities using intelligent crawling,
relevance analysis, and content insights extraction.
"""

import json
import logging
import time
from typing import Dict, List, Any
import mcp.types as types
from .crawl import adaptive_crawl_website_async, crawl_website_async
from .summarise import extract_key_insights

logger = logging.getLogger(__name__)


async def research_topic_async(
    starting_url: str,
    research_query: str,
    max_pages: int = 25,
    depth_strategy: str = "adaptive",
) -> List[types.TextContent]:
    """
    Conduct comprehensive research on a topic using intelligent crawling and analysis.
    
    This function combines adaptive crawling, relevance analysis, and intelligent summarisation
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
    try:
        # Determine which crawling approach to use
        if depth_strategy.lower() == "adaptive":
            # Use advanced adaptive crawling
            crawl_results = await adaptive_crawl_website_async(
                url=starting_url,
                query=research_query,
                max_budget=max_pages
            )
        else:
            # Use standard crawling with query-based relevance
            crawl_results = await crawl_website_async(
                url=starting_url,
                crawl_depth=3,
                max_pages=max_pages,
                query=research_query
            )
        
        # Parse crawl results
        crawl_data = json.loads(crawl_results[0].text)
        
        if not crawl_data.get("pages"):
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "research_query": research_query,
                    "starting_url": starting_url,
                    "error": "No pages were successfully crawled for research"
                }, indent=2)
            )]
        
        # Extract and analyse research content
        research_content = []
        high_relevance_pages = []
        
        for page in crawl_data["pages"]:
            if page.get("success", False):
                page_content = {
                    "url": page["url"],
                    "title": page.get("title", ""),
                    "relevance_score": page.get("relevance_score", 0.0),
                    "key_topics": page.get("key_topics", []),
                    "content": page.get("markdown", "") or page.get("cleaned_html", "")
                }
                
                research_content.append(page_content)
                
                # Identify high-relevance pages for detailed analysis
                if page.get("relevance_score", 0.0) > 0.6:
                    high_relevance_pages.append(page_content)
        
        # Generate comprehensive research insights
        research_focus_areas = [
            "key findings and conclusions",
            "important statistics and data",
            "expert opinions and perspectives",
            "current trends and developments",
            "practical applications and implications"
        ]
        
        # Combine content from high-relevance pages for analysis
        combined_content = "\n\n".join([
            f"## {page['title']} ({page['url']})\nRelevance: {page['relevance_score']:.2f}\n{page['content'][:2000]}"
            for page in high_relevance_pages[:10]  # Top 10 most relevant pages
        ])
        
        if not combined_content.strip():
            combined_content = "\n\n".join([
                f"## {page['title']} ({page['url']})\n{page['content'][:1500]}"
                for page in research_content[:15]  # Fallback to any successful pages
            ])
        
        # Extract key insights
        research_insights = await extract_key_insights(
            content=combined_content,
            focus_areas=research_focus_areas,
            max_tokens=800
        )
        
        # Calculate research statistics
        total_pages = len(crawl_data["pages"])
        successful_pages = len(research_content)
        avg_relevance = sum(p["relevance_score"] for p in research_content) / len(research_content) if research_content else 0.0
        
        # Prepare comprehensive research response
        research_results = {
            "success": True,
            "research_query": research_query,
            "starting_url": starting_url,
            "crawl_strategy": depth_strategy,
            "research_summary": {
                "total_pages_analysed": total_pages,
                "successful_pages": successful_pages,
                "high_relevance_pages": len(high_relevance_pages),
                "average_relevance_score": avg_relevance,
                "research_efficiency": len(high_relevance_pages) / max(1, successful_pages),
                "topics_discovered": len(set(topic for page in research_content for topic in page["key_topics"])),
                "depth_strategy": depth_strategy
            },
            "key_insights": research_insights,
            "high_value_sources": [
                {
                    "url": page["url"],
                    "title": page["title"],
                    "relevance_score": page["relevance_score"],
                    "key_topics": page["key_topics"][:5],  # Top 5 topics
                    "content_preview": page["content"][:300] + "..." if len(page["content"]) > 300 else page["content"]
                }
                for page in sorted(high_relevance_pages, key=lambda x: x["relevance_score"], reverse=True)[:8]
            ],
            "topic_coverage": list(set(topic for page in research_content for topic in page["key_topics"]))[:20],
            "crawl_details": crawl_data.get("crawl_summary", {}),
            "llm_evaluation": crawl_data.get("llm_evaluation", {}) if "llm_evaluation" in crawl_data else None
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(research_results, indent=2, ensure_ascii=False)
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "research_query": research_query,
                "starting_url": starting_url,
                "error": f"Research failed: {str(e)}",
                "error_type": type(e).__name__
            }, indent=2)
        )] 