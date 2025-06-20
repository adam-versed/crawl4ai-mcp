"""
LLM-Driven Decision Making for crawl4ai-mcp.

Provides intelligent decision-making capabilities using LLM analysis
for advanced adaptive crawling with contextual reasoning.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from .crawl_context import CrawlContext, CrawlStatus, PageCrawlResult
from .mcp_sampler import get_sampler
from .utils import count_tokens, truncate_to_token_limit

logger = logging.getLogger(__name__)


class LLMCrawlDecisionMaker:
    """Handles LLM-powered decision making for crawling operations."""
    
    def __init__(self):
        self.decision_cache = {}  # Simple cache for repeated decisions
        self.cache_ttl = 300  # 5 minutes cache lifetime
    
    async def should_continue_crawling(
        self, 
        context: CrawlContext
    ) -> Tuple[bool, str]:
        """
        Use LLM to determine whether crawling should continue.
        
        Args:
            context: Current crawl context with history and performance
            
        Returns:
            Tuple of (should_continue, detailed_reasoning)
        """
        try:
            # First check basic constraints (budget, etc.)
            basic_continue, basic_reason = context.should_continue_crawling()
            if not basic_continue:
                return False, f"Basic constraint: {basic_reason}"
            
            # Prepare context for LLM analysis
            context_summary = context.get_context_summary()
            recent_pages = context.crawled_pages[-5:] if len(context.crawled_pages) >= 5 else context.crawled_pages
            
            # Create LLM prompt for decision making
            decision_prompt = self._build_continuation_prompt(
                context_summary=context_summary,
                recent_pages=recent_pages,
                query=context.query
            )
            
            # Get LLM decision
            sampler = get_sampler()
            decision_response = await sampler.request_llm_completion(
                messages=[{"role": "user", "content": decision_prompt}],
                max_tokens=300
            )
            
            # Track token usage
            context.add_llm_token_usage(count_tokens(decision_prompt + decision_response))
            
            # Parse LLM response
            should_continue, reasoning = self._parse_continuation_decision(decision_response)
            
            # Record the decision
            context.record_decision("llm_continuation_decision", {
                "should_continue": should_continue,
                "reasoning": reasoning,
                "context_summary": context_summary
            })
            
            return should_continue, reasoning
            
        except Exception as e:
            logger.error(f"Error in LLM continuation decision: {str(e)}")
            # Fallback to basic logic
            return context.should_continue_crawling()
    
    async def select_next_links(
        self,
        available_links: List[Dict[str, Any]],
        context: CrawlContext,
        max_links: int = 5
    ) -> List[str]:
        """
        Use LLM to intelligently select which links to crawl next.
        
        Args:
            available_links: List of analysed link dictionaries
            context: Current crawl context
            max_links: Maximum number of links to select
            
        Returns:
            List of selected URLs to crawl
        """
        try:
            if not available_links:
                return []
            
            # Filter to only high-potential links
            candidate_links = [
                link for link in available_links 
                if link.get("relevance_score", 0.0) > 0.2 and link.get("should_crawl", False)
            ]
            
            if not candidate_links:
                return []
            
            # If we have few candidates, just return them
            if len(candidate_links) <= max_links:
                return [link["url"] for link in candidate_links]
            
            # Use LLM for complex selection
            selection_prompt = self._build_link_selection_prompt(
                candidate_links=candidate_links,
                context_summary=context.get_context_summary(),
                max_links=max_links
            )
            
            sampler = get_sampler()
            selection_response = await sampler.request_llm_completion(
                messages=[{"role": "user", "content": selection_prompt}],
                max_tokens=400
            )
            
            # Track token usage
            context.add_llm_token_usage(count_tokens(selection_prompt + selection_response))
            
            # Parse LLM response
            selected_urls = self._parse_link_selection(selection_response, candidate_links)
            
            # Record the decision
            context.record_decision("llm_link_selection", {
                "available_links": len(available_links),
                "candidate_links": len(candidate_links),
                "selected_links": len(selected_urls),
                "selection_reasoning": selection_response[:200]
            })
            
            return selected_urls[:max_links]  # Ensure we don't exceed limit
            
        except Exception as e:
            logger.error(f"Error in LLM link selection: {str(e)}")
            # Fallback to simple relevance-based selection
            sorted_links = sorted(
                available_links, 
                key=lambda x: x.get("relevance_score", 0.0), 
                reverse=True
            )
            return [link["url"] for link in sorted_links[:max_links]]
    
    async def evaluate_crawl_completion(
        self,
        context: CrawlContext,
        query: str
    ) -> Dict[str, Any]:
        """
        Use LLM to evaluate how well the crawl has fulfilled the original query.
        
        Args:
            context: Current crawl context
            query: Original search query
            
        Returns:
            Dictionary with completion evaluation and recommendations
        """
        try:
            # Prepare data for evaluation
            successful_pages = [p for p in context.crawled_pages if p.success]
            high_relevance_pages = [p for p in successful_pages if p.is_high_relevance()]
            
            evaluation_prompt = self._build_completion_evaluation_prompt(
                query=query,
                context_summary=context.get_context_summary(),
                successful_pages=successful_pages,
                high_relevance_pages=high_relevance_pages
            )
            
            sampler = get_sampler()
            evaluation_response = await sampler.request_llm_completion(
                messages=[{"role": "user", "content": evaluation_prompt}],
                max_tokens=500
            )
            
            # Track token usage
            context.add_llm_token_usage(count_tokens(evaluation_prompt + evaluation_response))
            
            # Parse evaluation response
            evaluation_results = self._parse_completion_evaluation(evaluation_response)
            
            # Add quantitative metrics
            evaluation_results.update({
                "quantitative_metrics": {
                    "total_pages_crawled": len(context.crawled_pages),
                    "successful_pages": len(successful_pages),
                    "high_relevance_pages": len(high_relevance_pages),
                    "average_relevance": context.get_average_relevance(),
                    "crawl_efficiency": context.get_crawl_efficiency(),
                    "topics_discovered": len(context.topics_discovered),
                    "budget_utilisation": context.budget.get_budget_status()
                }
            })
            
            # Record the evaluation
            context.record_decision("crawl_completion_evaluation", evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in crawl completion evaluation: {str(e)}")
            # Fallback to basic evaluation
            return {
                "completion_score": context.get_average_relevance(),
                "summary": f"Basic evaluation: Found {len([p for p in context.crawled_pages if p.success])} pages",
                "recommendations": ["Review crawl results manually"],
                "query_fulfillment": "partial" if context.get_average_relevance() > 0.4 else "low"
            }
    
    def _build_continuation_prompt(
        self,
        context_summary: Dict[str, Any],
        recent_pages: List[PageCrawlResult],
        query: str
    ) -> str:
        """Build prompt for continuation decision."""
        recent_summaries = []
        for page in recent_pages:
            if page.success:
                recent_summaries.append({
                    "url": page.url,
                    "relevance": page.relevance_score,
                    "topics": page.key_topics[:3]  # Top 3 topics
                })
        
        prompt = f"""
You are an intelligent web crawling assistant. Analyse the current crawling progress and decide whether to continue.

ORIGINAL QUERY: "{query}"

CURRENT PROGRESS:
- Pages crawled: {context_summary['performance']['pages_crawled']}
- Average relevance: {context_summary['performance']['average_relevance']:.2f}
- High relevance pages: {context_summary['performance']['high_relevance_pages']}
- Topics discovered: {context_summary['performance']['topics_discovered']}

RECENT PAGES CRAWLED:
{json.dumps(recent_summaries, indent=2)}

BUDGET STATUS:
- Pages: {context_summary['budget_status']['pages']['used']}/{context_summary['budget_status']['pages']['limit']}
- Time: {context_summary['budget_status']['time']['used']:.1f}s/{context_summary['budget_status']['time']['limit']}s

Should crawling continue? Consider:
1. Are we finding relevant content for the query?
2. Is there potential for finding better content?
3. Are we approaching budget limits?
4. Are recent pages showing diminishing returns?

Respond with: CONTINUE or STOP
Then provide detailed reasoning on the next line.
"""
        return prompt.strip()
    
    def _build_link_selection_prompt(
        self,
        candidate_links: List[Dict[str, Any]],
        context_summary: Dict[str, Any],
        max_links: int
    ) -> str:
        """Build prompt for link selection."""
        links_summary = []
        for i, link in enumerate(candidate_links[:15]):  # Limit to prevent overly long prompts
            links_summary.append({
                "index": i,
                "url": link["url"],
                "relevance": link.get("relevance_score", 0.0),
                "text": link.get("link_text", "")[:100],  # Truncate long text
                "priority": link.get("priority", "medium")
            })
        
        prompt = f"""
You are selecting the most promising links to crawl next.

QUERY: "{context_summary['query']}"
CURRENT PROGRESS: {context_summary['performance']['average_relevance']:.2f} avg relevance, {context_summary['performance']['high_relevance_pages']} high-relevance pages

CANDIDATE LINKS:
{json.dumps(links_summary, indent=2)}

Select the {max_links} most promising links that are likely to contain highly relevant content for the query.
Consider:
1. Link relevance scores
2. Link text indicators
3. URL structure
4. Diversity of content types

Respond with only the indices of selected links (e.g., "0,3,7,11,14"):
"""
        return prompt.strip()
    
    def _build_completion_evaluation_prompt(
        self,
        query: str,
        context_summary: Dict[str, Any],
        successful_pages: List[PageCrawlResult],
        high_relevance_pages: List[PageCrawlResult]
    ) -> str:
        """Build prompt for completion evaluation."""
        page_summaries = []
        for page in high_relevance_pages[:10]:  # Focus on top relevant pages
            page_summaries.append({
                "url": page.url,
                "title": page.title,
                "relevance": page.relevance_score,
                "topics": page.key_topics[:5]
            })
        
        prompt = f"""
Evaluate how well this web crawl fulfilled the original query.

ORIGINAL QUERY: "{query}"

CRAWL RESULTS:
- Total pages: {len(successful_pages)}
- High relevance pages: {len(high_relevance_pages)}
- Average relevance: {context_summary['performance']['average_relevance']:.2f}
- Topics discovered: {list(context_summary['discovered_topics'])[:10]}

TOP RELEVANT PAGES:
{json.dumps(page_summaries, indent=2)}

Provide evaluation in this format:
COMPLETION_SCORE: [0.0-1.0]
QUERY_FULFILLMENT: [excellent/good/partial/poor]
SUMMARY: [2-3 sentence summary of findings]
RECOMMENDATIONS: [List 2-3 specific recommendations]
"""
        return prompt.strip()
    
    def _parse_continuation_decision(self, response: str) -> Tuple[bool, str]:
        """Parse LLM response for continuation decision."""
        try:
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            if not lines:
                return False, "No response from LLM"
            
            decision_line = lines[0].upper()
            should_continue = "CONTINUE" in decision_line
            
            # Get reasoning from remaining lines
            reasoning = " ".join(lines[1:]) if len(lines) > 1 else "No detailed reasoning provided"
            
            return should_continue, reasoning
            
        except Exception as e:
            logger.error(f"Error parsing continuation decision: {str(e)}")
            return False, f"Parse error: {str(e)}"
    
    def _parse_link_selection(
        self,
        response: str,
        candidate_links: List[Dict[str, Any]]
    ) -> List[str]:
        """Parse LLM response for link selection."""
        try:
            # Extract indices from response
            response_clean = response.strip().replace(" ", "")
            if ":" in response_clean:
                response_clean = response_clean.split(":")[-1]  # Take part after colon
            
            indices = []
            for part in response_clean.split(","):
                try:
                    index = int(part.strip())
                    if 0 <= index < len(candidate_links):
                        indices.append(index)
                except ValueError:
                    continue
            
            # Return URLs for selected indices
            return [candidate_links[i]["url"] for i in indices]
            
        except Exception as e:
            logger.error(f"Error parsing link selection: {str(e)}")
            # Fallback: return top links by relevance
            sorted_links = sorted(
                candidate_links,
                key=lambda x: x.get("relevance_score", 0.0),
                reverse=True
            )
            return [link["url"] for link in sorted_links[:5]]
    
    def _parse_completion_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for completion evaluation."""
        try:
            evaluation = {
                "completion_score": 0.5,
                "query_fulfillment": "partial",
                "summary": "",
                "recommendations": []
            }
            
            lines = response.strip().split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("COMPLETION_SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                        evaluation["completion_score"] = max(0.0, min(1.0, score))
                    except (ValueError, IndexError):
                        pass
                
                elif line.startswith("QUERY_FULFILLMENT:"):
                    fulfillment = line.split(":", 1)[1].strip().lower()
                    if fulfillment in ["excellent", "good", "partial", "poor"]:
                        evaluation["query_fulfillment"] = fulfillment
                
                elif line.startswith("SUMMARY:"):
                    evaluation["summary"] = line.split(":", 1)[1].strip()
                
                elif line.startswith("RECOMMENDATIONS:"):
                    current_section = "recommendations"
                
                elif current_section == "recommendations" and line:
                    # Clean up recommendation text
                    rec = line.lstrip("- ").strip()
                    if rec:
                        evaluation["recommendations"].append(rec)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error parsing completion evaluation: {str(e)}")
            return {
                "completion_score": 0.5,
                "query_fulfillment": "partial",
                "summary": "Unable to parse LLM evaluation",
                "recommendations": ["Review results manually"]
            }


# Global instance for decision making
_decision_maker = None

def get_llm_decision_maker() -> LLMCrawlDecisionMaker:
    """Get the global LLM decision maker instance."""
    global _decision_maker
    if _decision_maker is None:
        _decision_maker = LLMCrawlDecisionMaker()
    return _decision_maker 