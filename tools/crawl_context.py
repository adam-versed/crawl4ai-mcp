"""
Crawl Context Management for crawl4ai-mcp.

Provides comprehensive context tracking and intelligent decision-making
for adaptive crawling operations with budget and performance monitoring.
"""

import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CrawlStatus(Enum):
    """Status of a crawl operation."""
    INITIALISING = "initialising"
    ACTIVE_CRAWLING = "active_crawling"
    BUDGET_LIMITED = "budget_limited"
    DIMINISHING_RETURNS = "diminishing_returns"
    TARGET_ACHIEVED = "target_achieved"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BudgetLimits:
    """Budget constraints for crawling operations."""
    max_pages: int = 20
    max_tokens_per_llm_call: int = 1000
    max_total_llm_tokens: int = 10000
    max_crawl_time_seconds: int = 300
    max_depth: int = 5
    
    # Current usage tracking
    pages_crawled: int = 0
    llm_tokens_used: int = 0
    crawl_start_time: float = field(default_factory=time.time)
    current_depth: int = 0
    
    def is_page_budget_exceeded(self) -> bool:
        """Check if page budget is exceeded."""
        return self.pages_crawled >= self.max_pages
    
    def is_token_budget_exceeded(self) -> bool:
        """Check if LLM token budget is exceeded."""
        return self.llm_tokens_used >= self.max_total_llm_tokens
    
    def is_time_budget_exceeded(self) -> bool:
        """Check if time budget is exceeded."""
        return (time.time() - self.crawl_start_time) >= self.max_crawl_time_seconds
    
    def is_depth_limit_reached(self) -> bool:
        """Check if maximum depth is reached."""
        return self.current_depth >= self.max_depth
    
    def is_any_budget_exceeded(self) -> bool:
        """Check if any budget constraint is exceeded."""
        return (self.is_page_budget_exceeded() or 
                self.is_token_budget_exceeded() or 
                self.is_time_budget_exceeded() or
                self.is_depth_limit_reached())
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget utilisation status."""
        current_time = time.time()
        return {
            "pages": {
                "used": self.pages_crawled,
                "limit": self.max_pages,
                "percentage": (self.pages_crawled / self.max_pages) * 100,
                "exceeded": self.is_page_budget_exceeded()
            },
            "tokens": {
                "used": self.llm_tokens_used,
                "limit": self.max_total_llm_tokens,
                "percentage": (self.llm_tokens_used / self.max_total_llm_tokens) * 100,
                "exceeded": self.is_token_budget_exceeded()
            },
            "time": {
                "used": current_time - self.crawl_start_time,
                "limit": self.max_crawl_time_seconds,
                "percentage": ((current_time - self.crawl_start_time) / self.max_crawl_time_seconds) * 100,
                "exceeded": self.is_time_budget_exceeded()
            },
            "depth": {
                "current": self.current_depth,
                "limit": self.max_depth,
                "exceeded": self.is_depth_limit_reached()
            }
        }


@dataclass
class PageCrawlResult:
    """Result of crawling a single page."""
    url: str
    success: bool
    relevance_score: float = 0.0
    depth: int = 0
    title: str = ""
    key_topics: List[str] = field(default_factory=list)
    links_found: int = 0
    content_length: int = 0
    error: Optional[str] = None
    crawl_timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    
    def is_high_relevance(self, threshold: float = 0.7) -> bool:
        """Check if page has high relevance."""
        return self.success and self.relevance_score >= threshold
    
    def is_relevant(self, threshold: float = 0.4) -> bool:
        """Check if page meets basic relevance threshold."""
        return self.success and self.relevance_score >= threshold


class CrawlContext:
    """Comprehensive context tracking for crawling operations."""
    
    def __init__(
        self, 
        starting_url: str, 
        query: str, 
        budget_limits: Optional[BudgetLimits] = None,
        target_keywords: Optional[List[str]] = None
    ):
        self.starting_url = starting_url
        self.query = query
        self.budget = budget_limits or BudgetLimits()
        self.target_keywords = target_keywords or []
        
        # Crawl state tracking
        self.status = CrawlStatus.INITIALISING
        self.crawled_pages: List[PageCrawlResult] = []
        self.crawled_urls: Set[str] = set()
        self.pending_urls: List[Tuple[float, int, str]] = []  # (priority, depth, url)
        
        # Performance metrics
        self.total_relevance_score = 0.0
        self.high_relevance_pages = 0
        self.topics_discovered: Set[str] = set()
        
        # Decision tracking
        self.decision_history: List[Dict[str, Any]] = []
        self.last_performance_check = time.time()
        
        logger.info(f"Initialised crawl context for query: '{query}' starting from {starting_url}")
    
    def add_crawled_page(self, page_result: PageCrawlResult) -> None:
        """Add a crawled page result to the context."""
        if page_result.url in self.crawled_urls:
            logger.warning(f"Page {page_result.url} already crawled, skipping duplicate")
            return
        
        self.crawled_pages.append(page_result)
        self.crawled_urls.add(page_result.url)
        self.budget.pages_crawled += 1
        
        if page_result.success:
            self.total_relevance_score += page_result.relevance_score
            if page_result.is_high_relevance():
                self.high_relevance_pages += 1
            
            # Track discovered topics
            self.topics_discovered.update(page_result.key_topics)
        
        # Update depth tracking
        if page_result.depth > self.budget.current_depth:
            self.budget.current_depth = page_result.depth
        
        logger.debug(f"Added page {page_result.url} (relevance: {page_result.relevance_score:.2f})")
    
    def add_llm_token_usage(self, tokens_used: int) -> None:
        """Track LLM token usage."""
        self.budget.llm_tokens_used += tokens_used
        logger.debug(f"Added {tokens_used} LLM tokens, total: {self.budget.llm_tokens_used}")
    
    def get_average_relevance(self) -> float:
        """Calculate average relevance score of successful pages."""
        successful_pages = [p for p in self.crawled_pages if p.success]
        if not successful_pages:
            return 0.0
        return sum(p.relevance_score for p in successful_pages) / len(successful_pages)
    
    def get_crawl_efficiency(self) -> float:
        """Calculate crawl efficiency (high relevance pages / total successful pages)."""
        successful_pages = [p for p in self.crawled_pages if p.success]
        if not successful_pages:
            return 0.0
        return self.high_relevance_pages / len(successful_pages)
    
    def detect_diminishing_returns(self, window_size: int = 5, threshold: float = 0.3) -> bool:
        """
        Detect if recent pages show diminishing returns in relevance.
        
        Args:
            window_size: Number of recent pages to analyse
            threshold: Minimum average relevance threshold
            
        Returns:
            True if diminishing returns detected
        """
        if len(self.crawled_pages) < window_size:
            return False
        
        recent_pages = self.crawled_pages[-window_size:]
        successful_recent = [p for p in recent_pages if p.success]
        
        if not successful_recent:
            return True  # No successful pages recently
        
        recent_avg_relevance = sum(p.relevance_score for p in successful_recent) / len(successful_recent)
        return recent_avg_relevance < threshold
    
    def should_continue_crawling(self) -> Tuple[bool, str]:
        """
        Determine if crawling should continue based on current context.
        
        Returns:
            Tuple of (should_continue, reasoning)
        """
        # Check budget constraints first
        if self.budget.is_any_budget_exceeded():
            budget_status = self.budget.get_budget_status()
            exceeded_budgets = [k for k, v in budget_status.items() if v.get("exceeded", False)]
            return False, f"Budget exceeded: {', '.join(exceeded_budgets)}"
        
        # Check if we have pending URLs to crawl
        if not self.pending_urls:
            return False, "No more URLs in crawl queue"
        
        # If no pages have been crawled yet, always continue (initial crawl attempt)
        if self.budget.pages_crawled == 0:
            return True, "Initial crawl attempt - no pages crawled yet"
        
        # Check if we have found sufficient high-relevance content
        if self.high_relevance_pages >= 3 and self.get_crawl_efficiency() > 0.6:
            return False, "Sufficient high-relevance content found"
        
        # Check for diminishing returns
        if self.detect_diminishing_returns():
            return False, "Diminishing returns detected in recent pages"
        
        # Continue if we haven't found enough relevant content yet
        if self.get_average_relevance() < 0.4 and self.budget.pages_crawled < self.budget.max_pages * 0.8:
            return True, "Searching for more relevant content"
        
        # Continue if we have budget and the crawl is performing well
        if self.get_average_relevance() >= 0.4:
            return True, "Crawl performing well, continuing"
        
        return False, "No clear reason to continue crawling"
    
    def prioritise_pending_urls(self, max_urls: int = 10) -> List[str]:
        """
        Get the highest priority URLs from the pending queue.
        
        Args:
            max_urls: Maximum number of URLs to return
            
        Returns:
            List of URLs ordered by priority
        """
        # Sort by priority (descending) and return top URLs
        self.pending_urls.sort(key=lambda x: x[0], reverse=True)
        top_urls = []
        
        for priority, depth, url in self.pending_urls[:max_urls]:
            if url not in self.crawled_urls:
                top_urls.append(url)
        
        return top_urls
    
    def add_pending_urls(self, urls_with_priority: List[Tuple[float, int, str]]) -> None:
        """Add URLs to the pending crawl queue with their priorities."""
        for priority, depth, url in urls_with_priority:
            if url not in self.crawled_urls:
                self.pending_urls.append((priority, depth, url))
        
        logger.debug(f"Added {len(urls_with_priority)} URLs to pending queue")
    
    def record_decision(self, decision_type: str, details: Dict[str, Any]) -> None:
        """Record a decision made during crawling for analysis."""
        decision_record = {
            "timestamp": time.time(),
            "decision_type": decision_type,
            "pages_crawled": self.budget.pages_crawled,
            "average_relevance": self.get_average_relevance(),
            "details": details
        }
        self.decision_history.append(decision_record)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the current crawl context."""
        return {
            "starting_url": self.starting_url,
            "query": self.query,
            "status": self.status.value,
            "budget_status": self.budget.get_budget_status(),
            "performance": {
                "pages_crawled": len(self.crawled_pages),
                "successful_pages": len([p for p in self.crawled_pages if p.success]),
                "high_relevance_pages": self.high_relevance_pages,
                "average_relevance": self.get_average_relevance(),
                "crawl_efficiency": self.get_crawl_efficiency(),
                "topics_discovered": len(self.topics_discovered)
            },
            "queue_status": {
                "pending_urls": len(self.pending_urls),
                "crawled_urls": len(self.crawled_urls)
            },
            "target_keywords": self.target_keywords,
            "discovered_topics": list(self.topics_discovered)
        }
    
    def update_status(self, new_status: CrawlStatus, reason: str = "") -> None:
        """Update the crawl status with optional reasoning."""
        old_status = self.status
        self.status = new_status
        
        self.record_decision("status_change", {
            "old_status": old_status.value,
            "new_status": new_status.value,
            "reason": reason
        })
        
        logger.info(f"Crawl status changed from {old_status.value} to {new_status.value}: {reason}")


def create_crawl_context(
    starting_url: str,
    query: str,
    max_pages: int = 20,
    max_tokens: int = 10000,
    max_time_seconds: int = 300,
    target_keywords: Optional[List[str]] = None
) -> CrawlContext:
    """
    Create a new crawl context with specified parameters.
    
    Args:
        starting_url: URL to start crawling from
        query: Search query driving the crawl
        max_pages: Maximum pages to crawl
        max_tokens: Maximum LLM tokens to use
        max_time_seconds: Maximum crawl time
        target_keywords: Optional specific keywords to target
        
    Returns:
        Configured CrawlContext instance
    """
    budget = BudgetLimits(
        max_pages=max_pages,
        max_total_llm_tokens=max_tokens,
        max_crawl_time_seconds=max_time_seconds
    )
    
    context = CrawlContext(
        starting_url=starting_url,
        query=query,
        budget_limits=budget,
        target_keywords=target_keywords
    )
    
    context.update_status(CrawlStatus.ACTIVE_CRAWLING, "Context initialised")
    return context 