"""
Intelligent Link Selection for crawl4ai-mcp.

Provides smart link analysis and selection for adaptive crawling,
using relevance scoring to prioritise which links to follow.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin, urlparse
from .relevance_scorer import get_relevance_scorer

logger = logging.getLogger(__name__)


class LinkSelector:
    """Handles intelligent link selection for adaptive crawling."""
    
    def __init__(self):
        self.relevance_threshold = 0.3
        self.max_links_per_page = 10
        self.domain_preference_weight = 0.2
        self.depth_penalty = 0.1
    
    async def analyse_links_for_crawling(
        self,
        links: List[str],
        page_content: str,
        query: str,
        current_url: str = "",
        page_title: str = ""
    ) -> List[Dict[str, Any]]:
        """Analyse links and score them for crawling relevance.
        
        Args:
            links: List of URLs to analyse
            page_content: Content of the page containing the links
            query: Search query for relevance analysis
            current_url: URL of the current page
            page_title: Title of the current page
            
        Returns:
            List of link analysis dictionaries with relevance scores
        """
        try:
            scorer = get_relevance_scorer()
            analysed_links = []
            
            # Extract link context from page content
            link_contexts = self._extract_link_contexts(page_content, links)
            
            for link in links:
                try:
                    # Skip invalid or problematic links
                    if not self._is_valid_link(link):
                        continue
                    
                    # Get absolute URL
                    absolute_url = urljoin(current_url, link) if current_url else link
                    
                    # Extract link text (simplified - in real implementation would parse HTML)
                    link_text = link_contexts.get(link, {}).get("text", "")
                    link_context = link_contexts.get(link, {}).get("context", "")
                    
                    # Score link relevance
                    relevance_analysis = await scorer.score_link_relevance(
                        link_text=link_text,
                        link_url=absolute_url,
                        query=query,
                        page_context=link_context
                    )
                    
                    # Add additional analysis
                    domain_score = self._calculate_domain_score(absolute_url, current_url)
                    depth_penalty = self._calculate_depth_penalty(absolute_url)
                    
                    # Combine scores
                    final_score = (
                        relevance_analysis["relevance_score"] * 0.7 +
                        domain_score * self.domain_preference_weight +
                        depth_penalty * self.depth_penalty
                    )
                    
                    link_analysis = {
                        "url": absolute_url,
                        "original_url": link,
                        "link_text": link_text,
                        "relevance_score": final_score,
                        "should_crawl": final_score > self.relevance_threshold,
                        "analysis": relevance_analysis,
                        "domain_score": domain_score,
                        "depth_penalty": depth_penalty,
                        "context": link_context[:200] if link_context else "",  # Truncate for brevity
                        "priority": self._calculate_priority(final_score, link_text, absolute_url)
                    }
                    
                    analysed_links.append(link_analysis)
                    
                except Exception as e:
                    logger.error(f"Error analysing link {link}: {str(e)}")
                    analysed_links.append({
                        "url": link,
                        "original_url": link,
                        "link_text": "",
                        "relevance_score": 0.0,
                        "should_crawl": False,
                        "error": str(e)
                    })
            
            # Sort by relevance score
            analysed_links.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            
            logger.info(f"Analysed {len(analysed_links)} links, {len([l for l in analysed_links if l.get('should_crawl', False)])} recommended for crawling")
            
            return analysed_links
            
        except Exception as e:
            logger.error(f"Error analysing links for crawling: {str(e)}")
            return []
    
    def _extract_link_contexts(
        self, 
        page_content: str, 
        links: List[str]
    ) -> Dict[str, Dict[str, str]]:
        """Extract context around links from page content.
        
        Args:
            page_content: HTML content of the page
            links: List of links to find context for
            
        Returns:
            Dictionary mapping links to their context information
        """
        contexts = {}
        
        # This is a simplified implementation
        # In a real implementation, would use proper HTML parsing
        try:
            import re
            
            for link in links:
                # Find link in content and extract surrounding text
                escaped_link = re.escape(link)
                pattern = rf'.{{0,100}}{escaped_link}.{{0,100}}'
                
                matches = re.findall(pattern, page_content, re.IGNORECASE | re.DOTALL)
                if matches:
                    context = matches[0]
                    
                    # Try to extract link text from HTML
                    link_text_pattern = rf'<a[^>]*href=["\']?[^"\']*{escaped_link}[^"\']*["\']?[^>]*>([^<]*)</a>'
                    text_matches = re.findall(link_text_pattern, page_content, re.IGNORECASE)
                    link_text = text_matches[0] if text_matches else ""
                    
                    contexts[link] = {
                        "text": link_text.strip(),
                        "context": context.strip()
                    }
                else:
                    contexts[link] = {"text": "", "context": ""}
                    
        except Exception as e:
            logger.error(f"Error extracting link contexts: {str(e)}")
        
        return contexts
    
    def _is_valid_link(self, link: str) -> bool:
        """Check if a link is valid for crawling.
        
        Args:
            link: URL to validate
            
        Returns:
            True if link is valid for crawling
        """
        if not link or not isinstance(link, str):
            return False
        
        # Skip common non-content links
        invalid_patterns = [
            'javascript:', 'mailto:', 'tel:', 'ftp:', 'file:',
            '#', 'data:', 'blob:', 'about:',
            '.css', '.js', '.ico', '.png', '.jpg', '.jpeg', '.gif', '.svg',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar'
        ]
        
        link_lower = link.lower()
        for pattern in invalid_patterns:
            if pattern in link_lower:
                return False
        
        # Check URL structure
        try:
            parsed = urlparse(link)
            if parsed.scheme and parsed.scheme not in ['http', 'https']:
                return False
        except Exception:
            return False
        
        return True
    
    def _calculate_domain_score(self, link_url: str, current_url: str) -> float:
        """Calculate domain preference score.
        
        Args:
            link_url: URL of the link
            current_url: URL of the current page
            
        Returns:
            Domain preference score (0.0 to 1.0)
        """
        try:
            if not current_url:
                return 0.5  # Neutral score if no current URL
            
            link_domain = urlparse(link_url).netloc.lower()
            current_domain = urlparse(current_url).netloc.lower()
            
            if link_domain == current_domain:
                return 1.0  # Same domain - highest preference
            elif link_domain.endswith('.' + current_domain) or current_domain.endswith('.' + link_domain):
                return 0.8  # Subdomain relationship
            else:
                return 0.3  # External domain - lower preference
                
        except Exception as e:
            logger.error(f"Error calculating domain score: {str(e)}")
            return 0.5
    
    def _calculate_depth_penalty(self, url: str) -> float:
        """Calculate penalty based on URL depth.
        
        Args:
            url: URL to analyse
            
        Returns:
            Depth penalty (negative value, 0.0 to -0.5)
        """
        try:
            parsed = urlparse(url)
            path_parts = [part for part in parsed.path.split('/') if part]
            depth = len(path_parts)
            
            # Penalty increases with depth
            if depth <= 2:
                return 0.0
            elif depth <= 4:
                return -0.1
            elif depth <= 6:
                return -0.2
            else:
                return -0.3
                
        except Exception as e:
            logger.error(f"Error calculating depth penalty: {str(e)}")
            return 0.0
    
    def _calculate_priority(self, relevance_score: float, link_text: str, url: str) -> str:
        """Calculate crawling priority category.
        
        Args:
            relevance_score: Relevance score for the link
            link_text: Text of the link
            url: URL of the link
            
        Returns:
            Priority category string
        """
        if relevance_score > 0.7:
            return "high"
        elif relevance_score > 0.5:
            return "medium"
        elif relevance_score > 0.3:
            return "low"
        else:
            return "skip"
    
    def should_crawl_link(
        self, 
        link_data: Dict[str, Any], 
        crawl_context: Dict[str, Any]
    ) -> bool:
        """Determine if a link should be crawled based on context.
        
        Args:
            link_data: Link analysis data
            crawl_context: Current crawling context
            
        Returns:
            True if link should be crawled
        """
        try:
            # Basic relevance check
            if not link_data.get("should_crawl", False):
                return False
            
            # Check crawl budget
            pages_crawled = crawl_context.get("pages_crawled", 0)
            max_pages = crawl_context.get("max_pages", 20)
            
            if pages_crawled >= max_pages:
                return False
            
            # Check if already crawled
            crawled_urls = crawl_context.get("crawled_urls", set())
            if link_data.get("url") in crawled_urls:
                return False
            
            # Priority-based decision
            priority = link_data.get("priority", "skip")
            relevance_score = link_data.get("relevance_score", 0.0)
            
            # High priority links are always crawled (if budget allows)
            if priority == "high":
                return True
            
            # Medium priority links crawled if we have budget
            if priority == "medium" and pages_crawled < (max_pages * 0.8):
                return True
            
            # Low priority links only if we have plenty of budget
            if priority == "low" and pages_crawled < (max_pages * 0.5):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining if link should be crawled: {str(e)}")
            return False
    
    def prioritise_links(
        self, 
        analysed_links: List[Dict[str, Any]], 
        max_links: int = 5
    ) -> List[str]:
        """Prioritise and select the best links for crawling.
        
        Args:
            analysed_links: List of analysed link dictionaries
            max_links: Maximum number of links to return
            
        Returns:
            List of prioritised URLs for crawling
        """
        try:
            # Filter to only crawlable links
            crawlable_links = [
                link for link in analysed_links 
                if link.get("should_crawl", False)
            ]
            
            # Sort by relevance score (already sorted from analysis)
            crawlable_links.sort(
                key=lambda x: x.get("relevance_score", 0.0), 
                reverse=True
            )
            
            # Apply diversity - avoid too many links from same domain/path
            selected_links = []
            used_domains = set()
            used_paths = set()
            
            for link in crawlable_links:
                if len(selected_links) >= max_links:
                    break
                
                url = link.get("url", "")
                if not url:
                    continue
                
                try:
                    parsed = urlparse(url)
                    domain = parsed.netloc.lower()
                    path_base = '/'.join(parsed.path.split('/')[:3])  # First 2 path segments
                    
                    # Prefer diversity in domains and paths
                    domain_count = sum(1 for d in used_domains if d == domain)
                    path_count = sum(1 for p in used_paths if p == path_base)
                    
                    # Allow some repetition but prefer diversity
                    if domain_count < 3 and path_count < 2:
                        selected_links.append(url)
                        used_domains.add(domain)
                        used_paths.add(path_base)
                    elif len(selected_links) < max_links // 2:
                        # Fill remaining slots if we haven't hit half capacity
                        selected_links.append(url)
                        
                except Exception as e:
                    logger.error(f"Error processing link for prioritisation: {str(e)}")
                    if len(selected_links) < max_links:
                        selected_links.append(url)
            
            logger.info(f"Prioritised {len(selected_links)} links from {len(crawlable_links)} crawlable options")
            return selected_links
            
        except Exception as e:
            logger.error(f"Error prioritising links: {str(e)}")
            # Fallback to simple selection
            return [
                link.get("url", "") for link in analysed_links[:max_links]
                if link.get("should_crawl", False) and link.get("url")
            ]
    
    async def select_best_links(
        self,
        links: List[str],
        page_content: str,
        query: str,
        crawl_context: Dict[str, Any],
        current_url: str = ""
    ) -> List[str]:
        """Complete link selection pipeline.
        
        Args:
            links: List of URLs found on the page
            page_content: Content of the current page
            query: Search query for relevance
            crawl_context: Current crawling context
            current_url: URL of the current page
            
        Returns:
            List of selected URLs for crawling
        """
        try:
            # Analyse all links
            analysed_links = await self.analyse_links_for_crawling(
                links=links,
                page_content=page_content,
                query=query,
                current_url=current_url
            )
            
            # Filter based on crawl context
            filtered_links = []
            for link_data in analysed_links:
                if self.should_crawl_link(link_data, crawl_context):
                    filtered_links.append(link_data)
            
            # Prioritise and select final links
            max_links = min(
                self.max_links_per_page,
                crawl_context.get("remaining_budget", self.max_links_per_page)
            )
            
            selected_urls = self.prioritise_links(filtered_links, max_links)
            
            logger.info(f"Selected {len(selected_urls)} links for crawling from {len(links)} total links")
            return selected_urls
            
        except Exception as e:
            logger.error(f"Error in link selection pipeline: {str(e)}")
            return []


# Global link selector instance
_link_selector: Optional[LinkSelector] = None


def get_link_selector() -> LinkSelector:
    """Get the global link selector instance.
    
    Returns:
        LinkSelector instance
    """
    global _link_selector
    if _link_selector is None:
        _link_selector = LinkSelector()
    return _link_selector


# Standalone function for backward compatibility with tests
async def analyse_links_for_crawling(
    links: List[str],
    page_content: str,
    query: str,
    current_url: str = "",
    page_title: str = ""
) -> List[Dict[str, Any]]:
    """Standalone function to analyse links for crawling.
    
    Args:
        links: List of URLs to analyse
        page_content: Content of the page containing the links
        query: Search query for relevance analysis
        current_url: URL of the current page
        page_title: Title of the current page
        
    Returns:
        List of link analysis dictionaries with relevance scores
    """
    selector = get_link_selector()
    return await selector.analyse_links_for_crawling(
        links=links,
        page_content=page_content,
        query=query,
        current_url=current_url,
        page_title=page_title
    ) 