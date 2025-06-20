"""
Relevance Scoring System for crawl4ai-mcp.

Provides intelligent scoring of page content and links for query relevance,
enabling adaptive crawling decisions based on content analysis.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse, urljoin
from .utils import count_tokens, truncate_to_token_limit
from .mcp_sampler import get_sampler

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """Handles relevance scoring for content and links."""
    
    def __init__(self):
        self.keyword_weight = 0.3
        self.semantic_weight = 0.5
        self.structural_weight = 0.2
        self.min_content_length = 50
    
    async def score_page_relevance(
        self, 
        page_content: str, 
        query: str, 
        target_keywords: Optional[List[str]] = None,
        page_title: str = "",
        page_url: str = ""
    ) -> Dict[str, Any]:
        """Score how relevant a page is to a given query.
        
        Args:
            page_content: The page content to analyse
            query: The search query
            target_keywords: Optional list of specific keywords to look for
            page_title: Optional page title for additional context
            page_url: Optional page URL for structural analysis
            
        Returns:
            Dictionary with relevance score and analysis details
        """
        try:
            if not page_content.strip() or len(page_content) < self.min_content_length:
                return {
                    "relevance_score": 0.0,
                    "reasoning": "Insufficient content for analysis",
                    "keyword_score": 0.0,
                    "semantic_score": 0.0,
                    "structural_score": 0.0,
                    "key_topics": [],
                    "matches_query": False
                }
            
            # Calculate keyword-based score
            keyword_score = self._calculate_keyword_score(
                page_content, query, target_keywords, page_title
            )
            
            # Calculate semantic score using LLM
            semantic_analysis = await self._calculate_semantic_score(
                page_content, query
            )
            semantic_score = semantic_analysis.get("relevance_score", 0.0)
            
            # Calculate structural score
            structural_score = self._calculate_structural_score(
                page_content, page_url, query
            )
            
            # Combine scores
            final_score = (
                keyword_score * self.keyword_weight +
                semantic_score * self.semantic_weight +
                structural_score * self.structural_weight
            )
            
            # Ensure score is between 0 and 1
            final_score = max(0.0, min(1.0, final_score))
            
            return {
                "relevance_score": final_score,
                "reasoning": semantic_analysis.get("reasoning", "Automated relevance analysis"),
                "keyword_score": keyword_score,
                "semantic_score": semantic_score,
                "structural_score": structural_score,
                "key_topics": semantic_analysis.get("key_topics", []),
                "matches_query": final_score > 0.5,
                "analysis_details": {
                    "content_length": len(page_content),
                    "query": query,
                    "target_keywords": target_keywords or [],
                    "page_title": page_title,
                    "page_url": page_url
                }
            }
            
        except Exception as e:
            logger.error(f"Error scoring page relevance: {str(e)}")
            return {
                "relevance_score": 0.0,
                "reasoning": f"Error in relevance analysis: {str(e)}",
                "keyword_score": 0.0,
                "semantic_score": 0.0,
                "structural_score": 0.0,
                "key_topics": [],
                "matches_query": False,
                "error": str(e)
            }
    
    def _calculate_keyword_score(
        self, 
        content: str, 
        query: str, 
        target_keywords: Optional[List[str]] = None,
        page_title: str = ""
    ) -> float:
        """Calculate relevance score based on keyword matching.
        
        Args:
            content: Page content
            query: Search query  
            target_keywords: Optional specific keywords
            page_title: Page title for additional weight
            
        Returns:
            Keyword relevance score (0.0 to 1.0)
        """
        try:
            # Normalise text for comparison
            content_lower = content.lower()
            query_lower = query.lower()
            title_lower = page_title.lower()
            
            # Extract query terms
            query_terms = re.findall(r'\b\w+\b', query_lower)
            if target_keywords:
                query_terms.extend([kw.lower() for kw in target_keywords])
            
            if not query_terms:
                return 0.0
            
            total_score = 0.0
            total_weight = 0.0
            
            for term in query_terms:
                if len(term) < 2:  # Skip very short terms
                    continue
                
                # Count occurrences in content
                content_matches = len(re.findall(r'\b' + re.escape(term) + r'\b', content_lower))
                title_matches = len(re.findall(r'\b' + re.escape(term) + r'\b', title_lower))
                
                # Weight title matches more heavily
                term_score = content_matches + (title_matches * 3)
                
                # Normalise by content length (per 1000 characters)
                normalised_score = min(1.0, term_score / max(1, len(content) / 1000))
                
                total_score += normalised_score
                total_weight += 1.0
            
            return min(1.0, total_score / total_weight) if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating keyword score: {str(e)}")
            return 0.0
    
    async def _calculate_semantic_score(
        self, 
        content: str, 
        query: str
    ) -> Dict[str, Any]:
        """Calculate semantic relevance using LLM analysis.
        
        Args:
            content: Page content
            query: Search query
            
        Returns:
            Dictionary with semantic analysis results
        """
        try:
            # Truncate content if too long
            max_tokens = 6000
            if count_tokens(content) > max_tokens:
                content = truncate_to_token_limit(content, max_tokens)
            
            sampler = get_sampler()
            return await sampler.analyse_relevance(
                content=content,
                query=query,
                max_tokens=300
            )
            
        except Exception as e:
            logger.error(f"Error calculating semantic score: {str(e)}")
            return {
                "relevance_score": 0.0,
                "reasoning": f"Semantic analysis failed: {str(e)}",
                "key_topics": [],
                "matches_query": False
            }
    
    def _calculate_structural_score(
        self, 
        content: str, 
        page_url: str, 
        query: str
    ) -> float:
        """Calculate relevance based on page structure and URL.
        
        Args:
            content: Page content
            page_url: Page URL
            query: Search query
            
        Returns:
            Structural relevance score (0.0 to 1.0)
        """
        try:
            score = 0.0
            
            # URL relevance
            if page_url:
                url_lower = page_url.lower()
                query_terms = re.findall(r'\b\w+\b', query.lower())
                
                for term in query_terms:
                    if len(term) > 2 and term in url_lower:
                        score += 0.2
            
            # Content structure indicators
            content_lower = content.lower()
            
            # Headers and structure
            if re.search(r'<h[1-6]', content_lower) or re.search(r'^#+\s', content_lower, re.MULTILINE):
                score += 0.1
            
            # Lists and organisation
            if re.search(r'<[uo]l>', content_lower) or re.search(r'^\s*[-*+]\s', content_lower, re.MULTILINE):
                score += 0.1
            
            # Links and navigation
            if re.search(r'<a\s+href', content_lower):
                score += 0.1
            
            # Content length bonus (well-developed pages)
            if len(content) > 2000:
                score += 0.1
            elif len(content) > 5000:
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating structural score: {str(e)}")
            return 0.0
    
    async def score_link_relevance(
        self, 
        link_text: str, 
        link_url: str, 
        query: str,
        page_context: str = ""
    ) -> Dict[str, Any]:
        """Score how relevant a link is for crawling based on query.
        
        Args:
            link_text: The anchor text of the link
            link_url: The URL the link points to
            query: The search query
            page_context: Optional context from the page containing the link
            
        Returns:
            Dictionary with link relevance analysis
        """
        try:
            # Basic keyword matching in link text and URL
            text_score = self._calculate_keyword_score(
                link_text + " " + link_url, query
            )
            
            # URL structure analysis
            url_score = self._analyse_url_structure(link_url, query)
            
            # Context relevance if available
            context_score = 0.0
            if page_context:
                # Look for context around the link
                context_score = self._calculate_keyword_score(
                    page_context, query
                ) * 0.5  # Weight context less than direct link content
            
            # Combine scores
            final_score = (text_score * 0.5) + (url_score * 0.3) + (context_score * 0.2)
            final_score = max(0.0, min(1.0, final_score))
            
            return {
                "relevance_score": final_score,
                "text_score": text_score,
                "url_score": url_score,
                "context_score": context_score,
                "should_crawl": final_score > 0.3,
                "link_text": link_text,
                "link_url": link_url,
                "reasoning": f"Link relevance based on text ({text_score:.2f}), URL ({url_score:.2f}), and context ({context_score:.2f})"
            }
            
        except Exception as e:
            logger.error(f"Error scoring link relevance: {str(e)}")
            return {
                "relevance_score": 0.0,
                "text_score": 0.0,
                "url_score": 0.0,
                "context_score": 0.0,
                "should_crawl": False,
                "link_text": link_text,
                "link_url": link_url,
                "error": str(e)
            }
    
    def _analyse_url_structure(self, url: str, query: str) -> float:
        """Analyse URL structure for query relevance.
        
        Args:
            url: URL to analyse
            query: Search query
            
        Returns:
            URL relevance score (0.0 to 1.0)
        """
        try:
            parsed = urlparse(url)
            url_parts = [
                parsed.path.lower(),
                parsed.query.lower(),
                parsed.fragment.lower()
            ]
            
            query_terms = re.findall(r'\b\w+\b', query.lower())
            score = 0.0
            
            for part in url_parts:
                for term in query_terms:
                    if len(term) > 2 and term in part:
                        score += 0.2
            
            # Bonus for relevant file extensions
            relevant_extensions = ['.html', '.htm', '.php', '.asp', '.jsp']
            if any(ext in parsed.path.lower() for ext in relevant_extensions):
                score += 0.1
            
            # Penalty for clearly irrelevant paths
            irrelevant_patterns = [
                'login', 'register', 'cart', 'checkout', 'admin',
                'api', 'css', 'js', 'img', 'images', 'static'
            ]
            if any(pattern in parsed.path.lower() for pattern in irrelevant_patterns):
                score *= 0.5
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error analysing URL structure: {str(e)}")
            return 0.0
    
    async def extract_page_topics(
        self, 
        content: str, 
        max_topics: int = 10
    ) -> List[str]:
        """Extract main topics from page content using LLM analysis.
        
        Args:
            content: Page content to analyse
            max_topics: Maximum number of topics to extract
            
        Returns:
            List of extracted topics
        """
        try:
            # Truncate content if too long
            max_tokens = 4000
            if count_tokens(content) > max_tokens:
                content = truncate_to_token_limit(content, max_tokens)
            
            sampler = get_sampler()
            
            system_prompt = f"""You are an expert at topic extraction. 
            Extract the {max_topics} most important topics from the given content.
            Respond with a JSON array of topic strings, e.g., ["topic1", "topic2", "topic3"]"""
            
            prompt = f"Extract the main topics from this content:\n\n{content}"
            
            response = await sampler.request_llm_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                system_prompt=system_prompt,
                temperature=0.1
            )
            
            # Parse JSON response
            import json
            try:
                topics = json.loads(response)
                if isinstance(topics, list):
                    return topics[:max_topics]
                else:
                    logger.warning(f"Unexpected topic extraction format: {response}")
                    return []
            except json.JSONDecodeError:
                # Fallback to simple extraction
                logger.warning(f"Failed to parse topic extraction JSON: {response}")
                return self._extract_topics_fallback(content, max_topics)
                
        except Exception as e:
            logger.error(f"Error extracting page topics: {str(e)}")
            return self._extract_topics_fallback(content, max_topics)
    
    def _extract_topics_fallback(self, content: str, max_topics: int) -> List[str]:
        """Fallback topic extraction using simple keyword analysis.
        
        Args:
            content: Content to analyse
            max_topics: Maximum topics to return
            
        Returns:
            List of extracted topics
        """
        try:
            # Simple keyword frequency analysis
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
            word_freq = {}
            
            # Common stop words to ignore
            stop_words = {
                'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know',
                'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when',
                'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over',
                'such', 'take', 'than', 'them', 'well', 'were', 'what', 'your'
            }
            
            for word in words:
                if word not in stop_words and len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and return top topics
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in sorted_words[:max_topics]]
            
        except Exception as e:
            logger.error(f"Error in fallback topic extraction: {str(e)}")
            return []


# Global scorer instance
_scorer: Optional[RelevanceScorer] = None


def get_relevance_scorer() -> RelevanceScorer:
    """Get the global relevance scorer instance.
    
    Returns:
        RelevanceScorer instance
    """
    global _scorer
    if _scorer is None:
        _scorer = RelevanceScorer()
    return _scorer


# Standalone function for backward compatibility with tests
async def score_page_relevance(
    page_content: str, 
    query: str, 
    target_keywords: Optional[List[str]] = None,
    page_title: str = "",
    page_url: str = ""
) -> float:
    """Standalone function to score page relevance.
    
    Args:
        page_content: The page content to analyse
        query: The search query
        target_keywords: Optional list of specific keywords to look for
        page_title: Optional page title for additional context
        page_url: Optional page URL for structural analysis
        
    Returns:
        Relevance score (0.0 to 1.0)
    """
    scorer = get_relevance_scorer()
    result = await scorer.score_page_relevance(
        page_content=page_content,
        query=query,
        target_keywords=target_keywords,
        page_title=page_title,
        page_url=page_url
    )
    return result.get("relevance_score", 0.0) 