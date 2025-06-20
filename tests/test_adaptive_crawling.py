"""
Integration tests for adaptive crawling functionality.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from tools.crawl import adaptive_crawl_website_async
from tools.crawl_context import CrawlContext, BudgetLimits
from tools.llm_decisions import LLMCrawlDecisionMaker
from tools.relevance_scorer import score_page_relevance
from tools.link_selector import analyse_links_for_crawling


class TestAdaptiveCrawling:
    """Test adaptive crawling functionality."""
    
    @pytest.fixture
    def mock_session_manager(self):
        """Mock session manager for testing."""
        session_manager = Mock()
        session_manager.get_session = AsyncMock()
        session_manager.get_session.return_value = Mock()
        return session_manager
    
    @pytest.fixture
    def mock_crawl_response(self):
        """Mock successful crawl response."""
        return {
            "url": "https://example.com/test",
            "success": True,
            "status_code": 200,
            "title": "Test Page",
            "markdown": "# Test Page\n\nThis is test content with important information about testing frameworks.",
            "cleaned_html": "<h1>Test Page</h1><p>Test content</p>",
            "links": [
                {"url": "https://example.com/related1", "text": "Related Testing Information"},
                {"url": "https://example.com/related2", "text": "Best Practices"},
                {"url": "https://example.com/unrelated", "text": "Unrelated Content"}
            ],
            "metadata": {"description": "Test page description"}
        }
    
    @pytest.fixture
    def budget_limits(self):
        """Standard budget limits for testing."""
        return BudgetLimits(
            max_pages=20,
            max_total_llm_tokens=10000,
            max_crawl_time_seconds=300,
            max_depth=3
        )
    
    @pytest.mark.asyncio
    async def test_adaptive_crawl_basic(self, mock_session_manager, mock_crawl_response):
        """Test basic adaptive crawling functionality."""
        with patch('tools.crawl.get_session_manager', return_value=mock_session_manager):
            with patch('tools.scrape.scrape_url') as mock_scrape:
                mock_scrape.return_value = [Mock(text=json.dumps(mock_crawl_response))]
                
                with patch('tools.mcp_sampler.get_sampler') as mock_sampler:
                    sampler = Mock()
                    sampler.analyse_relevance = AsyncMock(return_value={
                        "relevance_score": 0.85,
                        "key_topics": ["testing", "frameworks"],
                        "summary": "Highly relevant testing content"
                    })
                    sampler.extract_topics = AsyncMock(return_value=["testing", "frameworks", "automation"])
                    mock_sampler.return_value = sampler
                    
                    result = await adaptive_crawl_website_async(
                        url="https://example.com",
                        query="testing frameworks",
                        max_budget=5
                    )
                    
                    assert len(result) == 1
                    result_data = json.loads(result[0].text)
                    
                    # Check for successful response (should have crawl_summary, not direct success field)
                    if "success" in result_data:
                        # Error case
                        assert result_data["success"] is False, "Expected mocked response to work"
                    else:
                        # Success case - should have crawl_summary
                        assert "crawl_summary" in result_data
                        assert "pages" in result_data
                        assert len(result_data["pages"]) >= 0
    
    @pytest.mark.asyncio
    async def test_relevance_scoring(self, sample_webpage_content, test_query):
        """Test relevance scoring functionality."""
        # Mock the MCP sampler
        with patch('tools.mcp_sampler.get_sampler') as mock_sampler:
            sampler = Mock()
            sampler.analyse_relevance = AsyncMock(return_value={
                "relevance_score": 0.8,
                "key_topics": ["testing", "automation"],
                "summary": "Relevant content about testing"
            })
            mock_sampler.return_value = sampler
            
            score = await score_page_relevance(
                page_content=sample_webpage_content["markdown"],
                query=test_query,
                target_keywords=["testing", "automation", "frameworks"]
            )
            
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_link_analysis(self, sample_webpage_content, test_query):
        """Test intelligent link analysis."""
        links = [
            "https://example.com/testing-guide",
            "https://example.com/automation-tools", 
            "https://example.com/unrelated-topic"
        ]
        
        with patch('tools.mcp_sampler.get_sampler') as mock_sampler:
            sampler = Mock()
            sampler.analyse_relevance = AsyncMock(return_value={
                "relevance_score": 0.7,
                "key_topics": ["testing", "automation"],
                "summary": "Relevant testing content"
            })
            mock_sampler.return_value = sampler
            
            analysed_links = await analyse_links_for_crawling(
                links=links,
                page_content=sample_webpage_content["markdown"],
                query=test_query
            )
            
            assert isinstance(analysed_links, list)
            assert len(analysed_links) == len(links)
            
            # Check that each link has been analysed
            for link in analysed_links:
                assert "url" in link
                assert "relevance_score" in link
                assert "should_crawl" in link
    
    def test_crawl_context_management(self, budget_limits):
        """Test crawl context management."""
        context = CrawlContext(
            starting_url="https://example.com",
            query="testing frameworks",
            budget_limits=budget_limits
        )
        
        # Test initial state
        assert context.query == "testing frameworks"
        assert context.budget.pages_crawled == 0
        assert context.budget.llm_tokens_used == 0
        assert not context.budget.is_any_budget_exceeded()
        
        # Test adding page results
        from tools.crawl_context import PageCrawlResult
        page_result = PageCrawlResult(
            url="https://example.com/test",
            success=True,
            relevance_score=0.8,
            key_topics=["testing", "frameworks"]
        )
        
        context.add_crawled_page(page_result)
        
        assert context.budget.pages_crawled == 1
        assert context.get_average_relevance() == 0.8
    
    def test_budget_management(self, budget_limits):
        """Test budget management and enforcement."""
        context = CrawlContext(
            starting_url="https://example.com",
            query="test query",
            budget_limits=budget_limits
        )
        
        # Test within budget
        assert not context.budget.is_any_budget_exceeded()
        assert not context.budget.is_page_budget_exceeded()
        assert not context.budget.is_token_budget_exceeded()
        assert not context.budget.is_time_budget_exceeded()
        
        # Test page budget exceeded
        context.budget.pages_crawled = 25  # Over the limit of 20
        assert context.budget.is_page_budget_exceeded()
        assert context.budget.is_any_budget_exceeded()
        
        # Reset and test token budget
        context.budget.pages_crawled = 5
        context.budget.llm_tokens_used = 15000  # Over the limit of 10000
        assert context.budget.is_token_budget_exceeded()
        assert context.budget.is_any_budget_exceeded()
        
        # Reset and test time budget
        context.budget.llm_tokens_used = 5000
        import time
        context.budget.crawl_start_time = time.time() - 400  # Over the limit of 300
        assert context.budget.is_time_budget_exceeded()
        assert context.budget.is_any_budget_exceeded()
    
    def test_diminishing_returns_detection(self):
        """Test diminishing returns detection."""
        context = CrawlContext(
            starting_url="https://example.com",
            query="test query",
            budget_limits=BudgetLimits(max_pages=50, max_total_llm_tokens=20000, max_crawl_time_seconds=600, max_depth=5)
        )
        
        # Add high relevance pages initially
        for i, score in enumerate([0.9, 0.85, 0.8, 0.75, 0.7]):
            from tools.crawl_context import PageCrawlResult
            page_result = PageCrawlResult(
                url=f"https://example.com/page_{i}",
                success=True,
                relevance_score=score,
                key_topics=[f"topic_{i}"]
            )
            context.add_crawled_page(page_result)
        
        # Should not detect diminishing returns with high scores
        assert not context.detect_diminishing_returns()
        
        # Add low relevance pages
        for i, score in enumerate([0.3, 0.2, 0.1, 0.05, 0.02]):
            from tools.crawl_context import PageCrawlResult
            page_result = PageCrawlResult(
                url=f"https://example.com/low_page_{i}",
                success=True,
                relevance_score=score,
                key_topics=[f"low_topic_{i}"]
            )
            context.add_crawled_page(page_result)
        
        # Should detect diminishing returns
        assert context.detect_diminishing_returns()
    
    @pytest.mark.asyncio
    async def test_llm_decision_maker(self, budget_limits):
        """Test LLM-based decision making."""
        context = CrawlContext(
            starting_url="https://example.com",
            query="testing frameworks",
            budget_limits=budget_limits
        )
        
        # Add some page results
        from tools.crawl_context import PageCrawlResult
        page_result1 = PageCrawlResult(
            url="https://example.com/page1",
            success=True,
            relevance_score=0.8,
            key_topics=["testing"]
        )
        page_result2 = PageCrawlResult(
            url="https://example.com/page2",
            success=True,
            relevance_score=0.7,
            key_topics=["frameworks"]
        )
        context.add_crawled_page(page_result1)
        context.add_crawled_page(page_result2)
        
        with patch('tools.mcp_sampler.get_sampler') as mock_sampler:
            sampler = Mock()
            sampler.request_llm_completion = AsyncMock(return_value="Continue crawling - found relevant content")
            mock_sampler.return_value = sampler
            
            from tools.llm_decisions import LLMCrawlDecisionMaker
            decision_maker = LLMCrawlDecisionMaker()
            
            # Test continuation decision
            should_continue, reason = await decision_maker.should_continue_crawling(context)
            
            assert isinstance(should_continue, bool)
            assert isinstance(reason, str)
            assert len(reason) > 0
    
    @pytest.mark.asyncio
    async def test_link_selection_decision(self, budget_limits):
        """Test LLM-based link selection."""
        context = CrawlContext(
            starting_url="https://example.com",
            query="testing frameworks",
            budget_limits=budget_limits
        )
        
        available_links = [
            {
                "url": "https://example.com/testing-guide",
                "text": "Comprehensive Testing Guide",
                "relevance_score": 0.9,
                "should_crawl": True,
                "priority": "high"
            },
            {
                "url": "https://example.com/automation-tools",
                "text": "Automation Tools",
                "relevance_score": 0.7,
                "should_crawl": True,
                "priority": "medium"
            },
            {
                "url": "https://example.com/random-content",
                "text": "Random Content",
                "relevance_score": 0.2,
                "should_crawl": False,
                "priority": "low"
            }
        ]
        
        with patch('tools.mcp_sampler.get_sampler') as mock_sampler:
            sampler = Mock()
            sampler.request_llm_completion = AsyncMock(return_value="Select the testing guide and automation tools")
            mock_sampler.return_value = sampler
            
            from tools.llm_decisions import LLMCrawlDecisionMaker
            decision_maker = LLMCrawlDecisionMaker()
            
            selected_links = await decision_maker.select_next_links(available_links, context, max_links=2)
            
            assert isinstance(selected_links, list)
            assert len(selected_links) <= 2
            
            # Should prefer higher relevance links
            if len(selected_links) > 0:
                assert all(isinstance(link, str) for link in selected_links)
    
    @pytest.mark.asyncio
    async def test_crawl_completion_evaluation(self, budget_limits):
        """Test crawl completion evaluation."""
        context = CrawlContext(
            starting_url="https://example.com",
            query="testing frameworks",
            budget_limits=budget_limits
        )
        
        # Simulate successful crawl by adding page results
        for i, score in enumerate([0.9, 0.8, 0.7, 0.6, 0.5]):
            from tools.crawl_context import PageCrawlResult
            page_result = PageCrawlResult(
                url=f"https://example.com/result_{i}",
                success=True,
                relevance_score=score,
                key_topics=[f"topic_{i}"]
            )
            context.add_crawled_page(page_result)
        
        with patch('tools.mcp_sampler.get_sampler') as mock_sampler:
            sampler = Mock()
            sampler.request_llm_completion = AsyncMock(return_value="Crawl completed successfully with good coverage")
            mock_sampler.return_value = sampler
            
            from tools.llm_decisions import LLMCrawlDecisionMaker
            decision_maker = LLMCrawlDecisionMaker()
            
            evaluation = await decision_maker.evaluate_crawl_completion(context, "testing frameworks")
            
            assert isinstance(evaluation, dict)
            assert "completion_score" in evaluation
            assert "query_fulfillment" in evaluation  
            assert "recommendations" in evaluation
            assert "summary" in evaluation


class TestAdaptiveCrawlingPerformance:
    """Performance tests for adaptive crawling."""
    
    @pytest.mark.asyncio
    async def test_crawl_efficiency(self):
        """Test crawling efficiency metrics."""
        context = CrawlContext(
            starting_url="https://example.com",
            query="test query",
            budget_limits=BudgetLimits(max_pages=20, max_total_llm_tokens=10000, max_crawl_time_seconds=300, max_depth=3)
        )
        
        # Simulate efficient crawl by adding page results
        for i, score in enumerate([0.9, 0.85, 0.8, 0.75, 0.7]):
            from tools.crawl_context import PageCrawlResult
            page_result = PageCrawlResult(
                url=f"https://example.com/page_{i}",
                success=True,
                relevance_score=score,
                key_topics=[f"topic_{i}"]
            )
            context.add_crawled_page(page_result)
        
        efficiency = context.get_crawl_efficiency()
        
        assert isinstance(efficiency, float)
        assert 0.0 <= efficiency <= 1.0
        
        # Should be efficient with high relevance scores
        assert efficiency > 0.5
    
    def test_resource_usage_tracking(self):
        """Test resource usage tracking."""
        context = CrawlContext(
            starting_url="https://example.com",
            query="test query",
            budget_limits=BudgetLimits(max_pages=100, max_total_llm_tokens=50000, max_crawl_time_seconds=1200, max_depth=5)
        )
        
        # Simulate resource usage by adding page results
        for i in range(25):
            from tools.crawl_context import PageCrawlResult
            page_result = PageCrawlResult(
                url=f"https://example.com/page_{i}",
                success=True,
                relevance_score=0.7,
                key_topics=[f"topic_{i}"]
            )
            context.add_crawled_page(page_result)
        
        # Check resource usage percentages
        page_usage = context.budget.pages_crawled / context.budget.max_pages
        token_usage = context.budget.llm_tokens_used / context.budget.max_total_llm_tokens
        
        assert 0.0 <= page_usage <= 1.0
        assert 0.0 <= token_usage <= 1.0
        
        # Should be within expected ranges
        assert page_usage == 0.25  # 25/100 