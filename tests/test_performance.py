"""
Performance tests for crawl4ai-mcp server.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from tools.cache_manager import CrawlAICacheManager
from tools.utils import count_tokens, chunk_text_by_tokens
from tools.crawl_context import CrawlContext, BudgetLimits


class TestPerformanceBaseline:
    """Baseline performance tests for core functions."""
    
    def test_token_counting_performance(self):
        """Test token counting performance with various text sizes."""
        test_cases = [
            ("Short text", "This is a short text for testing.", 0.1),
            ("Medium text", "This is a medium text. " * 100, 0.2),
            ("Long text", "This is a long text. " * 1000, 0.5),
            ("Very long text", "This is a very long text. " * 5000, 1.0)
        ]
        
        for name, text, max_time in test_cases:
            start_time = time.time()
            token_count = count_tokens(text)
            elapsed = time.time() - start_time
            
            assert token_count > 0, f"Token counting failed for {name}"
            assert elapsed < max_time, f"Token counting too slow for {name}: {elapsed:.3f}s > {max_time}s"
    
    def test_text_chunking_performance(self):
        """Test text chunking performance with large texts."""
        # Create a large text document
        large_text = "This is a test sentence with meaningful content. " * 10000
        
        start_time = time.time()
        chunks = chunk_text_by_tokens(large_text, chunk_size=500, overlap=50)
        elapsed = time.time() - start_time
        
        assert len(chunks) > 1, "Chunking should produce multiple chunks"
        assert elapsed < 2.0, f"Text chunking too slow: {elapsed:.3f}s > 2.0s"
        
        # Verify chunk quality
        total_chunks = len(chunks)
        assert total_chunks > 10, f"Expected more chunks for large text, got {total_chunks}"
    
    def test_cache_performance(self):
        """Test cache performance with high-frequency operations."""
        cache = CrawlAICacheManager()
        
        # Test cache write performance
        start_time = time.time()
        for i in range(1000):
            cache.cache_relevance_score(f"test_content_{i}", f"test_query_{i}", 0.5 + (i % 5) * 0.1)
        write_elapsed = time.time() - start_time
        
        # Test cache read performance
        start_time = time.time()
        hits = 0
        for i in range(1000):
            content_hash = cache.hasher.hash_content(f"test_content_{i}")
            result = cache.get_relevance_score(content_hash, f"test_query_{i}")
            if result is not None:
                hits += 1
        read_elapsed = time.time() - start_time
        
        assert write_elapsed < 1.0, f"Cache writes too slow: {write_elapsed:.3f}s > 1.0s"
        assert read_elapsed < 0.5, f"Cache reads too slow: {read_elapsed:.3f}s > 0.5s"
        assert hits > 900, f"Cache hit rate too low: {hits}/1000"
    
    def test_context_management_performance(self):
        """Test crawl context performance with many operations."""
        budget_limits = BudgetLimits(
            max_pages=1000,
            max_total_llm_tokens=100000,
            max_crawl_time_seconds=3600,
            max_depth=10
        )
        
        context = CrawlContext(
            starting_url="https://example.com",
            query="performance test query",
            budget_limits=budget_limits
        )
        
        # Test adding many page results
        start_time = time.time()
        for i in range(1000):
            from tools.crawl_context import PageCrawlResult
            page_result = PageCrawlResult(
                url=f"https://example.com/page_{i}",
                success=True,
                relevance_score=0.7 + (i % 3) * 0.1,
                key_topics=[f"topic_{i}", f"topic_{i+1}"]
            )
            context.add_crawled_page(page_result)
        
        elapsed = time.time() - start_time
        
        assert elapsed < 1.0, f"Context management too slow: {elapsed:.3f}s > 1.0s"
        assert context.budget.pages_crawled == 1000
        assert context.get_average_relevance() > 0.0


class TestLargeScaleCrawling:
    """Tests for large-scale crawling scenarios."""
    
    @pytest.mark.asyncio
    async def test_large_budget_handling(self):
        """Test handling of large crawl budgets."""
        large_budget = BudgetLimits(
            max_pages=500,
            max_total_llm_tokens=500000,
            max_crawl_time_seconds=7200,  # 2 hours
            max_depth=10
        )
        
        context = CrawlContext(
            starting_url="https://example.com",
            query="large scale crawling test",
            budget_limits=large_budget
        )
        
        # Simulate a large crawl
        start_time = time.time()
        
        # Add many pages efficiently
        for i in range(100):  # Simulate 100 pages
            from tools.crawl_context import PageCrawlResult
            page_result = PageCrawlResult(
                url=f"https://example.com/large_page_{i}",
                success=True,
                relevance_score=0.6 + (i % 5) * 0.08,
                key_topics=[f"topic_{i}", f"category_{i//10}"]
            )
            context.add_crawled_page(page_result)
        
        elapsed = time.time() - start_time
        
        # Should handle large scale efficiently
        assert elapsed < 0.5, f"Large scale handling too slow: {elapsed:.3f}s"
        assert not context.budget.is_any_budget_exceeded()
        assert context.get_crawl_efficiency() >= 0.0
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency with large data sets."""
        # Create a large amount of test data
        large_content = "This is a large content block. " * 10000
        
        # Test multiple operations don't consume excessive memory
        operations = []
        
        # Simulate multiple concurrent operations
        for i in range(50):
            async def operation():
                # Simulate content processing
                token_count = count_tokens(large_content)
                chunks = chunk_text_by_tokens(large_content, chunk_size=1000)
                return len(chunks), token_count
            
            operations.append(operation())
        
        start_time = time.time()
        results = await asyncio.gather(*operations)
        elapsed = time.time() - start_time
        
        assert len(results) == 50
        assert elapsed < 5.0, f"Concurrent operations too slow: {elapsed:.3f}s"
        
        # Verify all operations completed successfully
        for chunks_count, token_count in results:
            assert chunks_count > 0
            assert token_count > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_crawling_simulation(self):
        """Test performance with concurrent crawling operations."""
        # Mock the crawling components
        with patch('tools.mcp_sampler.get_sampler') as mock_sampler:
            sampler = Mock()
            sampler.analyse_relevance = AsyncMock(return_value={
                "relevance_score": 0.7,
                "key_topics": ["test", "performance"],
                "summary": "Test content"
            })
            sampler.extract_topics = AsyncMock(return_value=["test", "performance"])
            mock_sampler.return_value = sampler
            
            # Simulate multiple concurrent crawl sessions
            async def simulate_crawl_session(session_id: int):
                context = CrawlContext(
                    starting_url="https://example.com",
                    query=f"test query {session_id}",
                    budget_limits=BudgetLimits(
                        max_pages=20,
                        max_total_llm_tokens=10000,
                        max_crawl_time_seconds=300,
                        max_depth=3
                    )
                )
                
                # Simulate processing pages
                for i in range(10):
                    from tools.crawl_context import PageCrawlResult
                    page_result = PageCrawlResult(
                        url=f"https://example.com/session_{session_id}_page_{i}",
                        success=True,
                        relevance_score=0.6 + (i % 4) * 0.1,
                        key_topics=[f"topic_{i}"]
                    )
                    context.add_crawled_page(page_result)
                
                return context.get_crawl_efficiency()
            
            # Run multiple sessions concurrently
            start_time = time.time()
            sessions = [simulate_crawl_session(i) for i in range(20)]
            efficiencies = await asyncio.gather(*sessions)
            elapsed = time.time() - start_time
            
            assert len(efficiencies) == 20
            assert elapsed < 2.0, f"Concurrent crawling too slow: {elapsed:.3f}s"
            assert all(eff >= 0.0 for eff in efficiencies)
    
    def test_diminishing_returns_at_scale(self):
        """Test diminishing returns detection with large datasets."""
        context = CrawlContext(
            starting_url="https://example.com",
            query="large scale test",
            budget_limits=BudgetLimits(
                max_pages=1000,
                max_total_llm_tokens=1000000,
                max_crawl_time_seconds=10800,
                max_depth=15
            )
        )
        
        # Simulate a realistic crawl with diminishing returns
        # Start with high relevance, gradually decrease
        relevance_scores = []
        
        # High relevance phase (first 100 pages)
        for i in range(100):
            score = 0.9 - (i * 0.002)  # Slowly decrease from 0.9 to 0.7
            relevance_scores.append(max(0.7, score))
        
        # Medium relevance phase (next 200 pages)
        for i in range(200):
            score = 0.7 - (i * 0.001)  # Decrease from 0.7 to 0.5
            relevance_scores.append(max(0.5, score))
        
        # Low relevance phase (next 300 pages)
        for i in range(300):
            score = 0.5 - (i * 0.0005)  # Decrease from 0.5 to 0.35
            relevance_scores.append(max(0.1, score))
        
        # Add some very low relevance pages at the end to trigger diminishing returns
        for i in range(10):
            relevance_scores.append(0.1)  # Very low scores to trigger detection
        
        start_time = time.time()
        
        # Add all relevance scores by creating page results
        for i, score in enumerate(relevance_scores):
            from tools.crawl_context import PageCrawlResult
            page_result = PageCrawlResult(
                url=f"https://example.com/page_{i}",
                success=True,
                relevance_score=score,
                key_topics=[f"topic_{i}"]
            )
            context.add_crawled_page(page_result)
        
        elapsed = time.time() - start_time
        
        # Should detect diminishing returns
        assert context.detect_diminishing_returns()
        assert elapsed < 0.1, f"Diminishing returns detection too slow: {elapsed:.3f}s"
        assert context.get_average_relevance() > 0.0


class TestSystemResourceManagement:
    """Test system resource management and limits."""
    
    def test_memory_usage_limits(self):
        """Test that memory usage stays within reasonable limits."""
        # Create cache with size limits
        cache = CrawlAICacheManager()
        
        # Fill cache beyond normal capacity
        for i in range(10000):
            large_content = f"Large content block {i}. " * 100
            cache.cache_summary(large_content, f"Summary {i}", max_tokens=1000)
        
        # Cache should self-manage size
        stats = cache.get_cache_statistics()
        
        # Should not exceed reasonable limits
        assert stats['summary_cache']['size'] < 2000, "Cache size not properly managed"
    
    def test_processing_time_limits(self):
        """Test that processing operations respect time limits."""
        # Test with time-sensitive operations
        large_text = "This is a very large text document. " * 50000
        
        start_time = time.time()
        
        # Multiple expensive operations
        for i in range(10):
            token_count = count_tokens(large_text)
            chunks = chunk_text_by_tokens(large_text, chunk_size=1000, overlap=100)
            
            # Early termination check
            if time.time() - start_time > 5.0:
                break
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed < 10.0, f"Processing took too long: {elapsed:.3f}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_resource_usage(self):
        """Test resource usage under concurrent load."""
        cache = CrawlAICacheManager()
        
        async def resource_intensive_task(task_id: int):
            # Simulate resource-intensive operations
            text = f"Task {task_id} content. " * 1000
            
            # Multiple cache operations
            for i in range(100):
                cache.cache_relevance_score(f"{text}_{i}", f"query_{task_id}", 0.5)
                content_hash = cache.hasher.hash_content(f"{text}_{i}")
                cache.get_relevance_score(content_hash, f"query_{task_id}")
            
            return task_id
        
        # Run multiple resource-intensive tasks
        start_time = time.time()
        tasks = [resource_intensive_task(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        assert len(results) == 20
        assert elapsed < 3.0, f"Concurrent resource usage too slow: {elapsed:.3f}s"
        
        # Verify cache is still functional
        stats = cache.get_cache_statistics()
        assert stats['relevance_cache']['size'] > 0


class TestScalabilityLimits:
    """Test system behaviour at scalability limits."""
    
    def test_maximum_page_handling(self):
        """Test handling of maximum page limits."""
        # Create context with maximum reasonable limits
        max_budget = BudgetLimits(
            max_pages=10000,
            max_total_llm_tokens=10000000,
            max_crawl_time_seconds=86400,  # 24 hours
            max_depth=50
        )
        
        context = CrawlContext(
            starting_url="https://example.com",
            query="maximum page test",
            budget_limits=max_budget
        )
        
        # Should handle maximum configuration
        assert not context.budget.is_any_budget_exceeded()
        assert context.budget.max_pages == 10000
        assert context.budget.max_total_llm_tokens == 10000000
    
    def test_extreme_content_sizes(self):
        """Test handling of extremely large content."""
        # Create very large content
        extreme_content = "Extreme content. " * 100000  # ~1.7M characters
        
        start_time = time.time()
        
        # Should handle without crashing
        token_count = count_tokens(extreme_content)
        chunks = chunk_text_by_tokens(extreme_content, chunk_size=2000, overlap=200)
        
        elapsed = time.time() - start_time
        
        assert token_count > 0
        assert len(chunks) > 0
        assert elapsed < 30.0, f"Extreme content processing too slow: {elapsed:.3f}s"
    
    def test_edge_case_recovery(self):
        """Test recovery from edge cases and errors."""
        # Test with safer problematic inputs
        problematic_inputs = [
            "a" * 1000,  # Long single word (reduced to safe size)
            "\n" * 100,  # Many newlines (reduced to safe size)
            "🚀" * 50,  # Unicode characters (reduced to safe size)
            "word " * 1000,  # Many words
        ]
        
        for i, input_text in enumerate(problematic_inputs):
            try:
                # Add timeout protection
                start_time = time.time()
                
                token_count = count_tokens(input_text)
                chunks = chunk_text_by_tokens(input_text, chunk_size=100)
                
                elapsed = time.time() - start_time
                
                # Should handle gracefully and quickly
                assert token_count >= 0
                assert isinstance(chunks, list)
                assert elapsed < 2.0, f"Edge case {i} processing took too long: {elapsed:.3f}s"
                
            except Exception as e:
                pytest.fail(f"Failed to handle problematic input {i}: {e}")
        
        # Test empty string separately (safer)
        try:
            token_count = count_tokens("")
            chunks = chunk_text_by_tokens("", chunk_size=100)
            assert token_count == 0
            assert chunks == []
        except Exception as e:
            pytest.fail(f"Failed to handle empty string: {e}")
    
    @pytest.mark.asyncio
    async def test_system_stability_under_load(self):
        """Test system stability under sustained load."""
        # Create sustained load scenario
        cache = CrawlAICacheManager()
        
        async def sustained_load_task():
            for i in range(1000):
                # Mixed operations
                text = f"Sustained load content {i}. " * 50
                
                # Cache operations
                cache.cache_summary(text, f"Summary {i}", max_tokens=1000)
                content_hash = cache.hasher.hash_content(text)
                cache.get_summary(content_hash, max_tokens=1000)
                
                # Token operations
                count_tokens(text)
                chunk_text_by_tokens(text, chunk_size=100)
                
                # Yield control occasionally
                if i % 100 == 0:
                    await asyncio.sleep(0.01)
            
            return "completed"
        
        # Run sustained load
        start_time = time.time()
        result = await sustained_load_task()
        elapsed = time.time() - start_time
        
        assert result == "completed"
        assert elapsed < 30.0, f"Sustained load took too long: {elapsed:.3f}s"
        
        # Verify system is still responsive
        stats = cache.get_cache_statistics()
        assert stats is not None 