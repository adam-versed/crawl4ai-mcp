"""
Pytest configuration and fixtures for crawl4ai-mcp tests.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def sample_webpage_content():
    """Sample webpage content for testing."""
    return {
        "success": True,
        "url": "https://example.com/test",
        "title": "Test Page Title",
        "markdown": "# Test Content\n\nThis is sample markdown content for testing purposes.\n\n## Section 1\n\nSome important information here.\n\n## Section 2\n\nMore content with keywords related to the test query.",
        "cleaned_html": "<h1>Test Content</h1><p>This is sample content for testing.</p>",
        "metadata": {
            "description": "A test page for crawl4ai-mcp testing",
            "keywords": ["test", "crawl4ai", "mcp"],
            "author": "Test Author"
        },
        "links": [
            {"url": "https://example.com/page1", "text": "Related Page 1"},
            {"url": "https://example.com/page2", "text": "Important Information"},
            {"url": "https://example.com/page3", "text": "Additional Resources"}
        ]
    }


@pytest.fixture
def sample_crawl_results():
    """Sample crawl results for testing."""
    return {
        "success": True,
        "pages": [
            {
                "url": "https://example.com/test",
                "success": True,
                "title": "Test Page 1",
                "markdown": "# Test Page 1\n\nThis page contains information about testing.",
                "relevance_score": 0.85,
                "key_topics": ["testing", "automation", "tools"]
            },
            {
                "url": "https://example.com/test2",
                "success": True,
                "title": "Test Page 2",
                "markdown": "# Test Page 2\n\nThis page has additional testing information.",
                "relevance_score": 0.72,
                "key_topics": ["testing", "frameworks", "best practices"]
            },
            {
                "url": "https://example.com/test3",
                "success": False,
                "error": "Page not found",
                "relevance_score": 0.0
            }
        ],
        "crawl_summary": {
            "total_pages": 3,
            "successful_pages": 2,
            "failed_pages": 1,
            "average_relevance": 0.785
        }
    }


@pytest.fixture
def mock_mcp_sampler():
    """Mock MCP sampler for testing LLM interactions."""
    sampler = Mock()
    sampler.sample_completion = AsyncMock()
    sampler.analyse_relevance = AsyncMock()
    sampler.extract_topics = AsyncMock()
    
    # Default return values
    sampler.sample_completion.return_value = "This is a sample LLM response for testing purposes."
    sampler.analyse_relevance.return_value = {
        "relevance_score": 0.8,
        "key_topics": ["testing", "automation"],
        "summary": "Highly relevant content for testing purposes."
    }
    sampler.extract_topics.return_value = ["testing", "automation", "tools"]
    
    return sampler


@pytest.fixture
def mock_crawl4ai_session():
    """Mock Crawl4AI session for testing."""
    session = AsyncMock()
    session.arun = AsyncMock()
    session.arun_many = AsyncMock()
    return session


@pytest.fixture
def test_query():
    """Standard test query for testing."""
    return "testing automation tools and frameworks"


@pytest.fixture
def test_urls():
    """Standard test URLs for testing."""
    return [
        "https://example.com/test1",
        "https://example.com/test2",
        "https://example.com/test3"
    ]


@pytest.fixture
def sample_token_counts():
    """Sample token counts for testing token management."""
    return {
        "short_text": 50,
        "medium_text": 500,
        "long_text": 2000,
        "very_long_text": 8000
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "llm": {
            "default_model": "gpt-4",
            "max_tokens": 4000,
            "temperature": 0.1
        },
        "crawling": {
            "default_max_pages": 20,
            "default_crawl_depth": 3,
            "request_delay": 1.0
        },
        "token_limits": {
            "summarisation_input": 8000,
            "summarisation_output": 500,
            "relevance_analysis": 4000
        },
        "caching": {
            "max_size": 1000,
            "ttl_seconds": 3600
        }
    } 