"""
Unit tests for token counting and utility functions.
"""

import pytest
from unittest.mock import patch, Mock
from tools.utils import (
    count_tokens,
    truncate_to_token_limit,
    chunk_text_by_tokens,
    estimate_reading_time,
    clean_text,
    extract_domain,
    is_valid_url,
    get_url_depth,
    sanitise_filename
)


class TestTokenFunctions:
    """Test token counting and management functions."""
    
    def test_count_tokens_basic(self):
        """Test basic token counting functionality."""
        # Short text
        text = "Hello world"
        count = count_tokens(text)
        assert isinstance(count, int)
        assert count > 0
        
        # Empty text
        assert count_tokens("") == 0
        
        # Text with special characters
        special_text = "Hello! How are you? 🚀"
        special_count = count_tokens(special_text)
        assert special_count > 0
    
    def test_count_tokens_different_models(self):
        """Test token counting with different models."""
        text = "This is a test sentence for token counting."
        
        # Test with different models
        gpt4_count = count_tokens(text, model="gpt-4")
        gpt35_count = count_tokens(text, model="gpt-3.5-turbo")
        
        assert isinstance(gpt4_count, int)
        assert isinstance(gpt35_count, int)
        assert gpt4_count > 0
        assert gpt35_count > 0
    
    def test_count_tokens_long_text(self):
        """Test token counting with long text."""
        long_text = "This is a test sentence. " * 1000
        count = count_tokens(long_text)
        assert count > 1000  # Should be significantly more than 1000 tokens
    
    def test_truncate_to_token_limit(self):
        """Test text truncation to token limits."""
        text = "This is a test sentence that will be truncated. " * 20
        original_count = count_tokens(text)
        
        # Truncate to 50 tokens
        truncated = truncate_to_token_limit(text, 50)
        truncated_count = count_tokens(truncated)
        
        assert truncated_count <= 50
        assert len(truncated) < len(text)
        assert truncated.endswith("...")
    
    def test_truncate_to_token_limit_short_text(self):
        """Test truncation with text shorter than limit."""
        text = "Short text"
        truncated = truncate_to_token_limit(text, 100)
        
        # Should return original text
        assert truncated == text
    
    def test_chunk_text_by_tokens(self):
        """Test text chunking by token count."""
        text = "This is a test sentence. " * 100
        chunks = chunk_text_by_tokens(text, chunk_size=50, overlap=10)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        
        # Check each chunk is within token limit
        for chunk in chunks:
            assert count_tokens(chunk) <= 50
    
    def test_chunk_text_by_tokens_small_text(self):
        """Test chunking with text smaller than chunk size."""
        text = "Small text"
        chunks = chunk_text_by_tokens(text, chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_by_tokens_overlap(self):
        """Test chunking with overlap."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunk_text_by_tokens(text, chunk_size=10, overlap=5)
        
        assert len(chunks) >= 2
        # Verify overlap exists (this is a basic check)
        if len(chunks) > 1:
            assert len(chunks[0]) > 0
            assert len(chunks[1]) > 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_estimate_reading_time(self):
        """Test reading time estimation."""
        # Short text
        short_text = "This is a short text."
        time = estimate_reading_time(short_text)
        assert time == 1  # Minimum 1 minute
        
        # Long text (approximately 1000 words)
        long_text = "word " * 1000
        time = estimate_reading_time(long_text)
        assert time >= 5  # Should be at least 5 minutes
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "  This\tis\na\r\ntest   text  with  extra  spaces  "
        clean = clean_text(dirty_text)
        
        assert clean == "This is a test text with extra spaces"
        assert not clean.startswith(" ")
        assert not clean.endswith(" ")
    
    def test_clean_text_empty(self):
        """Test cleaning empty or whitespace text."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""
        assert clean_text("\n\t\r") == ""
    
    def test_extract_domain(self):
        """Test domain extraction from URLs."""
        test_cases = [
            ("https://example.com/path", "example.com"),
            ("http://subdomain.example.com", "subdomain.example.com"),
            ("https://example.com:8080/path", "example.com"),
            ("https://example.com", "example.com"),
            ("invalid-url", "")
        ]
        
        for url, expected in test_cases:
            result = extract_domain(url)
            assert result == expected
    
    def test_is_valid_url(self):
        """Test URL validation."""
        valid_urls = [
            "https://example.com",
            "http://example.com",
            "https://subdomain.example.com/path",
            "https://example.com:8080/path?param=value"
        ]
        
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # Not HTTP/HTTPS
            "https://",
            "http://",
            "example.com",  # Missing protocol
            ""
        ]
        
        for url in valid_urls:
            assert is_valid_url(url), f"Should be valid: {url}"
        
        for url in invalid_urls:
            assert not is_valid_url(url), f"Should be invalid: {url}"
    
    def test_get_url_depth(self):
        """Test URL depth calculation."""
        test_cases = [
            ("https://example.com", 0),
            ("https://example.com/", 0),
            ("https://example.com/page", 1),
            ("https://example.com/section/page", 2),
            ("https://example.com/a/b/c/d", 4),
            ("https://example.com/page?param=value", 1),
            ("https://example.com/page#section", 1)
        ]
        
        for url, expected_depth in test_cases:
            depth = get_url_depth(url)
            assert depth == expected_depth, f"URL {url} should have depth {expected_depth}, got {depth}"
    
    def test_sanitise_filename(self):
        """Test filename sanitisation."""
        test_cases = [
            ("normal_filename.txt", "normal_filename.txt"),
            ("file with spaces.txt", "file_with_spaces.txt"),
            ("file/with\\slashes.txt", "file_with_slashes.txt"),
            ("file:with*special?chars.txt", "file_with_special_chars.txt"),
            ("file<with>dangerous|chars.txt", "file_with_dangerous_chars.txt"),
            ("", "unnamed_file"),
            ("...", "unnamed_file"),
            ("file.txt" * 100, "file.txt" * 100)  # Will be truncated
        ]
        
        for input_name, expected in test_cases:
            result = sanitise_filename(input_name)
            if len(expected) > 200:  # Account for truncation
                assert len(result) <= 255
            else:
                assert result == expected
            
            # Ensure no dangerous characters remain
            dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            for char in dangerous_chars:
                assert char not in result


class TestTokenManagementIntegration:
    """Integration tests for token management functions."""
    
    def test_token_workflow(self):
        """Test complete token management workflow."""
        # Create long text
        original_text = "This is a test sentence for token management. " * 200
        
        # Count tokens
        original_count = count_tokens(original_text)
        assert original_count > 100
        
        # Truncate to limit
        truncated = truncate_to_token_limit(original_text, 100)
        truncated_count = count_tokens(truncated)
        assert truncated_count <= 100
        
        # Chunk the original text
        chunks = chunk_text_by_tokens(original_text, chunk_size=50, overlap=10)
        assert len(chunks) > 1
        
        # Verify all chunks are within limits
        for chunk in chunks:
            chunk_count = count_tokens(chunk)
            assert chunk_count <= 50
    
    def test_edge_cases(self):
        """Test edge cases for token functions."""
        # Empty text
        assert count_tokens("") == 0
        assert truncate_to_token_limit("", 100) == ""
        assert chunk_text_by_tokens("", 100) == []
        
        # Single character
        single_char = "a"
        assert count_tokens(single_char) >= 1
        assert truncate_to_token_limit(single_char, 100) == single_char
        assert chunk_text_by_tokens(single_char, 100) == [single_char]
        
        # Very long single word
        long_word = "a" * 1000
        long_count = count_tokens(long_word)
        assert long_count > 0
        
        # Truncate very long word
        truncated_word = truncate_to_token_limit(long_word, 10)
        assert count_tokens(truncated_word) <= 10 