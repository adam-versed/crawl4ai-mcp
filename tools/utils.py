import re
import tiktoken
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_and_normalize_url(url: str) -> str | None:
    """Validate and normalize a URL.

    Args:
        url: The URL string to validate.

    Returns:
        The normalized URL with https scheme if valid, otherwise None.
    """
    # Simple validation for domains/subdomains with http(s)
    # Allows for optional paths
    url_pattern = re.compile(
        r"^(?:https?://)?(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    if not url_pattern.match(url):
        return None

    # Add https:// if missing
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"https://{url}"

    return url


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string for a given model.
    
    Args:
        text: The text to count tokens for
        model: The model to use for tokenisation (default: gpt-4)
        
    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base encoding for unknown models
        logger.warning(f"Unknown model {model}, using cl100k_base encoding")
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncate text to fit within a token limit.
    
    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens allowed
        model: The model to use for tokenisation (default: gpt-4)
        
    Returns:
        Truncated text that fits within the token limit
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"Unknown model {model}, using cl100k_base encoding")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    # Reserve tokens for "..." indicator
    ellipsis_tokens = encoding.encode("...")
    available_tokens = max_tokens - len(ellipsis_tokens)
    
    if available_tokens <= 0:
        return "..."
    
    truncated_tokens = tokens[:available_tokens]
    truncated_text = encoding.decode(truncated_tokens)
    return truncated_text + "..."


def chunk_text_by_tokens(text: str, chunk_size: int, overlap: int = 100, model: str = "gpt-4") -> List[str]:
    """Split text into chunks based on token count with optional overlap.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks (default: 100)
        model: The model to use for tokenisation (default: gpt-4)
        
    Returns:
        List of text chunks
    """
    # Handle empty text
    if not text.strip():
        return []
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"Unknown model {model}, using cl100k_base encoding")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    
    # If text is shorter than chunk_size, return as single chunk
    if len(tokens) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        if end >= len(tokens):
            break
            
        # Move start position with overlap
        # Ensure we make progress even with large overlap
        next_start = max(start + 1, end - overlap)
        start = next_start
    
    return chunks


def estimate_tokens_from_chars(text: str) -> int:
    """Rough estimation of token count based on character count.
    
    This is a fast approximation - use count_tokens() for accuracy.
    Rule of thumb: ~4 characters per token for English text.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated number of tokens
    """
    return max(1, len(text) // 4)


def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Estimate reading time for text in minutes.
    
    Args:
        text: The text to estimate reading time for
        words_per_minute: Average reading speed (default: 200 wpm)
        
    Returns:
        Estimated reading time in minutes (minimum 1)
    """
    word_count = len(text.split())
    minutes = max(1, word_count // words_per_minute)
    return minutes


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing spacing.
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple whitespace characters with single spaces
    cleaned = re.sub(r'\s+', ' ', text.strip())
    return cleaned


def extract_domain(url: str) -> str:
    """Extract domain from URL.
    
    Args:
        url: The URL to extract domain from
        
    Returns:
        Domain name or empty string if invalid
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.split(':')[0]  # Remove port if present
    except Exception:
        return ""


def is_valid_url(url: str) -> bool:
    """Check if URL is valid HTTP/HTTPS URL.
    
    Args:
        url: The URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
    except Exception:
        return False


def get_url_depth(url: str) -> int:
    """Get the depth of a URL (number of path segments).
    
    Args:
        url: The URL to analyze
        
    Returns:
        Number of path segments (depth)
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        if not path:
            return 0
        return len(path.split('/'))
    except Exception:
        return 0


def sanitise_filename(filename: str) -> str:
    """Sanitise a filename by replacing problematic characters.
    
    Args:
        filename: The filename to sanitise
        
    Returns:
        Sanitised filename
    """
    # Replace problematic characters with underscores
    sanitised = re.sub(r'[<>:"/\\|?*\s]', '_', filename)
    # Remove any trailing dots or spaces that Windows doesn't like
    sanitised = sanitised.rstrip('. ')
    # Ensure it's not empty
    if not sanitised:
        sanitised = "unnamed_file"
    
    # Truncate to filesystem limit (255 characters is common)
    if len(sanitised) > 255:
        # Try to preserve extension if possible
        parts = sanitised.rsplit('.', 1)
        if len(parts) == 2 and len(parts[1]) < 20:  # Reasonable extension length
            ext = '.' + parts[1]
            base = parts[0][:255 - len(ext)]
            sanitised = base + ext
        else:
            sanitised = sanitised[:255]
    
    return sanitised
