"""
Caching manager for crawl4ai-mcp server.

Implements intelligent caching for:
- Relevance scores for similar content
- Summarisations with content hashes
- Smart deduplication of content
- TTL-based cache expiry
"""

import hashlib
import time
import logging
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from collections import OrderedDict
from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    content_hash: Optional[str] = None
    
    def is_expired(self, ttl: int) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > ttl
    
    def touch(self):
        """Update access information."""
        self.access_count += 1


class LRUCache:
    """LRU Cache with TTL support."""
    
    def __init__(self, max_size: int, ttl: int):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            self._misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if entry.is_expired(self.ttl):
            del self.cache[key]
            self._misses += 1
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        entry.touch()
        self._hits += 1
        return entry.value
    
    def put(self, key: str, value: Any, content_hash: Optional[str] = None):
        """Put value in cache."""
        # Remove if already exists
        if key in self.cache:
            del self.cache[key]
        
        # Remove oldest if at capacity
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        # Add new entry
        entry = CacheEntry(
            value=value,
            timestamp=time.time(),
            content_hash=content_hash
        )
        self.cache[key] = entry
    
    def remove(self, key: str) -> bool:
        """Remove key from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0
    
    def cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry.timestamp > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl": self.ttl
        }


class ContentHasher:
    """Utilities for content hashing and similarity detection."""
    
    @staticmethod
    def hash_content(content: str) -> str:
        """Generate hash for content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def hash_url(url: str) -> str:
        """Generate hash for URL."""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    @staticmethod
    def content_similarity(content1: str, content2: str) -> float:
        """Calculate content similarity (simple implementation)."""
        if not content1 or not content2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def url_similarity(url1: str, url2: str) -> float:
        """Calculate URL similarity."""
        if not url1 or not url2:
            return 0.0
        
        # Remove protocol and www
        def normalise_url(url: str) -> str:
            url = url.lower()
            url = url.replace('https://', '').replace('http://', '')
            url = url.replace('www.', '')
            return url.rstrip('/')
        
        norm_url1 = normalise_url(url1)
        norm_url2 = normalise_url(url2)
        
        if norm_url1 == norm_url2:
            return 1.0
        
        # Check if one is a path of the other
        if norm_url1 in norm_url2 or norm_url2 in norm_url1:
            return 0.8
        
        # Check domain similarity
        domain1 = norm_url1.split('/')[0]
        domain2 = norm_url2.split('/')[0]
        
        if domain1 == domain2:
            return 0.6
        
        return 0.0


class CacheManager:
    """Main cache manager for all caching operations."""
    
    def __init__(self):
        config = get_config()
        cache_config = config.cache_settings
        
        # Initialize caches
        self.relevance_cache = LRUCache(
            max_size=cache_config.max_relevance_cache_size,
            ttl=cache_config.relevance_cache_ttl
        ) if cache_config.enable_relevance_cache else None
        
        self.summary_cache = LRUCache(
            max_size=cache_config.max_summary_cache_size,
            ttl=cache_config.summary_cache_ttl
        ) if cache_config.enable_summary_cache else None
        
        self.content_hash_cache = LRUCache(
            max_size=cache_config.max_content_hash_cache_size,
            ttl=cache_config.content_hash_ttl
        ) if cache_config.enable_content_deduplication else None
        
        self.config = cache_config
        self.hasher = ContentHasher()
        
        logger.info(f"Initialized cache manager with relevance={self.relevance_cache is not None}, "
                   f"summary={self.summary_cache is not None}, "
                   f"deduplication={self.content_hash_cache is not None}")
    
    # Relevance caching
    def get_relevance_score(self, content_hash: str, query: str) -> Optional[float]:
        """Get cached relevance score."""
        if not self.relevance_cache:
            return None
        
        cache_key = f"{content_hash}:{hashlib.md5(query.encode()).hexdigest()}"
        return self.relevance_cache.get(cache_key)
    
    def cache_relevance_score(self, content: str, query: str, score: float):
        """Cache relevance score."""
        if not self.relevance_cache:
            return
        
        content_hash = self.hasher.hash_content(content)
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = f"{content_hash}:{query_hash}"
        
        self.relevance_cache.put(cache_key, score, content_hash)
        logger.debug(f"Cached relevance score {score:.3f} for content hash {content_hash[:8]}")
    
    # Summary caching
    def get_summary(self, content_hash: str, max_tokens: int, focus: Optional[str] = None) -> Optional[str]:
        """Get cached summary."""
        if not self.summary_cache:
            return None
        
        focus_hash = hashlib.md5(focus.encode()).hexdigest() if focus else "none"
        cache_key = f"{content_hash}:{max_tokens}:{focus_hash}"
        return self.summary_cache.get(cache_key)
    
    def cache_summary(self, content: str, summary: str, max_tokens: int, focus: Optional[str] = None):
        """Cache summary."""
        if not self.summary_cache:
            return
        
        content_hash = self.hasher.hash_content(content)
        focus_hash = hashlib.md5(focus.encode()).hexdigest() if focus else "none"
        cache_key = f"{content_hash}:{max_tokens}:{focus_hash}"
        
        self.summary_cache.put(cache_key, summary, content_hash)
        logger.debug(f"Cached summary for content hash {content_hash[:8]}")
    
    # Content deduplication
    def find_similar_content(self, content: str, url: str) -> Optional[Tuple[str, float]]:
        """Find similar content in cache."""
        if not self.content_hash_cache:
            return None
        
        current_hash = self.hasher.hash_content(content)
        
        # Check for exact match first
        if self.content_hash_cache.get(current_hash):
            return current_hash, 1.0
        
        # Check for similar content
        best_match = None
        best_similarity = 0.0
        
        for cached_hash, entry in self.content_hash_cache.cache.items():
            if entry.is_expired(self.content_hash_cache.ttl):
                continue
            
            # Compare content similarity
            if hasattr(entry.value, 'content'):
                similarity = self.hasher.content_similarity(content, entry.value['content'])
                if similarity > self.config.content_similarity_threshold and similarity > best_similarity:
                    best_match = cached_hash
                    best_similarity = similarity
            
            # Compare URL similarity
            if hasattr(entry.value, 'url'):
                url_sim = self.hasher.url_similarity(url, entry.value['url'])
                if url_sim > self.config.url_similarity_threshold and url_sim > best_similarity:
                    best_match = cached_hash
                    best_similarity = url_sim
        
        return (best_match, best_similarity) if best_match else None
    
    def cache_content(self, content: str, url: str, metadata: Dict[str, Any]):
        """Cache content for deduplication."""
        if not self.content_hash_cache:
            return
        
        content_hash = self.hasher.hash_content(content)
        cache_data = {
            'content': content,
            'url': url,
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        self.content_hash_cache.put(content_hash, cache_data, content_hash)
        logger.debug(f"Cached content for deduplication: {url}")
    
    def is_duplicate_content(self, content: str, url: str) -> bool:
        """Check if content is a duplicate."""
        similar = self.find_similar_content(content, url)
        if similar:
            _, similarity = similar
            return similarity > max(
                self.config.content_similarity_threshold,
                self.config.url_similarity_threshold
            )
        return False
    
    # Cache maintenance
    def cleanup_expired_entries(self):
        """Clean up expired entries from all caches."""
        total_cleaned = 0
        
        if self.relevance_cache:
            cleaned = self.relevance_cache.cleanup_expired()
            total_cleaned += cleaned
            logger.debug(f"Cleaned {cleaned} expired relevance cache entries")
        
        if self.summary_cache:
            cleaned = self.summary_cache.cleanup_expired()
            total_cleaned += cleaned
            logger.debug(f"Cleaned {cleaned} expired summary cache entries")
        
        if self.content_hash_cache:
            cleaned = self.content_hash_cache.cleanup_expired()
            total_cleaned += cleaned
            logger.debug(f"Cleaned {cleaned} expired content hash entries")
        
        if total_cleaned > 0:
            logger.info(f"Cleaned {total_cleaned} expired cache entries")
        
        return total_cleaned
    
    def clear_all_caches(self):
        """Clear all caches."""
        if self.relevance_cache:
            self.relevance_cache.clear()
        if self.summary_cache:
            self.summary_cache.clear()
        if self.content_hash_cache:
            self.content_hash_cache.clear()
        
        logger.info("Cleared all caches")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "enabled_caches": {
                "relevance": self.relevance_cache is not None,
                "summary": self.summary_cache is not None,
                "content_deduplication": self.content_hash_cache is not None
            }
        }
        
        if self.relevance_cache:
            stats["relevance_cache"] = self.relevance_cache.get_stats()
        
        if self.summary_cache:
            stats["summary_cache"] = self.summary_cache.get_stats()
        
        if self.content_hash_cache:
            stats["content_hash_cache"] = self.content_hash_cache.get_stats()
        
        return stats


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def clear_caches():
    """Clear all caches."""
    global _cache_manager
    if _cache_manager:
        _cache_manager.clear_all_caches()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return get_cache_manager().get_cache_statistics()


def cleanup_expired_caches():
    """Clean up expired cache entries."""
    return get_cache_manager().cleanup_expired_entries()


# Backward compatibility alias for tests
CrawlAICacheManager = CacheManager 