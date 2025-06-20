"""
Error handling and resilience system for crawl4ai-mcp server.

Provides:
- Graceful degradation when LLM sampling fails
- Fallback to traditional crawling when adaptive fails
- Better session cleanup for failed operations
- Retry mechanisms with exponential backoff
- Error recovery strategies
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any, Callable, Awaitable, Union, List
from dataclasses import dataclass
from enum import Enum
from .config import get_config

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorType(Enum):
    """Types of errors that can occur."""
    NETWORK_ERROR = "network_error"
    LLM_SAMPLING_ERROR = "llm_sampling_error"
    PARSING_ERROR = "parsing_error"
    SESSION_ERROR = "session_error"
    CRAWL_ERROR = "crawl_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    operation: str
    timestamp: float
    metadata: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3
    
    def can_retry(self) -> bool:
        """Check if the error can be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment retry count."""
        self.retry_count += 1


class FallbackStrategy:
    """Base class for fallback strategies."""
    
    async def execute(self, context: ErrorContext, original_function: Callable, *args, **kwargs) -> Any:
        """Execute the fallback strategy."""
        raise NotImplementedError


class TraditionalCrawlFallback(FallbackStrategy):
    """Fallback to traditional crawling when adaptive crawling fails."""
    
    async def execute(self, context: ErrorContext, original_function: Callable, *args, **kwargs) -> Any:
        """Execute traditional crawling as fallback."""
        logger.warning(f"Falling back to traditional crawling due to: {context.message}")
        
        # Import here to avoid circular imports
        from .crawl import crawl_website_async
        
        # Extract parameters for traditional crawling
        url = args[0] if args else kwargs.get('url')
        query = args[1] if len(args) > 1 else kwargs.get('query', '')
        max_pages = min(args[2] if len(args) > 2 else kwargs.get('max_budget', 20), 15)  # Reduced budget
        
        if not url:
            raise ValueError("URL is required for fallback crawling")
        
        # Use traditional crawling with conservative settings
        return await crawl_website_async(
            url=url,
            crawl_depth=2,  # Reduced depth
            max_pages=max_pages,
            query=query
        )


class SimpleSummaryFallback(FallbackStrategy):
    """Fallback to simple summarisation when LLM fails."""
    
    async def execute(self, context: ErrorContext, original_function: Callable, *args, **kwargs) -> Any:
        """Execute simple summarisation as fallback."""
        logger.warning(f"Falling back to simple summarisation due to: {context.message}")
        
        # Extract content
        content = args[0] if args else kwargs.get('content', '')
        max_tokens = args[1] if len(args) > 1 else kwargs.get('max_tokens', 500)
        
        if not content:
            return "Unable to generate summary: no content available"
        
        # Simple text truncation as fallback
        sentences = content.split('. ')
        target_length = min(max_tokens * 4, len(content) // 4)  # Rough estimation
        
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) < target_length:
                summary += sentence + ". "
            else:
                break
        
        return summary.strip() or content[:target_length] + "..."


class KeywordBasedRelevanceFallback(FallbackStrategy):
    """Fallback to keyword-based relevance when LLM fails."""
    
    async def execute(self, context: ErrorContext, original_function: Callable, *args, **kwargs) -> Any:
        """Execute keyword-based relevance scoring as fallback."""
        logger.warning(f"Falling back to keyword-based relevance due to: {context.message}")
        
        content = args[0] if args else kwargs.get('content', '')
        query = args[1] if len(args) > 1 else kwargs.get('query', '')
        
        if not content or not query:
            return 0.0
        
        # Simple keyword matching
        content_lower = content.lower()
        query_words = query.lower().split()
        
        matches = sum(1 for word in query_words if word in content_lower)
        score = matches / len(query_words) if query_words else 0.0
        
        return min(score, 0.8)  # Cap at 0.8 to indicate it's not LLM-based


class ErrorHandler:
    """Main error handler with retry logic and fallback strategies."""
    
    def __init__(self):
        self.config = get_config()
        self.fallback_strategies: Dict[str, FallbackStrategy] = {
            'adaptive_crawl': TraditionalCrawlFallback(),
            'summarisation': SimpleSummaryFallback(),
            'relevance_analysis': KeywordBasedRelevanceFallback()
        }
        self.error_history: List[ErrorContext] = []
        self.max_history_size = 1000
        
        logger.info("Initialized error handler with fallback strategies")
    
    def _classify_error(self, error: Exception, operation: str) -> ErrorContext:
        """Classify an error and create context."""
        error_type = ErrorType.UNKNOWN_ERROR
        severity = ErrorSeverity.MEDIUM
        
        error_str = str(error).lower()
        
        # Network-related errors
        if any(keyword in error_str for keyword in ['connection', 'network', 'timeout', 'dns']):
            error_type = ErrorType.NETWORK_ERROR
            severity = ErrorSeverity.MEDIUM
        
        # LLM sampling errors
        elif any(keyword in error_str for keyword in ['llm', 'sampling', 'model', 'openai', 'anthropic']):
            error_type = ErrorType.LLM_SAMPLING_ERROR
            severity = ErrorSeverity.HIGH
        
        # Session errors
        elif any(keyword in error_str for keyword in ['session', 'browser', 'crawler']):
            error_type = ErrorType.SESSION_ERROR
            severity = ErrorSeverity.HIGH
        
        # Rate limiting
        elif any(keyword in error_str for keyword in ['rate limit', '429', 'too many requests']):
            error_type = ErrorType.RATE_LIMIT_ERROR
            severity = ErrorSeverity.MEDIUM
        
        # Timeout errors
        elif 'timeout' in error_str:
            error_type = ErrorType.TIMEOUT_ERROR
            severity = ErrorSeverity.MEDIUM
        
        # Parsing errors
        elif any(keyword in error_str for keyword in ['json', 'parse', 'decode']):
            error_type = ErrorType.PARSING_ERROR
            severity = ErrorSeverity.LOW
        
        return ErrorContext(
            error_type=error_type,
            severity=severity,
            message=str(error),
            operation=operation,
            timestamp=time.time(),
            metadata={
                'error_class': type(error).__name__,
                'original_error': error
            },
            max_retries=self._get_max_retries_for_error(error_type)
        )
    
    def _get_max_retries_for_error(self, error_type: ErrorType) -> int:
        """Get maximum retries for error type."""
        retry_map = {
            ErrorType.NETWORK_ERROR: 3,
            ErrorType.LLM_SAMPLING_ERROR: 2,
            ErrorType.SESSION_ERROR: 2,
            ErrorType.RATE_LIMIT_ERROR: 5,
            ErrorType.TIMEOUT_ERROR: 3,
            ErrorType.PARSING_ERROR: 1,
            ErrorType.CONFIGURATION_ERROR: 0,
            ErrorType.UNKNOWN_ERROR: 2
        }
        return retry_map.get(error_type, 2)
    
    def _calculate_backoff_delay(self, retry_count: int, error_type: ErrorType) -> float:
        """Calculate exponential backoff delay."""
        base_delay = {
            ErrorType.NETWORK_ERROR: 1.0,
            ErrorType.LLM_SAMPLING_ERROR: 2.0,
            ErrorType.SESSION_ERROR: 1.5,
            ErrorType.RATE_LIMIT_ERROR: 5.0,
            ErrorType.TIMEOUT_ERROR: 1.0,
            ErrorType.PARSING_ERROR: 0.5,
            ErrorType.UNKNOWN_ERROR: 1.0
        }.get(error_type, 1.0)
        
        return min(base_delay * (2 ** retry_count), 60.0)  # Max 60 seconds
    
    async def _cleanup_failed_session(self, session_id: Optional[str]):
        """Clean up failed session."""
        if not session_id:
            return
        
        try:
            from .session_manager import session_manager
            await session_manager.handle_session_error(session_id)
            logger.info(f"Cleaned up failed session: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup session {session_id}: {e}")
    
    async def handle_with_retry(
        self,
        operation: str,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """Handle function execution with retry logic and fallbacks."""
        session_id = kwargs.get('session_id')
        error_context = None
        
        for attempt in range(self.config.llm_models.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retrying {operation}, attempt {attempt + 1}")
                
                result = await func(*args, **kwargs)
                
                # Success - log recovery if this was a retry
                if attempt > 0:
                    logger.info(f"Successfully recovered {operation} after {attempt} retries")
                
                return result
                
            except Exception as error:
                # Classify the error
                if error_context is None:
                    error_context = self._classify_error(error, operation)
                else:
                    error_context.increment_retry()
                
                # Add to error history
                self.error_history.append(error_context)
                if len(self.error_history) > self.max_history_size:
                    self.error_history.pop(0)
                
                logger.warning(f"Error in {operation} (attempt {attempt + 1}): {error}")
                
                # Check if we should retry
                if attempt < self.config.llm_models.max_retries and error_context.can_retry():
                    # Calculate backoff delay
                    delay = self._calculate_backoff_delay(attempt, error_context.error_type)
                    
                    # Special handling for rate limits
                    if error_context.error_type == ErrorType.RATE_LIMIT_ERROR:
                        delay = min(delay * 2, 300)  # Up to 5 minutes for rate limits
                    
                    logger.info(f"Waiting {delay:.1f}s before retry...")
                    await asyncio.sleep(delay)
                    continue
                
                # No more retries - try fallback
                logger.error(f"All retry attempts failed for {operation}: {error}")
                
                # Clean up session if needed
                if error_context.error_type == ErrorType.SESSION_ERROR:
                    await self._cleanup_failed_session(session_id)
                
                # Try fallback strategy
                if operation in self.fallback_strategies:
                    try:
                        logger.info(f"Attempting fallback strategy for {operation}")
                        fallback_result = await self.fallback_strategies[operation].execute(
                            error_context, func, *args, **kwargs
                        )
                        logger.info(f"Fallback strategy succeeded for {operation}")
                        return fallback_result
                    except Exception as fallback_error:
                        logger.error(f"Fallback strategy also failed for {operation}: {fallback_error}")
                
                # No fallback available or fallback failed - re-raise original error
                raise error
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        # Count errors by type
        error_counts = {}
        severity_counts = {}
        
        for error_ctx in self.error_history:
            error_type = error_ctx.error_type.value
            severity = error_ctx.severity.value
            
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Recent errors (last hour)
        recent_errors = [
            err for err in self.error_history
            if time.time() - err.timestamp < 3600
        ]
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors_1h": len(recent_errors),
            "error_counts_by_type": error_counts,
            "error_counts_by_severity": severity_counts,
            "fallback_strategies_available": list(self.fallback_strategies.keys())
        }
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        logger.info("Cleared error history")


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


async def handle_with_resilience(
    operation: str,
    func: Callable[..., Awaitable[Any]],
    *args,
    **kwargs
) -> Any:
    """Convenience function to handle operations with resilience."""
    return await get_error_handler().handle_with_retry(operation, func, *args, **kwargs)


def get_error_stats() -> Dict[str, Any]:
    """Get error statistics."""
    return get_error_handler().get_error_statistics()


def clear_error_history():
    """Clear error history."""
    get_error_handler().clear_error_history()


# Decorator for resilient operations
def resilient_operation(operation_name: str):
    """Decorator to make functions resilient with automatic retry and fallback."""
    def decorator(func: Callable[..., Awaitable[Any]]):
        async def wrapper(*args, **kwargs):
            return await handle_with_resilience(operation_name, func, *args, **kwargs)
        return wrapper
    return decorator 