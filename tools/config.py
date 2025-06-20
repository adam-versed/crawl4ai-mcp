"""
Configuration management for crawl4ai-mcp server.

This module provides centralized configuration management with:
- Default token limits per operation
- LLM model selection for different tasks
- Relevance thresholds for crawl decisions
- Budget allocation strategies
- Performance and caching settings
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenLimits:
    """Token limits for different operations."""
    # Input limits (content processing)
    max_input_tokens_summarisation: int = 8000
    max_input_tokens_relevance_analysis: int = 4000
    max_input_tokens_insight_extraction: int = 6000
    max_input_tokens_llm_decisions: int = 3000
    
    # Output limits (LLM responses)
    max_output_tokens_summary: int = 500
    max_output_tokens_relevance: int = 400
    max_output_tokens_insights: int = 600
    max_output_tokens_decisions: int = 300
    
    # Batch processing
    max_tokens_per_batch: int = 15000
    overlap_tokens_chunking: int = 100


@dataclass
class LLMModelConfig:
    """LLM model selection for different tasks."""
    # Primary models for different tasks
    summarisation_model: str = "gpt-4"
    relevance_analysis_model: str = "gpt-4"
    insight_extraction_model: str = "gpt-4"
    crawl_decisions_model: str = "gpt-4"
    topic_extraction_model: str = "gpt-3.5-turbo"  # Lighter task
    
    # Fallback models
    fallback_model: str = "gpt-3.5-turbo"
    
    # Model-specific settings
    temperature: float = 0.1  # Low temperature for consistent results
    max_retries: int = 3
    request_timeout: int = 30


@dataclass
class RelevanceThresholds:
    """Thresholds for relevance scoring and crawl decisions."""
    # Page relevance thresholds
    high_relevance_threshold: float = 0.7
    medium_relevance_threshold: float = 0.4
    low_relevance_threshold: float = 0.2
    
    # Link selection thresholds
    link_crawl_threshold: float = 0.3
    priority_link_threshold: float = 0.6
    
    # Content quality thresholds
    min_content_length: int = 100
    min_word_count: int = 20
    
    # Crawl continuation decisions
    continue_crawl_threshold: float = 0.3
    diminishing_returns_threshold: float = 0.1


@dataclass
class BudgetAllocation:
    """Budget allocation strategies for different operations."""
    # Default budgets
    default_page_budget: int = 20
    default_token_budget: int = 15000
    default_time_budget_seconds: int = 300  # 5 minutes
    
    # Adaptive crawling budgets
    adaptive_page_budget: int = 25
    adaptive_token_budget: int = 20000
    adaptive_time_budget_seconds: int = 600  # 10 minutes
    
    # Research topic budgets (higher limits)
    research_page_budget: int = 30
    research_token_budget: int = 25000
    research_time_budget_seconds: int = 900  # 15 minutes
    
    # Budget allocation percentages
    llm_decision_token_percentage: float = 0.3  # 30% for LLM decisions
    content_processing_token_percentage: float = 0.7  # 70% for content processing
    
    # Emergency limits (hard stops)
    max_absolute_pages: int = 100
    max_absolute_tokens: int = 50000
    max_absolute_time_seconds: int = 1800  # 30 minutes


@dataclass
class CrawlerSettings:
    """Browser and crawler configuration settings."""
    # Browser configuration
    headless: bool = True
    page_timeout_ms: int = 30000
    navigation_timeout_ms: int = 20000
    
    # Content extraction
    word_threshold: int = 10
    only_text: bool = True
    magic_mode: bool = True
    
    # Session management
    max_concurrent_sessions: int = 20
    session_cleanup_threshold: int = 15
    session_cleanup_buffer: int = 10
    
    # Retry settings
    max_retries_per_page: int = 2
    retry_delay_seconds: float = 1.0


@dataclass
class CacheSettings:
    """Caching configuration for performance optimisation."""
    # Enable/disable caching features
    enable_relevance_cache: bool = True
    enable_summary_cache: bool = True
    enable_content_deduplication: bool = True
    
    # Cache limits
    max_relevance_cache_size: int = 1000
    max_summary_cache_size: int = 500
    max_content_hash_cache_size: int = 2000
    
    # Cache expiry (in seconds)
    relevance_cache_ttl: int = 3600  # 1 hour
    summary_cache_ttl: int = 7200   # 2 hours
    content_hash_ttl: int = 86400   # 24 hours
    
    # Similarity thresholds for deduplication
    content_similarity_threshold: float = 0.9
    url_similarity_threshold: float = 0.8


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_llm_decisions: bool = True
    log_relevance_scores: bool = True
    log_performance_metrics: bool = True
    log_budget_usage: bool = True
    
    # Performance monitoring
    enable_performance_tracking: bool = True
    track_token_usage: bool = True
    track_crawl_efficiency: bool = True


@dataclass
class CrawlAIMCPConfig:
    """Main configuration class for crawl4ai-mcp server."""
    # Core configuration sections
    token_limits: TokenLimits = field(default_factory=TokenLimits)
    llm_models: LLMModelConfig = field(default_factory=LLMModelConfig)
    relevance_thresholds: RelevanceThresholds = field(default_factory=RelevanceThresholds)
    budget_allocation: BudgetAllocation = field(default_factory=BudgetAllocation)
    crawler_settings: CrawlerSettings = field(default_factory=CrawlerSettings)
    cache_settings: CacheSettings = field(default_factory=CacheSettings)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Environment-based overrides
    environment: str = "production"  # development, testing, production
    debug_mode: bool = False
    
    def __post_init__(self):
        """Apply environment-specific configurations."""
        if self.environment == "development":
            self._apply_development_config()
        elif self.environment == "testing":
            self._apply_testing_config()
    
    def _apply_development_config(self):
        """Apply development-specific settings."""
        self.debug_mode = True
        self.logging_config.log_level = "DEBUG"
        self.crawler_settings.page_timeout_ms = 60000  # Longer timeouts for debugging
        self.budget_allocation.default_time_budget_seconds = 600  # More time for development
        logger.info("Applied development configuration")
    
    def _apply_testing_config(self):
        """Apply testing-specific settings."""
        self.budget_allocation.default_page_budget = 5  # Smaller budgets for testing
        self.budget_allocation.default_token_budget = 5000
        self.cache_settings.enable_relevance_cache = False  # Disable caching for consistent tests
        self.cache_settings.enable_summary_cache = False
        logger.info("Applied testing configuration")
    
    @classmethod
    def from_environment(cls) -> 'CrawlAIMCPConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Environment detection
        config.environment = os.getenv("CRAWL4AI_MCP_ENV", "production").lower()
        config.debug_mode = os.getenv("CRAWL4AI_MCP_DEBUG", "false").lower() == "true"
        
        # Token limits from environment
        if max_input_sum := os.getenv("CRAWL4AI_MAX_INPUT_TOKENS_SUMMARY"):
            config.token_limits.max_input_tokens_summarisation = int(max_input_sum)
        
        if max_output_sum := os.getenv("CRAWL4AI_MAX_OUTPUT_TOKENS_SUMMARY"):
            config.token_limits.max_output_tokens_summary = int(max_output_sum)
        
        # Model configuration from environment
        if model_primary := os.getenv("CRAWL4AI_PRIMARY_MODEL"):
            config.llm_models.summarisation_model = model_primary
            config.llm_models.relevance_analysis_model = model_primary
            config.llm_models.insight_extraction_model = model_primary
            config.llm_models.crawl_decisions_model = model_primary
        
        if model_fallback := os.getenv("CRAWL4AI_FALLBACK_MODEL"):
            config.llm_models.fallback_model = model_fallback
        
        # Budget limits from environment
        if page_budget := os.getenv("CRAWL4AI_DEFAULT_PAGE_BUDGET"):
            config.budget_allocation.default_page_budget = int(page_budget)
        
        if token_budget := os.getenv("CRAWL4AI_DEFAULT_TOKEN_BUDGET"):
            config.budget_allocation.default_token_budget = int(token_budget)
        
        # Relevance thresholds from environment
        if high_threshold := os.getenv("CRAWL4AI_HIGH_RELEVANCE_THRESHOLD"):
            config.relevance_thresholds.high_relevance_threshold = float(high_threshold)
        
        # Cache settings from environment
        cache_enabled = os.getenv("CRAWL4AI_ENABLE_CACHE", "true").lower() == "true"
        config.cache_settings.enable_relevance_cache = cache_enabled
        config.cache_settings.enable_summary_cache = cache_enabled
        
        # Apply environment-specific post-processing
        config.__post_init__()
        
        logger.info(f"Configuration loaded for environment: {config.environment}")
        return config
    
    def get_token_limit_for_operation(self, operation: str, operation_type: str = "input") -> int:
        """Get token limit for specific operation."""
        operation_map = {
            "summarisation": {
                "input": self.token_limits.max_input_tokens_summarisation,
                "output": self.token_limits.max_output_tokens_summary
            },
            "relevance_analysis": {
                "input": self.token_limits.max_input_tokens_relevance_analysis,
                "output": self.token_limits.max_output_tokens_relevance
            },
            "insight_extraction": {
                "input": self.token_limits.max_input_tokens_insight_extraction,
                "output": self.token_limits.max_output_tokens_insights
            },
            "llm_decisions": {
                "input": self.token_limits.max_input_tokens_llm_decisions,
                "output": self.token_limits.max_output_tokens_decisions
            }
        }
        
        return operation_map.get(operation, {}).get(operation_type, 1000)  # Default fallback
    
    def get_model_for_task(self, task: str) -> str:
        """Get appropriate model for specific task."""
        task_models = {
            "summarisation": self.llm_models.summarisation_model,
            "relevance_analysis": self.llm_models.relevance_analysis_model,
            "insight_extraction": self.llm_models.insight_extraction_model,
            "crawl_decisions": self.llm_models.crawl_decisions_model,
            "topic_extraction": self.llm_models.topic_extraction_model
        }
        
        return task_models.get(task, self.llm_models.fallback_model)
    
    def get_budget_for_operation(self, operation: str) -> Dict:
        """Get budget limits for specific operation."""
        if operation == "adaptive_crawl":
            return {
                "pages": self.budget_allocation.adaptive_page_budget,
                "tokens": self.budget_allocation.adaptive_token_budget,
                "time_seconds": self.budget_allocation.adaptive_time_budget_seconds
            }
        elif operation == "research_topic":
            return {
                "pages": self.budget_allocation.research_page_budget,
                "tokens": self.budget_allocation.research_token_budget,
                "time_seconds": self.budget_allocation.research_time_budget_seconds
            }
        else:  # default
            return {
                "pages": self.budget_allocation.default_page_budget,
                "tokens": self.budget_allocation.default_token_budget,
                "time_seconds": self.budget_allocation.default_time_budget_seconds
            }


# Global configuration instance
_config_instance: Optional[CrawlAIMCPConfig] = None


def get_config() -> CrawlAIMCPConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = CrawlAIMCPConfig.from_environment()
    return _config_instance


def reload_config() -> CrawlAIMCPConfig:
    """Reload configuration from environment."""
    global _config_instance
    _config_instance = CrawlAIMCPConfig.from_environment()
    return _config_instance


# Convenience functions for common configuration access
def get_token_limit(operation: str, operation_type: str = "input") -> int:
    """Get token limit for operation."""
    return get_config().get_token_limit_for_operation(operation, operation_type)


def get_model_for_task(task: str) -> str:
    """Get model for task."""
    return get_config().get_model_for_task(task)


def get_budget_for_operation(operation: str) -> Dict:
    """Get budget for operation."""
    return get_config().get_budget_for_operation(operation)


def get_relevance_threshold(threshold_type: str) -> float:
    """Get relevance threshold."""
    config = get_config()
    thresholds = {
        "high": config.relevance_thresholds.high_relevance_threshold,
        "medium": config.relevance_thresholds.medium_relevance_threshold,
        "low": config.relevance_thresholds.low_relevance_threshold,
        "link_crawl": config.relevance_thresholds.link_crawl_threshold,
        "priority_link": config.relevance_thresholds.priority_link_threshold,
        "continue_crawl": config.relevance_thresholds.continue_crawl_threshold,
        "diminishing_returns": config.relevance_thresholds.diminishing_returns_threshold
    }
    return thresholds.get(threshold_type, 0.5)  # Default fallback 