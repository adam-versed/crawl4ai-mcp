"""
System Diagnostics Module for crawl4ai-mcp.

Provides comprehensive system diagnostics including cache performance,
error statistics, and configuration status.
"""

import json
import logging
import time
from typing import Dict, List, Any
import mcp.types as types
from .cache_manager import get_cache_stats, cleanup_expired_caches
from .error_handler import get_error_stats
from .config import get_config

logger = logging.getLogger(__name__)


async def system_diagnostics_async(
    include_cache_stats: bool = True,
    include_error_stats: bool = True,
    include_config_info: bool = False,
) -> List[types.TextContent]:
    """
    Get comprehensive system diagnostics including cache performance, error statistics, and configuration status.
    
    This function provides detailed information about the crawl4ai-mcp server's operational status,
    performance metrics, and system health. Useful for monitoring and troubleshooting.

    Args:
        include_cache_stats: Include cache performance statistics (default: True).
        include_error_stats: Include error handling statistics (default: True).
        include_config_info: Include configuration information (default: False).

    Returns:
        List containing TextContent with comprehensive system diagnostics as JSON.
    """
    try:
        diagnostics = {
            "success": True,
            "timestamp": time.time(),
            "server_status": "operational",
            "components": {}
        }
        
        # Cache statistics
        if include_cache_stats:
            try:
                cache_stats = get_cache_stats()
                diagnostics["components"]["cache_system"] = {
                    "status": "operational",
                    "statistics": cache_stats,
                    "performance_summary": {
                        "total_hit_rate": None,
                        "cache_efficiency": None,
                        "total_entries": 0
                    }
                }
                
                # Calculate overall performance metrics
                total_hits = 0
                total_requests = 0
                total_entries = 0
                
                for cache_name, cache_data in cache_stats.items():
                    if isinstance(cache_data, dict) and "hits" in cache_data:
                        total_hits += cache_data["hits"]
                        total_requests += cache_data["hits"] + cache_data["misses"]
                        total_entries += cache_data["size"]
                
                if total_requests > 0:
                    diagnostics["components"]["cache_system"]["performance_summary"]["total_hit_rate"] = total_hits / total_requests
                    diagnostics["components"]["cache_system"]["performance_summary"]["cache_efficiency"] = "good" if total_hits / total_requests > 0.5 else "needs_improvement"
                
                diagnostics["components"]["cache_system"]["performance_summary"]["total_entries"] = total_entries
                
                # Cleanup expired entries
                cleaned_entries = cleanup_expired_caches()
                if cleaned_entries > 0:
                    diagnostics["components"]["cache_system"]["maintenance"] = f"Cleaned {cleaned_entries} expired entries"
                
            except Exception as e:
                diagnostics["components"]["cache_system"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Error statistics
        if include_error_stats:
            try:
                error_stats = get_error_stats()
                diagnostics["components"]["error_handling"] = {
                    "status": "operational",
                    "statistics": error_stats,
                    "health_assessment": "good"
                }
                
                # Assess error health
                if error_stats.get("recent_errors_1h", 0) > 10:
                    diagnostics["components"]["error_handling"]["health_assessment"] = "concerning"
                elif error_stats.get("recent_errors_1h", 0) > 5:
                    diagnostics["components"]["error_handling"]["health_assessment"] = "needs_attention"
                
            except Exception as e:
                diagnostics["components"]["error_handling"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Configuration information
        if include_config_info:
            try:
                config = get_config()
                diagnostics["components"]["configuration"] = {
                    "status": "operational",
                    "environment": config.environment,
                    "debug_mode": config.debug_mode,
                    "token_limits": {
                        "summarisation_input": config.token_limits.max_input_tokens_summarisation,
                        "summarisation_output": config.token_limits.max_output_tokens_summary,
                        "relevance_input": config.token_limits.max_input_tokens_relevance_analysis,
                        "relevance_output": config.token_limits.max_output_tokens_relevance
                    },
                    "llm_models": {
                        "primary_model": config.llm_models.summarisation_model,
                        "fallback_model": config.llm_models.fallback_model,
                        "temperature": config.llm_models.temperature
                    },
                    "budget_allocation": {
                        "default_page_budget": config.budget_allocation.default_page_budget,
                        "adaptive_page_budget": config.budget_allocation.adaptive_page_budget,
                        "research_page_budget": config.budget_allocation.research_page_budget
                    },
                    "caching_enabled": {
                        "relevance_cache": config.cache_settings.enable_relevance_cache,
                        "summary_cache": config.cache_settings.enable_summary_cache,
                        "content_deduplication": config.cache_settings.enable_content_deduplication
                    }
                }
            except Exception as e:
                diagnostics["components"]["configuration"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Session manager status
        try:
            from .session_manager import session_manager
            session_count = len(session_manager._sessions)
            active_sessions = sum(1 for active in session_manager._sessions.values() if active)
            
            diagnostics["components"]["session_manager"] = {
                "status": "operational",
                "total_sessions": session_count,
                "active_sessions": active_sessions,
                "inactive_sessions": session_count - active_sessions,
                "crawler_initialized": session_manager._crawler is not None
            }
        except Exception as e:
            diagnostics["components"]["session_manager"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Overall health assessment
        component_statuses = [comp.get("status", "unknown") for comp in diagnostics["components"].values()]
        if all(status == "operational" for status in component_statuses):
            diagnostics["overall_health"] = "excellent"
        elif any(status == "error" for status in component_statuses):
            diagnostics["overall_health"] = "degraded"
        else:
            diagnostics["overall_health"] = "good"
        
        # Add recommendations
        recommendations = []
        
        if include_cache_stats and diagnostics["components"].get("cache_system", {}).get("performance_summary", {}).get("total_hit_rate", 0) < 0.3:
            recommendations.append("Consider adjusting cache settings or reviewing content patterns to improve cache hit rates")
        
        if include_error_stats and diagnostics["components"].get("error_handling", {}).get("statistics", {}).get("recent_errors_1h", 0) > 5:
            recommendations.append("Recent error rate is elevated - review error logs and consider system optimization")
        
        if recommendations:
            diagnostics["recommendations"] = recommendations
        
        return [types.TextContent(
            type="text",
            text=json.dumps(diagnostics, indent=2, ensure_ascii=False)
        )]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": f"System diagnostics failed: {str(e)}",
                "error_type": type(e).__name__,
                "timestamp": time.time()
            }, indent=2)
        )] 