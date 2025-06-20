import asyncio
import logging
import uuid
from typing import Dict, Optional
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from .config import get_config

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages browser sessions for the MCP server to prevent session corruption."""
    
    def __init__(self):
        self._crawler: Optional[AsyncWebCrawler] = None
        self._sessions: Dict[str, bool] = {}  # session_id -> is_active
        self._lock = asyncio.Lock()
        self._config = get_config()
        
    async def get_crawler(self) -> AsyncWebCrawler:
        """Get or create the main crawler instance."""
        async with self._lock:
            if self._crawler is None:
                browser_config = BrowserConfig(
                    headless=self._config.crawler_settings.headless,
                    verbose=False,
                    # Enhanced browser configuration for stability
                    extra_args=[
                        "--disable-gpu",
                        "--disable-web-security", 
                        "--disable-features=VizDisplayCompositor",
                        "--disable-blink-features=AutomationControlled",
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--disable-background-timer-throttling",
                        "--disable-backgrounding-occluded-windows",
                        "--disable-renderer-backgrounding",
                        "--disable-background-networking",
                        "--disable-sync",
                        "--disable-translate",
                        "--hide-scrollbars",
                        "--disable-plugins",
                        "--disable-extensions",
                        "--disable-http-cache",  # Disable HTTP cache
                        "--disable-application-cache",  # Disable application cache
                        "--disable-offline-load-stale-cache",  # Disable offline cache
                    ]
                )
                
                self._crawler = AsyncWebCrawler(config=browser_config)
                await self._crawler.start()
                logger.info("Created new crawler instance")
                
            return self._crawler
    
    async def get_session_config(self, operation_type: str = "default") -> CrawlerRunConfig:
        """Get a crawler run config with unique session ID for each request."""
        # Create unique session ID for each request to prevent response caching
        session_id = f"{operation_type}_{uuid.uuid4().hex[:8]}"
        
        async with self._lock:
            self._sessions[session_id] = True
            
        config = CrawlerRunConfig(
            session_id=session_id,
            cache_mode=CacheMode.BYPASS,  # Always bypass cache
            page_timeout=self._config.crawler_settings.page_timeout_ms,
            verbose=False,
            # Additional settings to prevent caching
            wait_until="domcontentloaded",
            delay_before_return_html=0.5,
            # Force fresh page load - but don't use aggressive settings that break sessions
            magic=self._config.crawler_settings.magic_mode,  # Enable magic mode for better content extraction
        )
        
        logger.info(f"Created new session: {session_id}")
        return config
    
    async def cleanup_session(self, session_id: str):
        """Mark session as inactive but don't immediately clean up to avoid browser context issues."""
        async with self._lock:
            if session_id in self._sessions:
                # Just mark as inactive rather than immediate cleanup
                self._sessions[session_id] = False
                logger.debug(f"Marked session as inactive: {session_id}")
    
    async def cleanup_old_sessions(self):
        """Clean up inactive sessions periodically."""
        async with self._lock:
            # Only clean up if we have too many sessions
            if len(self._sessions) > self._config.crawler_settings.max_concurrent_sessions:
                inactive_sessions = [sid for sid, active in self._sessions.items() if not active]
                
                # Clean up oldest inactive sessions, keep some buffer
                cleanup_threshold = self._config.crawler_settings.session_cleanup_threshold
                cleanup_buffer = self._config.crawler_settings.session_cleanup_buffer
                sessions_to_cleanup = inactive_sessions[:-cleanup_buffer] if len(inactive_sessions) > cleanup_threshold else []
                
                for session_id in sessions_to_cleanup:
                    try:
                        if self._crawler and hasattr(self._crawler.crawler_strategy, 'kill_session'):
                            await self._crawler.crawler_strategy.kill_session(session_id)
                            logger.info(f"Cleaned up old session: {session_id}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up old session {session_id}: {e}")
                    finally:
                        if session_id in self._sessions:
                            del self._sessions[session_id]
    
    async def cleanup_all_sessions(self):
        """Clean up all sessions and shut down the crawler."""
        async with self._lock:
            # Clean up all sessions
            for session_id in list(self._sessions.keys()):
                try:
                    if self._crawler and hasattr(self._crawler.crawler_strategy, 'kill_session'):
                        await self._crawler.crawler_strategy.kill_session(session_id)
                        logger.info(f"Cleaned up session: {session_id}")
                except Exception as e:
                    logger.warning(f"Error cleaning up session {session_id}: {e}")
            
            self._sessions.clear()
            
            # Shut down the crawler
            if self._crawler:
                try:
                    await self._crawler.stop()
                    logger.info("Shut down crawler")
                except Exception as e:
                    logger.warning(f"Error shutting down crawler: {e}")
                finally:
                    self._crawler = None

    async def handle_session_error(self, session_id: str):
        """Handle session errors by marking as inactive."""
        logger.warning(f"Handling session error for: {session_id}")
        async with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id] = False


# Global session manager instance
session_manager = SessionManager()


async def cleanup_on_exit():
    """Cleanup function to be called on server shutdown."""
    await session_manager.cleanup_all_sessions() 