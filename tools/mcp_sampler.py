"""
MCP Sampling Integration for LLM-powered functionality.

This module provides functionality to request LLM completions via MCP sampling,
enabling the crawl4ai server to use LLM capabilities for intelligent decisions.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from mcp.server.fastmcp import FastMCP
from mcp import types
import uuid

logger = logging.getLogger(__name__)


class MCPSampler:
    """Handles LLM sampling requests via MCP protocol."""
    
    def __init__(self, server: FastMCP):
        self.server = server
        self.default_model = "claude-3-5-sonnet-20241022"  # Default Anthropic model
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def request_llm_completion(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1000,
        model: Optional[str] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """Request an LLM completion via MCP sampling.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens in response
            model: Model to use (defaults to configured default)
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            
        Returns:
            LLM response text
            
        Raises:
            Exception: If sampling fails after retries
        """
        if not model:
            model = self.default_model
            
        # Attempt sampling with retries
        for attempt in range(self.max_retries):
            try:
                response = await self._perform_sampling(
                    messages=messages,
                    system_prompt=system_prompt,
                    model=model, 
                    max_tokens=max_tokens, 
                    temperature=temperature
                )
                
                logger.info(f"LLM sampling successful on attempt {attempt + 1}")
                return response
                
            except Exception as e:
                logger.warning(f"LLM sampling attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All LLM sampling attempts failed")
                    return "LLM sampling is not supported in the current MCP client. Please ensure your MCP client supports sampling capabilities to enable AI-powered analysis and decision-making features."
    
    async def _perform_sampling(
        self, 
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        model: str, 
        max_tokens: int, 
        temperature: float
    ) -> str:
        """Perform the actual sampling request via MCP protocol.
        
        This implementation uses the FastMCP server's sampling capabilities
        to request completions from the connected LLM client.
        """
        try:
            # Prepare the sampling request according to MCP specification
            sampling_messages = []
            
            # Add system message if provided
            if system_prompt:
                sampling_messages.append(types.SamplingMessage(
                    role="system",
                    content=types.TextContent(
                        type="text",
                        text=system_prompt
                    )
                ))
            
            # Add user messages
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                sampling_messages.append(types.SamplingMessage(
                    role=role,
                    content=types.TextContent(
                        type="text",
                        text=content
                    )
                ))
            
            # Create the sampling request using the correct MCP API
            params = types.CreateMessageRequestParams(
                messages=sampling_messages,
                systemPrompt=system_prompt,
                maxTokens=max_tokens,
                temperature=temperature,
                metadata={
                    "source": "crawl4ai-mcp",
                    "purpose": "intelligent_crawling"
                }
            )
            
            sampling_request = types.CreateMessageRequest(
                method="sampling/createMessage",
                params=params
            )
            
            # Make the sampling request through the MCP server
            # Note: This is the proper way to request sampling in MCP
            if hasattr(self.server, 'request_sampling'):
                sampling_result = await self.server.request_sampling(sampling_request)
            else:
                # Fallback: try to use the server's session to make sampling request
                session = getattr(self.server, 'session', None)
                if session and hasattr(session, 'create_message'):
                    sampling_result = await session.create_message(sampling_request)
                else:
                    # No sampling capability available
                    raise Exception("MCP client does not support sampling capabilities")
            
            # Extract the response text from the sampling result
            if isinstance(sampling_result, types.CreateMessageResult):
                if sampling_result.content and len(sampling_result.content) > 0:
                    first_content = sampling_result.content[0]
                    if isinstance(first_content, types.TextContent):
                        return first_content.text
                    else:
                        return str(first_content)
                else:
                    raise Exception("Empty response from sampling")
            else:
                # Handle different response formats
                if hasattr(sampling_result, 'message') and hasattr(sampling_result.message, 'content'):
                    return sampling_result.message.content
                elif hasattr(sampling_result, 'text'):
                    return sampling_result.text
                else:
                    return str(sampling_result)
                    
        except Exception as e:
            logger.error(f"MCP sampling failed: {str(e)}")
            raise e
    
    async def summarise_text(
        self, 
        text: str, 
        max_output_tokens: int = 500,
        focus: Optional[str] = None
    ) -> str:
        """Summarise text using LLM sampling.
        
        Args:
            text: Text to summarise
            max_output_tokens: Maximum tokens in summary
            focus: Optional focus area for summarisation
            
        Returns:
            Summary text
        """
        system_prompt = "You are a helpful assistant that creates clear, concise summaries."
        
        if focus:
            prompt = f"Please summarise the following text, focusing on {focus}:\n\n{text}"
        else:
            prompt = f"Please summarise the following text:\n\n{text}"
        
        messages = [{"role": "user", "content": prompt}]
        
        return await self.request_llm_completion(
            messages=messages,
            max_tokens=max_output_tokens,
            system_prompt=system_prompt,
            temperature=0.3  # Lower temperature for more consistent summaries
        )
    
    async def analyse_relevance(
        self, 
        content: str, 
        query: str, 
        max_tokens: int = 300
    ) -> Dict[str, Any]:
        """Analyse content relevance to a query using LLM.
        
        Args:
            content: Content to analyse
            query: Query to check relevance against
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with relevance analysis
        """
        system_prompt = """You are an expert at analysing content relevance. 
        Respond with a JSON object containing:
        - relevance_score: float between 0.0 and 1.0
        - reasoning: brief explanation
        - key_topics: list of main topics found
        - matches_query: boolean indicating if content matches the query"""
        
        prompt = f"""Analyse how relevant this content is to the query: "{query}"

Content to analyse:
{content}

Provide your analysis as a JSON object."""
        
        messages = [{"role": "user", "content": prompt}]
        
        response = await self.request_llm_completion(
            messages=messages,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            temperature=0.1  # Very low temperature for consistent JSON
        )
        
        # Check if response indicates sampling not supported
        if "sampling is not supported" in response.lower():
            return {
                "relevance_score": 0.5,
                "reasoning": "LLM sampling not available - using default scoring",
                "key_topics": [],
                "matches_query": False,
                "sampling_unavailable": True
            }
        
        # Parse JSON response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {response}")
            return {
                "relevance_score": 0.5,
                "reasoning": "Failed to parse LLM response - sampling may not be available",
                "key_topics": [],
                "matches_query": False,
                "parsing_error": True
            }


# Global sampler instance (to be initialised with server)
_sampler: Optional[MCPSampler] = None


def initialise_sampler(server: FastMCP) -> MCPSampler:
    """Initialise the global MCP sampler instance.
    
    Args:
        server: FastMCP server instance
        
    Returns:
        Initialised MCPSampler instance
    """
    global _sampler
    _sampler = MCPSampler(server)
    logger.info("MCP sampler initialised successfully")
    return _sampler


def get_sampler() -> MCPSampler:
    """Get the global MCP sampler instance.
    
    Returns:
        MCPSampler instance
        
    Raises:
        RuntimeError: If sampler not initialised
    """
    if _sampler is None:
        raise RuntimeError("MCP sampler not initialised. Call initialise_sampler() first.")
    return _sampler 