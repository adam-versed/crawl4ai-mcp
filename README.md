# Crawl4AI MCP Server

A comprehensive Model Context Protocol (MCP) server implementation that provides both basic and advanced web crawling capabilities for Cursor AI and Claude Desktop's agent mode. Built on Crawl4AI, it offers everything from simple webpage scraping to sophisticated LLM-driven intelligent research tools.

## 🚀 Features

### Basic Web Crawling (No LLM Required)

- **Single Page Scraping**: Extract content and metadata from any webpage
- **Multi-Page Crawling**: Crawl websites with configurable depth and page limits
- **Content Extraction**: Get clean markdown and HTML content with metadata
- **Link Discovery**: Automatic link extraction and following
- **Robust Error Handling**: Graceful handling of failed pages and network issues

### Advanced LLM-Enhanced Capabilities (Optional)

- **Intelligent Web Crawling**: LLM-driven adaptive crawling with relevance scoring
- **Smart Summarisation**: Token-aware content summarisation with focus areas
- **Content Analysis**: Advanced relevance analysis and key insight extraction
- **Research Tools**: Comprehensive topic research with structured findings
- **Adaptive Link Selection**: AI-powered link prioritisation based on content relevance
- **Diminishing Returns Detection**: Automatic crawl termination when value decreases
- **Budget Management**: Multi-dimensional resource control (pages, tokens, time, depth)
- **Context-Aware Decisions**: LLM-driven crawling strategy adjustments
- **Topic Extraction**: Intelligent content categorisation and topic discovery

### System Features

- **Performance Monitoring**: Real-time system diagnostics and health assessment
- **Advanced Caching**: LRU cache with TTL support for optimal performance
- **Error Resilience**: Comprehensive error handling with fallback strategies
- **Flexible Configuration**: Environment-based configuration management

## 📋 System Requirements

- Python 3.12 or higher
- UV package manager (recommended) or pip
- Memory: 2GB+ RAM recommended for large-scale crawling
- Network: Stable internet connection for web crawling

## 🛠️ Installation

### Quick Start with UV (Recommended)

First, install UV package manager:

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**After installation:**

```bash
# Clone the repository
git clone <repository-url>
cd crawl4ai-mcp

# Install dependencies and activate environment
uv venv
uv sync
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install test dependencies (optional)
uv sync --extra test

# Run the server
python main.py
```

### Alternative Installation with Pip

```bash
# Clone and setup
git clone <repository-url>
cd crawl4ai-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

## ⚙️ Configuration

### MCP Server Configuration

Add to your Cursor or Claude Desktop configuration:

```json
{
  "mcpServers": {
    "Crawl4AI": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/crawl4ai-mcp",
        "run",
        "main.py"
      ]
    }
  }
}
```

> **Note**: Use the full path to the UV executable. Find it with `which uv` (macOS/Linux) or `where uv` (Windows).

### Environment Configuration

Create a `.env` file for custom configuration:

```env
# Crawling Configuration
CRAWLING_DEFAULT_MAX_PAGES=20
CRAWLING_DEFAULT_DEPTH=3
CRAWLING_REQUEST_DELAY=1.0

# Token Limits (for content processing, not LLM API calls)
TOKEN_SUMMARISATION_INPUT=8000
TOKEN_SUMMARISATION_OUTPUT=500
TOKEN_RELEVANCE_ANALYSIS=4000

# Caching Configuration
CACHE_MAX_SIZE=1000
CACHE_TTL_SECONDS=3600

# Error Handling
ERROR_MAX_RETRIES=3
ERROR_RETRY_DELAY=2.0
```

> **Important**: LLM model, temperature, and API settings are controlled by your MCP client (Cursor/Claude Desktop), not by this server. The server uses MCP sampling to request LLM operations from the client, so you don't need to configure LLM access separately.

## 🔧 Available Tools

The server provides 9 comprehensive MCP tools, ranging from basic web scraping (no LLM required) to advanced AI-powered research capabilities:

## Basic Tools (No LLM Required)

### 1. Web Page Scraping

#### `scrape_webpage(url: str)`

**Simple and Fast**: Scrapes content and metadata from a single webpage using Crawl4AI's core engine.

**Example:**

```python
# In Cursor Composer or Claude
scrape_webpage("https://example.com/article")
```

**Response:**

```json
{
  "success": true,
  "url": "https://example.com/article",
  "title": "Article Title",
  "markdown": "# Article Content...",
  "metadata": {
    "description": "Article description",
    "keywords": ["keyword1", "keyword2"]
  }
}
```

### 2. Website Crawling

#### `crawl_website(url: str, crawl_depth: int = 1, max_pages: int = 5, query: str = "")`

**Traditional Crawling**: Crawls multiple pages with configurable depth and limits. The optional `query` parameter adds basic relevance filtering.

**Example (Basic Usage):**

```python
# Simple crawling without LLM features
crawl_website(
    url="https://docs.python.org",
    crawl_depth=2,
    max_pages=10
)
```

**Example (With Basic Query Filtering):**

```python
# Basic relevance filtering (keyword matching)
crawl_website(
    url="https://docs.python.org",
    crawl_depth=2,
    max_pages=10,
    query="async programming tutorials"
)
```

## Advanced LLM-Enhanced Tools

### 3. Adaptive Intelligent Crawling

#### `adaptive_crawl_website(url: str, query: str, max_budget: int = 20)`

**Most Advanced**: LLM-driven crawling with intelligent link selection and relevance scoring.

**Example:**

```python
adaptive_crawl_website(
    url="https://machinelearning.com",
    query="transformer architecture and attention mechanisms",
    max_budget=30
)
```

**Key Features:**

- AI-powered link prioritisation
- Automatic relevance scoring
- Diminishing returns detection
- Budget-aware crawling
- Context-driven decisions

### 4. Content Summarisation

#### `summarise_webpage(url: str, max_tokens: int = 500, focus: str = "")`

**LLM-Powered**: Scrapes and intelligently summarises webpage content with token management and focus areas.

**Example:**

```python
summarise_webpage(
    url="https://research-paper.com/article",
    max_tokens=300,
    focus="key findings and methodology"
)
```

#### `summarise_crawl_results_tool(crawl_results: str, query: str = "", include_page_summaries: bool = True)`

**LLM-Powered**: Summarises crawl results with query focus and individual page summaries.

### 5. Content Analysis

#### `analyse_content_relevance(content: str, query: str)`

**LLM-Powered**: Uses advanced LLM analysis to score content relevance against a specific query.

**Example:**

```python
analyse_content_relevance(
    content="Article about machine learning techniques...",
    query="deep learning neural networks"
)
```

#### `extract_key_insights_tool(content: str, focus_areas: list[str] = [], max_tokens: int = 600)`

**LLM-Powered**: Extracts key insights from content with optional focus areas using intelligent analysis.

**Example:**

```python
extract_key_insights_tool(
    content="Research paper content...",
    focus_areas=["methodology", "results", "conclusions"],
    max_tokens=400
)
```

### 6. Research Tools

#### `research_topic(starting_url: str, research_query: str, max_pages: int = 25, depth_strategy: str = "adaptive")`

**Most Comprehensive & LLM-Powered**: Conducts thorough research using intelligent crawling and analysis.

**Example:**

```python
research_topic(
    starting_url="https://arxiv.org/list/cs.AI/recent",
    research_query="recent advances in large language model training efficiency",
    max_pages=50,
    depth_strategy="adaptive"
)
```

**Response Structure:**

```json
{
  "success": true,
  "research_query": "...",
  "research_summary": {
    "total_pages_analysed": 45,
    "successful_pages": 42,
    "high_relevance_pages": 15,
    "average_relevance_score": 0.73,
    "research_efficiency": 0.85
  },
  "key_insights": "Comprehensive analysis of findings...",
  "high_value_sources": [...],
  "topic_coverage": [...],
  "llm_evaluation": {...}
}
```

## System Tools

### 7. System Monitoring

#### `system_diagnostics(include_cache_stats: bool = True, include_error_stats: bool = True, include_config_info: bool = False)`

**System Health**: Provides comprehensive system health and performance metrics. Works with or without LLM features.

**Example:**

```python
system_diagnostics()
```

## 🤔 Basic vs. LLM-Enhanced Usage

### Basic Usage (No LLM Required)

Perfect for simple web scraping and crawling tasks:

```python
# Simple page scraping
scrape_webpage("https://example.com")

# Basic website crawling
crawl_website("https://docs.python.org", crawl_depth=2, max_pages=10)

# System monitoring
system_diagnostics()
```

**Advantages:**

- ✅ Fast and lightweight
- ✅ No external API dependencies
- ✅ Reliable and consistent
- ✅ Perfect for basic data extraction

### LLM-Enhanced Usage (Requires MCP Sampling)

Ideal for intelligent content analysis and research:

```python
# AI-powered adaptive crawling
adaptive_crawl_website("https://example.com", "machine learning tutorials", max_budget=20)

# Intelligent content summarisation
summarise_webpage("https://research-paper.com", focus="key findings")

# Advanced topic research
research_topic("https://arxiv.org", "quantum computing algorithms", max_pages=30)
```

**Advantages:**

- ✅ Intelligent content analysis
- ✅ Adaptive crawling decisions
- ✅ Relevance-based filtering
- ✅ Comprehensive research capabilities

## 📊 Advanced Features

### Budget Management

Control resource usage across multiple dimensions:

- **Page Budget**: Maximum pages to crawl
- **Token Budget**: Maximum LLM tokens to consume
- **Time Budget**: Maximum processing time
- **Depth Budget**: Maximum crawl depth

### Intelligent Caching

- **LRU Cache**: Automatic memory management
- **TTL Support**: Time-based cache expiry
- **Content Deduplication**: Smart similarity detection
- **Performance Monitoring**: Real-time cache statistics

### Error Resilience

- **Exponential Backoff**: Smart retry logic
- **Graceful Degradation**: Fallback to traditional crawling
- **Session Cleanup**: Automatic resource management
- **Error Classification**: Intelligent error handling

## 📈 Performance Optimization

### Recommended Settings by Use Case

**Basic Web Scraping (Fast & Lightweight):**

```python
# Single page scraping
scrape_webpage("https://example.com")

# Simple multi-page crawling
crawl_website("https://docs.python.org", crawl_depth=2, max_pages=15)
```

**Quick Research with LLM (5-10 pages):**

```python
adaptive_crawl_website(
    url="starting_url",
    query="your_query",
    max_budget=10
)
```

**Comprehensive Research (20-50 pages):**

```python
research_topic(
    starting_url="starting_url",
    research_query="detailed_query",
    max_pages=50,
    depth_strategy="adaptive"
)
```

**Large-Scale Analysis (100+ pages):**

```python
# Use environment configuration
# Set higher token limits and cache sizes
research_topic(
    starting_url="starting_url",
    research_query="comprehensive_query",
    max_pages=200,
    depth_strategy="adaptive"
)
```

## 🧪 Testing

### Run the Test Suite

```bash
# Install test dependencies
uv sync --extra test

# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m performance   # Performance tests only

# Run with coverage
pytest --cov=tools --cov-report=html
```

### Test Categories

- **Unit Tests**: Core functionality (token counting, utilities)
- **Integration Tests**: Adaptive crawling, LLM integration
- **Performance Tests**: Large-scale crawling, resource usage

## 🔍 Usage Examples

### Example 1: Academic Research

```python
# Research a specific academic topic
research_topic(
    starting_url="https://scholar.google.com/scholar?q=quantum+computing",
    research_query="quantum computing algorithms and error correction",
    max_pages=30,
    depth_strategy="adaptive"
)
```

### Example 2: Technical Documentation Analysis

```python
# Analyse technical documentation
adaptive_crawl_website(
    url="https://kubernetes.io/docs/",
    query="container orchestration best practices",
    max_budget=25
)
```

### Example 3: Content Summarisation Workflow

```python
# 1. Crawl relevant pages
crawl_results = crawl_website(
    url="https://blog.example.com",
    query="machine learning tutorials",
    max_pages=15
)

# 2. Summarise the results
summarise_crawl_results_tool(
    crawl_results=crawl_results,
    query="machine learning tutorials",
    include_page_summaries=True
)
```

### Example 4: System Health Monitoring

```python
# Monitor system performance
system_diagnostics(
    include_cache_stats=True,
    include_error_stats=True,
    include_config_info=True
)
```

## 🛡️ Security & Privacy

- **No Data Persistence**: Content is processed in memory only
- **Session Management**: Automatic cleanup of crawl sessions
- **Rate Limiting**: Respectful crawling with configurable delays
- **Error Isolation**: Failures don't affect other operations
- **Resource Limits**: Configurable bounds on resource usage

## 🔧 Troubleshooting

### Common Issues

**1. MCP Server Not Starting**

- Verify Python 3.12+ is installed
- Check UV installation: `uv --version`
- Ensure all dependencies are installed: `uv sync`

**2. Slow Crawling Performance**

- Increase `CRAWLING_REQUEST_DELAY` in configuration
- Reduce `max_budget` for faster results
- Check system resources with `system_diagnostics()`

**3. LLM Integration Issues**

- Verify MCP sampling is working
- Check error logs in system diagnostics
- Reduce token limits if hitting API limits

**4. Memory Usage**

- Monitor cache statistics
- Reduce cache sizes in configuration
- Use smaller `max_budget` values

### Performance Tuning

**For Speed:**

```env
CRAWLING_REQUEST_DELAY=0.5
CACHE_MAX_SIZE=500
TOKEN_SUMMARISATION_INPUT=4000
```

**For Accuracy:**

```env
CRAWLING_REQUEST_DELAY=2.0
CACHE_MAX_SIZE=2000
TOKEN_SUMMARISATION_INPUT=12000
```

## 📚 API Reference

### Configuration Options

| Setting                      | Default | Description                        |
| ---------------------------- | ------- | ---------------------------------- |
| `CRAWLING_DEFAULT_MAX_PAGES` | `20`    | Default page limit                 |
| `TOKEN_SUMMARISATION_INPUT`  | `8000`  | Max tokens for summarisation input |
| `CACHE_MAX_SIZE`             | `1000`  | Maximum cache entries              |
| `ERROR_MAX_RETRIES`          | `3`     | Maximum retry attempts             |

> **Note**: LLM-related settings (model, temperature, etc.) are not configurable here as they are controlled by your MCP client.

### Response Formats

All tools return JSON responses with consistent structure:

```json
{
  "success": true|false,
  "data": {...},
  "error": "error message (if applicable)",
  "metadata": {...}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `uv sync --extra test`
4. Run tests: `pytest`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Crawl4AI](https://github.com/unclecode/crawl4ai) for the core crawling engine
- [Model Context Protocol](https://modelcontextprotocol.io/) for the MCP framework
- [tiktoken](https://github.com/openai/tiktoken) for accurate token counting

---

**Made with ❤️ for intelligent web crawling and research**
