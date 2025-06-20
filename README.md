# Crawl4AI MCP Server

A comprehensive Model Context Protocol (MCP) server implementation that provides both basic and advanced web crawling capabilities for Cursor AI and Claude Desktop's agent mode. Built on Crawl4AI, it offers everything from simple webpage scraping to sophisticated LLM-driven intelligent crawling tools.

## 🚀 Features

### Basic Web Crawling (No LLM Required)

- **Single Page Scraping**: Extract content and metadata from any webpage
- **Multi-Page Crawling**: Crawl websites with configurable depth and page limits
- **Content Extraction**: Get clean markdown and HTML content with metadata
- **Link Discovery**: Automatic link extraction and following
- **Robust Error Handling**: Graceful handling of failed pages and network issues

### Advanced LLM-Enhanced Capabilities (Optional)

- **Intelligent Web Crawling**: LLM-driven adaptive crawling with relevance scoring
- **Adaptive Link Selection**: AI-powered link prioritisation based on content relevance
- **Diminishing Returns Detection**: Automatic crawl termination when value decreases
- **Budget Management**: Multi-dimensional resource control (pages, tokens, time, depth)
- **Context-Aware Decisions**: LLM-driven crawling strategy adjustments
- **Topic Extraction**: Intelligent content categorisation and topic discovery

### System Features

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

The server provides 3 comprehensive MCP tools, ranging from basic web scraping (no LLM required) to advanced AI-powered crawling capabilities:

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
    max_pages=15,
    query="async programming"
)
```

## Advanced Tools (LLM-Enhanced)

### 3. Adaptive Website Crawling

#### `adaptive_crawl_website(url: str, query: str, max_budget: int = 20)`

**LLM-Powered Intelligent Crawling**: This is the most sophisticated tool, using AI to make intelligent decisions about which pages to crawl based on relevance to your query.

**Key Features:**

- **Relevance Scoring**: Each page gets an AI-calculated relevance score (0.0-1.0)
- **Diminishing Returns Detection**: Automatically stops when value decreases
- **Link Prioritisation**: AI selects the most promising links to follow
- **Budget Management**: Smart resource allocation across multiple dimensions
- **Topic Discovery**: Automatic categorisation and topic extraction

**Example:**

```python
adaptive_crawl_website(
    url="https://arxiv.org",
    query="machine learning transformers attention mechanisms",
    max_budget=25
)
```

**Response includes:**

- All crawled pages with relevance scores
- LLM evaluation and decision rationale
- Topic categorisation and insights
- Crawling efficiency metrics
- Budget utilisation breakdown

## 🎯 Use Cases and Examples

### Simple Content Extraction

```python
# Extract content from a blog post
result = scrape_webpage("https://blog.example.com/ai-trends-2024")
```

### Documentation Crawling

```python
# Crawl technical documentation
crawl_website(
    url="https://fastapi.tiangolo.com",
    crawl_depth=3,
    max_pages=25,
    query="authentication middleware"
)
```

### AI-Powered Research

```python
# Intelligent research on a specific topic
adaptive_crawl_website(
    url="https://paperswithcode.com",
    query="computer vision object detection YOLO architectures",
    max_budget=30
)
```

## 📊 Performance and Limitations

### Performance Characteristics

- **Basic Scraping**: ~1-3 seconds per page
- **Standard Crawling**: Scales linearly with page count
- **Adaptive Crawling**: ~2-5 seconds per page (includes LLM analysis)
- **Memory Usage**: ~50-200MB for typical crawling sessions
- **Concurrent Requests**: Configurable, defaults to respectful crawling

### Rate Limiting and Respect

- Built-in delays between requests (configurable)
- Respects robots.txt files
- Implements exponential backoff for failed requests
- User-agent identification for transparency

### Current Limitations

- **JavaScript-Heavy Sites**: May miss dynamically loaded content
- **Authentication**: No support for login-required pages
- **Large Files**: Not optimised for documents >10MB
- **Real-time Updates**: No websocket or streaming support

## 🔧 Configuration Options

### Basic Configuration

```env
# Crawling behaviour
CRAWLING_DEFAULT_MAX_PAGES=20
CRAWLING_DEFAULT_DEPTH=3
CRAWLING_REQUEST_DELAY=1.0

# Performance tuning
CACHE_MAX_SIZE=1000
CACHE_TTL_SECONDS=3600
```

### Advanced Configuration

```env
# Token management (for content processing)
TOKEN_SUMMARISATION_INPUT=8000
TOKEN_SUMMARISATION_OUTPUT=500
TOKEN_RELEVANCE_ANALYSIS=4000

# Error handling
ERROR_MAX_RETRIES=3
ERROR_RETRY_DELAY=2.0

# Cache behaviour
CACHE_ENABLE_RELEVANCE_CACHE=true
CACHE_ENABLE_SUMMARY_CACHE=true
CACHE_ENABLE_CONTENT_DEDUPLICATION=true
```

## 🧪 Development and Testing

### Running Tests

```bash
# Install test dependencies
uv sync --extra test

# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=tools

# Run specific test categories
python -m pytest tests/test_performance.py
python -m pytest tests/test_adaptive_crawling.py
```

### Development Setup

```bash
# Development environment
uv sync --extra dev

# Pre-commit hooks (optional)
pre-commit install

# Run linting
ruff check .
ruff format .
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Run the test suite: `python -m pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on [Crawl4AI](https://github.com/unclecode/crawl4ai) - An excellent web crawling framework
- Uses [FastMCP](https://github.com/pydantic/fastmcp) for MCP server implementation
- Inspired by the needs of AI-powered research and content analysis

---

**Note**: This is an independent implementation and is not officially affiliated with Anthropic's MCP specification, though it follows MCP standards for compatibility.
