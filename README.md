# Nike Shoe Scraper API

A sophisticated web scraping system that uses AI agents to extract Nike shoe product data from nike.com. The system combines LangChain, LangGraph, Anthropic's Claude AI, and Bright Data's MCP (Model Context Protocol) server to create an intelligent scraping pipeline.

##  Architecture Overview

The system uses a multi-stage workflow orchestrated by LangGraph:

```
User Query → Load MCP Tools → Scrape with AI Agent → Parse Data → Return Products
```

## 🔧 Core Components

### 1. **FastAPI Web Server** (`app`)
- Provides REST API endpoint at `/search`
- Handles CORS for frontend integration
- Accepts search queries and returns structured product data

### 2. **LangGraph Workflow** (`langgraph_app`)
A state machine that orchestrates the scraping process through these nodes:

#### **Load Tools Node** (`load_tools_node`)
- Initializes connection to Bright Data MCP server
- Loads web scraping tools using the official MCP adapter
- Returns available tools for the AI agent

#### **Scrape with Tools Node** (`scrape_with_tools_node`)
- Creates a ReAct (Reasoning + Acting) agent using Claude AI
- Instructs the agent to scrape the specific Nike URL: `https://www.nike.com/w/mens-shoes-nik1zy7ok`
- Uses `scrape_as_markdown` tool to extract page content
- Focuses on extracting product names, prices, URLs, and images

#### **Parse Data Node** (`parse_data_node`)
- Uses Claude AI to parse the raw scraped markdown content
- Converts unstructured data into structured JSON
- Extracts and validates product URLs (handles partial URLs and markdown links)
- Creates `ProductData` objects with proper validation

### 3. **MCP Connection Setup** (`mcp_server.py`)
- Establishes connection to Bright Data's MCP server
- Configures the Web Unlocker zone for bypassing anti-bot protection
- Uses stdio transport for communication with the MCP server

## 📊 Data Models

### **ProductData**
```python
{
    "name": str,           # Product name
    "price": str,          # Price (e.g., "$120.00")
    "currency": str,       # Always "USD"
    "availability": str,   # "In Stock", "Out of Stock", or "Limited"
    "product_url": str,    # Direct Nike product URL
    "image_url": str       # Product image URL (optional)
}
```

### **ScrapeProcessState**
Tracks the workflow state through each processing step:
- `user_query`: Original search query
- `optimized_query`: AI-optimized query for scraping
- `mcp_tools`: Available MCP tools
- `messages`: LangGraph message chain
- `scraped_data`: Raw scraped content
- `parsed_products`: Final structured products
- `error_message`: Any errors encountered

##  How It Works

### 1. **Request Processing**
```python
POST /search
{
    "query": "nike air max shoes"
}
```

### 2. **Tool Loading**
- Connects to Bright Data MCP server using environment credentials
- Loads web scraping capabilities through the MCP protocol
- Validates connection and available tools

### 3. **AI-Powered Scraping**
- Creates a ReAct agent with Claude AI and MCP tools
- Agent receives specific instructions to:
  - Use ONLY the `scrape_as_markdown` tool
  - Target the exact Nike mens shoes URL
  - Extract product information systematically
  - Focus on complete URL extraction

### 4. **Intelligent Parsing**
- Claude AI analyzes the scraped markdown content
- Identifies product information using pattern recognition
- Handles various URL formats:
  - Complete URLs: `https://www.nike.com/t/product-name/code`
  - Markdown links: `[Product Name](URL)`
  - Partial URLs: `/t/product-path` → converted to full URLs
  - Reference-style markdown links

### 5. **Response Formatting**
- Validates all extracted data
- Creates structured `ProductData` objects
- Returns JSON response with product array

##  Environment Variables

```bash
ANTHROPIC_API_KEY=your_claude_api_key
BRIGHTDATA_API_TOKEN=your_brightdata_token
```

## 🛠️ Key Features

### **Anti-Bot Protection**
- Uses Bright Data's Web Unlocker to bypass Nike's bot detection
- MCP protocol provides reliable access to protected content

### **AI-Driven Extraction**
- Claude AI understands context and extracts relevant product data
- Handles dynamic content and various page layouts
- Intelligent URL completion and validation

### **Error Handling**
- Comprehensive error catching at each workflow stage
- Graceful degradation when tools or parsing fail
- Detailed logging for debugging

### **Structured Workflow**
- LangGraph ensures proper execution order
- State management tracks progress through each step
- Conditional edges handle error scenarios

## URL Extraction Intelligence

The system is particularly sophisticated in URL extraction:

1. **Markdown Link Detection**: Finds `[text](url)` patterns
2. **Embedded URL Scanning**: Searches entire content for Nike URLs
3. **Partial URL Completion**: Converts `/t/product` to full URLs
4. **Product Code Matching**: Links product codes to their URLs
5. **Contextual Association**: Matches product names with nearby URLs

##  API Usage

### Request
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "running shoes"}'
```

### Response Format
```json
{
    "products": [
        {
            "name": "Nike Air Max DN8",
            "price": "$120.00",
            "currency": "USD",
            "availability": "In Stock",
            "product_url": "https://www.nike.com/t/air-max-dn8-mens-shoes/FQ7860-010",
            "image_url": "https://static.nike.com/..."
        }
    ]
}
```

## 🔍 Troubleshooting

### Common Issues:
1. **MCP Connection Fails**: Check `BRIGHTDATA_API_TOKEN` environment variable
2. **Parsing Errors**: Check Claude AI API key and quota
3. **Tool Loading Issues**: Ensure MCP server dependencies are installed

### Debug Logging:
The system provides detailed console output for each stage:
- 🔧 Tool initialization
- 🔍 Scraping progress  
- 📄 Content extraction
- 📊 Data parsing
- ✅ Success confirmations
- ❌ Error details

##  Running the System

1. Install dependencies
2. Set environment variables
3. Start FastAPI server: `uvicorn main:app --reload`
4. Send POST requests to `/search` endpoint

The system will automatically handle the complete scraping pipeline and return structured Nike shoe data.