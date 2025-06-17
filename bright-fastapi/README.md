# FastAPI Application with LangGraph and MCP

A FastAPI application that leverages LangGraph for workflow orchestration and MCP (Model Context Protocol) adapters for intelligent data processing with Anthropic's Claude.

## 🚀 Features

- **FastAPI**: Modern, fast web framework for building APIs
- **LangGraph**: Workflow orchestration for complex AI applications
- **MCP Integration**: Model Context Protocol adapters for seamless tool integration
- **Anthropic Claude**: Advanced AI capabilities for data processing
- **Async Support**: Full asynchronous operation for better performance

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- An Anthropic API key

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-directory>
```

### 2. Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the root directory and add your environment variables:

```env
ANTHROPIC_API_KEY=
BRIGHTDATA_API_TOKEN=""
BRIGHTDATA_WEB_UNLOCKER_ZONE="mcp_unlocker" # e.g., mcp_unlocker
BRIGHTDATA_SCRAPING_BROWSER_AUTH=""  
```

**To get BrightMcP key:**

1. Visit the mcp website create an account
2. Create an API Token and 
3. Creata a scrapping browser auth


**To get your Anthropic API key:**
1. Sign up at [Anthropic Console](https://console.anthropic.com/)
2. Navigate to API Keys section
3. Create a new API key
4. Copy and paste it into your `.env` file

##  Running the Application

### Development Mode

```bash
uvicorn main:app --reload
```

## 📁 Project Structure

```
project/
├── main.py              # FastAPI application entry point
├── requirements.txt     # Python dependencies
├── .env                # Environment variables (create this)
├── .mcp_server.py       # mcp server
├── README.md           # This file
└── ...
```

##  Requirements.txt

```txt
fastapi
uvicorn
python-dotenv
langgraph
langchain-mcp-adapters
langchain-anthropic
```

##  Package Overview

- **FastAPI**: Web framework for building APIs with automatic OpenAPI documentation
- **Uvicorn**: ASGI server for running FastAPI applications
- **python-dotenv**: Load environment variables from .env files
- **LangGraph**: Graph-based workflow orchestration for AI applications
- **langchain-mcp-adapters**: Adapters for Model Context Protocol integration
- **langchain-anthropic**: LangChain integration with Anthropic's Claude models

## API Endpoint

```bash
 http://localhost:8000/search

```

## 🚨 Troubleshooting

### Common Issues

**1. Module Not Found Error**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**3. Port Already in Use**
```bash
# Use a different port
uvicorn main:app --reload --port 8001
```

### Virtual Environment Commands

**Activate virtual environment:**
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```


**Remove virtual environment:**
```bash
# Windows
rmdir /s venv

# macOS/Linux
rm -rf venv
```

### Code Structure

The application follows FastAPI best practices:
- Async/await for non-blocking operations
- Pydantic models for request/response validation
- Proper error handling and HTTP status codes
- Environment-based configuration





##  Support

For issues and questions:
- Check the [FastAPI documentation](https://fastapi.tiangolo.com/)
- Review [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- Consult [Anthropic API documentation](https://docs.anthropic.com/)

---
