# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, ValidationError
import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Any, Optional
import asyncio

# Import updated MCP server functions
from mcp_server import (
    connect_to_mcp, 
    process_query,
    # get_mcp_tools,
    # get_mcp_server()
   
    
)

# LangChain/LangGraph imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# --- Configuration ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it.")

# Bright Data MCP Configuration
BRIGHT_DATA_API_KEY = os.getenv("BRIGHTDATA_API_TOKEN")

if not BRIGHT_DATA_API_KEY:
    raise ValueError("BRIGHT_DATA_API_KEY not found. Please set it in environment variables")

print(f"✅ Anthropic API Key loaded: {ANTHROPIC_API_KEY[:5]}...")
print(f"✅ Bright Data API Key configured")

# --- FastAPI App Setup ---
app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LangChain's Claude LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.0, api_key=ANTHROPIC_API_KEY)

# Initialize MCP on startup
async def initialize_mcp():
    """Initialize the MCP client with Bright Data server"""
    try:
        mcp_client = await connect_to_mcp()
        if mcp_client:
            print("✅ MCP client initialized successfully")
            # List available tools
            tools = await get_mcp_tools()
            print(f"✅ Available MCP tools: {tools}")
            return mcp_client
        else:
            print("❌ Failed to initialize MCP client")
            raise Exception("MCP initialization failed")
    except Exception as e:
        print(f"❌ Failed to initialize MCP client: {e}")
        raise

async def close_mcp():
    """Close the MCP client"""
    await close_mcp_server()

# --- Pydantic Models ---
class SearchRequest(BaseModel):
    query: str

class ProductData(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra='ignore'
    )
    
    name: str = Field(..., description="The full name of the product.")
    price: str = Field(..., description="The price of the product, including currency symbol.")
    currency: str = Field(..., description="The currency symbol or code.")
    availability: str = Field(..., description="Availability status.")
    source_url: str = Field(..., description="The direct URL to the product page.")
    image_url: Optional[str] = Field(None, description="The URL of the product's main image.")
    colors: Optional[str] = Field(None, description="Number of colors available or color names.")
    model_number: Optional[str] = Field(None, description="The product's model or style number.")
    description_snippet: Optional[str] = Field(None, description="A short snippet from the product description.")

class SearchResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)
    products: List[ProductData]

# --- Updated MCP Tool for LangChain ---
@tool
async def scrape_nike_shoes_with_mcp(url: str, user_query_context: str) -> str:
    """
    Scrapes product data from a specific URL using Bright Data MCP tools.
    
    Args:
        url (str): The URL to scrape.
        user_query_context (str): The original user's query for context.
        
    Returns:
        str: Raw HTML content from the scraped page or error message.
    """
    try:
        print(f"Attempting to scrape URL: {url} using Bright Data MCP...")
        
        # Get available MCP tools
        available_tools = await get_mcp_tools()
        print(f"Available MCP tools: {available_tools}")
        
        # Look for a suitable scraping tool - common Bright Data MCP tool names
        scraping_tool_name = None
        potential_tool_names = ['scrape', 'fetch', 'browse', 'get_page', 'web_scraper', 'scrape_page']
        
        for tool_name in available_tools:
            tool_lower = tool_name.lower()
            if any(keyword in tool_lower for keyword in potential_tool_names):
                scraping_tool_name = tool_name
                break
        
        if not scraping_tool_name:
            return f"No suitable scraping tool found. Available tools: {available_tools}"
        
        print(f"Using MCP tool: {scraping_tool_name}")
        
        # Execute the scraping tool with the URL
        # Note: The exact parameter name might vary depending on the Bright Data MCP implementation
        try:
            result = await process_query(scraping_tool_name, url=url)
        except Exception as e:
            # Try alternative parameter names if 'url' doesn't work
            try:
                result = await process_query(scraping_tool_name, target_url=url)
            except Exception:
                try:
                    result = await process_query(scraping_tool_name, webpage_url=url)
                except Exception:
                    raise e  # Re-raise the original exception
        
        # Process the result
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            # Extract content from dictionary response
            content_keys = ['content', 'html', 'text', 'body', 'page_content', 'data']
            for key in content_keys:
                if key in result:
                    return str(result[key])
            # If no standard key found, return the whole result as string
            return str(result)
        else:
            return str(result)
            
    except Exception as e:
        print(f"Error calling Bright Data MCP: {e}")
        return f"Failed to scrape data: {e}"

# --- LangGraph Agent State ---
class ScrapeProcessState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    user_query: str
    optimized_query: Optional[str] = None
    raw_scraped_data: Optional[str] = None
    parsed_products: List[ProductData] = Field(default_factory=list)
    error_message: Optional[str] = None

async def optimize_query_node(state: ScrapeProcessState) -> dict:
    print("Step 1: Optimizing user query with Claude...")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant. Rephrase the user's query to give context for a web scraper targeting 'Nike men's shoes'. Make the query more specific and actionable for web scraping."),
        ("human", "User query: {query}"),
    ])
    chain = prompt_template | llm
    result = await chain.ainvoke({"query": state.user_query})
    return {"optimized_query": result.content}

async def perform_scraping_node(state: ScrapeProcessState) -> dict:
    print(f"Step 2: Performing scrape for '{state.optimized_query}'...")
    target_url = "https://www.nike.com/w/mens-shoes-nik1zy7ok"
    raw_data = await scrape_nike_shoes_with_mcp(url=target_url, user_query_context=state.optimized_query)
    
    # Check if scraping was successful
    error_indicators = [
        "Failed to scrape data", 
        "No suitable scraping tool found", 
        "Tool '", 
        "not found"
    ]
    
    if raw_data and not any(error in raw_data for error in error_indicators):
        return {"raw_scraped_data": raw_data}
    else:
        return {"error_message": raw_data}

async def parse_to_json_node(state: ScrapeProcessState) -> dict:
    print("Step 3: Parsing scraped data to JSON with Claude...")
    
    if not state.raw_scraped_data:
        return {"error_message": "No raw data to parse."}
    
    json_schema = json.dumps(ProductData.model_json_schema(), indent=2)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert web scraping data parser specializing in e-commerce product data.
        Extract Nike men's shoe product information from the provided HTML content.
        
        Return ONLY a valid JSON array of ProductData objects. Do not include markdown code blocks, explanations, or any other text.
        
        ProductData Schema: {json_schema}
        
        Important:
        - Extract actual product names, prices, and URLs
        - If currency is not explicit, assume USD ($)
        - For availability, use "In Stock", "Out of Stock", or "Limited" based on the content
        - Extract product URLs that link to individual product pages
        - Include image URLs if available
        - If no products are found, return an empty array []"""),
        ("human", "HTML content to parse:\n\n{raw_data}"),
    ])
    
    chain = prompt_template | llm
    
    try:
        result = await chain.ainvoke({"raw_data": state.raw_scraped_data})
        claude_response_content = result.content
        
        # Clean up the response
        response_content = claude_response_content.strip()
        
        # Remove markdown code blocks if present
        if response_content.startswith("```json"):
            response_content = response_content[7:]  # Remove ```json
        if response_content.endswith("```"):
            response_content = response_content[:-3]  # Remove ```
        
        response_content = response_content.strip()
        
        # Parse JSON
        parsed_products_raw = json.loads(response_content)
        
        # Validate and create ProductData objects
        if isinstance(parsed_products_raw, list):
            parsed_products = []
            for p in parsed_products_raw:
                try:
                    product = ProductData(**p)
                    parsed_products.append(product)
                except ValidationError as ve:
                    print(f"Validation error for product {p}: {ve}")
                    continue
            
            return {"parsed_products": parsed_products}
        else:
            return {"error_message": "Parsed data is not a list of products"}
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return {"error_message": f"JSON parsing failed: {e}"}
    except ValidationError as e:
        print(f"Validation error: {e}")
        return {"error_message": f"Product data validation failed: {e}"}
    except Exception as e:
        print(f"Unexpected error in parsing: {e}")
        return {"error_message": f"Parsing failed: {e}"}

# --- Build Async LangGraph Workflow ---
workflow = StateGraph(ScrapeProcessState)

workflow.add_node("optimize_query", optimize_query_node)
workflow.add_node("perform_scraping", perform_scraping_node)
workflow.add_node("parse_to_json", parse_to_json_node)

workflow.set_entry_point("optimize_query")
workflow.add_edge("optimize_query", "perform_scraping")
workflow.add_conditional_edges(
    "perform_scraping",
    lambda state: "parse_to_json" if not state.get("error_message") else END,
    {"parse_to_json": "parse_to_json", END: END}
)
workflow.add_edge("parse_to_json", END)

langgraph_app = workflow.compile()

# --- FastAPI Endpoints ---

@app.get("/")
async def root():
    return {"message": "Welcome to the Nike Men's Shoes Scraper API with Bright Data MCP"}

@app.get("/health")
async def health_check():
    """Health check endpoint that includes MCP server status."""
    mcp_client = get_mcp_server()
    tools = await get_mcp_tools() if mcp_client else []
    return {
        "status": "healthy",
        "mcp_client_initialized": mcp_client is not None,
        "available_tools": tools,
        "tool_count": len(tools)
    }

@app.get("/mcp/tools")
async def list_mcp_tools():
    """List all available MCP tools with their details."""
    try:
        tools = await get_mcp_tools()
        tool_info = []
        for tool in tools:
            tool_info.append({
                "name": tool.name,
                "description": getattr(tool, 'description', 'No description available'),
                "args": getattr(tool, 'args', {})
            })
        return {
            "tools": tool_info,
            "count": len(tool_info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tools: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_nike_shoes(request: SearchRequest):
    print(f"Received request for query: {request.query}")
    
    # Ensure MCP client is initialized
    mcp_client = get_mcp_server()
    if not mcp_client:
        print("MCP client not initialized, attempting to initialize...")
        success = await ()
        if not success:
            raise HTTPException(status_code=503, detail="MCP client unavailable. Please check Bright Data credentials and connection.")
    
    initial_state = ScrapeProcessState(user_query=request.query)
    
    try:
        # Execute the async LangGraph workflow
        final_state: ScrapeProcessState = await langgraph_app.ainvoke(initial_state)
        
        if final_state.error_message:
            raise HTTPException(status_code=500, detail=f"Scraping or parsing failed: {final_state.error_message}")
        
        if not final_state.parsed_products:
            raise HTTPException(status_code=404, detail="No products found. The page might not contain product data or the scraping failed.")

        return SearchResponse(products=final_state.parsed_products)
        
    except Exception as e:
        print(f"Workflow execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# @app.on_event("startup")
# async def startup_event():
#     print("FastAPI app starting up. Initializing MCP client...")
#     await initialize_mcp()

# @app.on_event("shutdown")
# async def shutdown_event():
#     print("FastAPI app shutting down. Closing MCP client...")
#     await close_mcp()
