# Correct imports based on the official documentation
import asyncio
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Annotated, Sequence
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from mcp_server import setup_mcp_connection
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import traceback

load_dotenv()

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

# Your existing models
class ProductData(BaseModel):
    name: str
    price: str
    currency: str = "USD"
    availability: str
    product_url: str
    image_url: str = ""

class SearchResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)
    products: List[ProductData]

class SearchRequest(BaseModel):
    query: str

# Updated state model to work with LangGraph message pattern
class ScrapeProcessState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    user_query: str
    optimized_query: Optional[str] = None
    mcp_tools: List = Field(default_factory=list)
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list)
    scraped_data: Optional[str] = None
    parsed_products: List[ProductData] = Field(default_factory=list)
    error_message: Optional[str] = None

# Initialize LLM
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

async def load_tools_node(state: ScrapeProcessState) -> dict:
    """Load MCP tools using the official adapter"""
    print("Loading MCP tools...")
    
    try:
        # Use the official load_mcp_tools function
        mcp_tools = await setup_mcp_connection()
        
        print(f"✅ Loaded {len(mcp_tools)} tools")
      
        
        return {"mcp_tools": mcp_tools}
        
    except Exception as e:
        error_msg = f"Failed to load MCP tools: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {"error_message": error_msg}
    
async def optimize_query_node(state: ScrapeProcessState) -> dict:
    print("Step 1: Optimizing user query with Claude...")
    try:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. Rephrase the user's query to give context for a web scraper targeting service targeting 'https://www.nike.com/w/mens-shoes-nik1zy7ok'. Make the query more specific and actionable for web scraping. The bright data mcp should use the scrape_as_markdown tool to look for for the query"),
            ("human", "User query: {query}"),
        ])
        chain = prompt_template | llm
        result = await chain.ainvoke({"query": state.user_query})
        return {"optimized_query": result.content}
    except Exception as e:
        error_msg = f"Failed to optimize query: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error_message": error_msg}

async def scrape_with_tools_node(state: ScrapeProcessState) -> dict:
    print("🔍 Scraping with MCP tools...")
    try:
        if not state.mcp_tools:
            return {"error_message": "No MCP tools available for scraping"}

        agent = create_react_agent(llm, state.mcp_tools)
        query_to_use = state.optimized_query or state.user_query

        initial_messages = [
            HumanMessage(
                content=(
                    f"IMPORTANT: Use ONLY the scrape_as_markdown tool to scrape this EXACT URL: "
                    f"https://www.nike.com/w/mens-shoes-nik1zy7ok\n\n"
                    f"CRITICAL REQUIREMENTS:\n"
                    f"1. Do NOT use any search tools or scrape multiple pages\n"
                    f"2. Focus on extracting these elements for each product:\n"
                    f"   - Product name\n"
                    f"   - Price\n"
                    f"   - FULL product URL (look for href attributes containing '/t/' or full nike.com URLs)\n"
                    f"   - Image URLs if available\n\n"
                    f"3. When you find a product link, include the COMPLETE URL\n"
                    f"4. If you see partial URLs starting with '/t/', convert them to full URLs by adding 'https://www.nike.com'\n\n"
                    f"Query to match: {query_to_use}\n\n"
                    f"Return the complete scraped markdown content including all URLs and links found."
                )
            )
        ]

        response = await agent.ainvoke({"messages": initial_messages})

        if "messages" in response and response["messages"]:
            last_message = response["messages"][-1]
            scraped_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            scraped_content = "No response from agent"

        print(f"📄 Scraped content preview: {scraped_content[:200]}...")
        return {"scraped_data": scraped_content}

    except Exception as e:
        error_msg = f"Failed to scrape data: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {"error_message": error_msg}


async def parse_data_node(state: ScrapeProcessState) -> dict:
    """Parse the scraped data into structured products"""
    print("📊 Parsing scraped data...")
    
    if not state.scraped_data:
        return {"error_message": "No scraped data to parse"}
    
    try:
        # Use Claude to structure the data with enhanced URL detection
        parse_prompt = f"""Parse the following scraped Nike shoe data into a JSON array of products.

        CRITICAL INSTRUCTIONS FOR URL EXTRACTION:
        
        The scraped data appears to be in MARKDOWN format. Look carefully for:
        
        1. **Direct URLs**: Look for any complete URLs like:
           - https://www.nike.com/t/[product-name]/[product-code]
           - https://www.nike.com/t/[shoe-name]-[gender]-shoes-[identifier]/[style-code]
        
        2. **Markdown Links**: Look for markdown link format like:
           - [Product Name](https://www.nike.com/t/...)
           - [Link text](/t/product-url)
        
        3. **Embedded URLs**: Search the entire content for any strings containing:
           - "/t/" followed by product identifiers
           - "nike.com" URLs
           - Product codes like "HF1553-003", "FQ7860-010", "DR2615-107"
        
        4. **Link References**: Look for reference-style markdown links at the bottom of the content
        
        5. **Partial URLs**: If you find partial URLs like "/t/air-max-...", convert to full URLs with "https://www.nike.com"
        
        6. **NO URL GENERATION**: If NO actual URLs are found in the scraped data, use "https://www.nike.com" as fallback
        
        7. **Contextual Matching**: Try to match product names with any URLs found nearby in the text

        Each product should have:
        - name: Product name (exactly as it appears)
        - price: Price as string (e.g., "$120.00")
        - currency: "USD" 
        - availability: "In Stock", "Out of Stock", or "Limited" (determine from scraped data)
        - product_url: ACTUAL direct link extracted from scraped data (or https://www.nike.com if none found)
        - image_url: Product image URL (if available in scraped data)

        MARKDOWN URL EXAMPLES to look for:
        - [Nike Air Max DN8](https://www.nike.com/t/air-max-dn8-mens-shoes-YPsmAOxu/FQ7860-010)
        - https://www.nike.com/t/zoom-vomero-5-mens-shoes-MgsTqZ/HF1553-003
        - /t/invincible-3-mens-road-running-shoes-6MqQ72/DR2615-107

        IMPORTANT:
        - Scan the ENTIRE scraped content for ANY URLs or links
        - Pay special attention to markdown link syntax [text](url)
        - Look for URLs that might be split across lines
        - If the data is truly just text with no URLs, all products will fallback to https://www.nike.com
        - Only include products that have a clear name and price
        - Include ALL shoes mentioned in the data

        Return ONLY a JSON array, no other text.

        Scraped content:
        {state.scraped_data}
        """

        result = await llm.ainvoke([HumanMessage(content=parse_prompt)])

        print(f"📦 Parsing result: {result.content[:200]}...")
        
        # Clean and parse the JSON response
        response_content = result.content.strip()
        if "```json" in response_content:
            response_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            response_content = response_content.split("```")[1].strip()
        
        # Parse JSON and create ProductData objects
        import json
        parsed_data = json.loads(response_content)
        
        products = []
        for item in parsed_data:
            # Ensure all required fields are present
            product_data = {
                "name": item.get("name", "") or "",
                "price": item.get("price", "$0") or "$0",
                "currency": "USD",
                "availability": item.get("availability", "In Stock") or "In Stock",
                "product_url": item.get("product_url", "https://www.nike.com") or "https://www.nike.com",
                "image_url": item.get("image_url", "") or ""
            }
            
            # Only add products with valid names and prices
            if product_data["name"] and product_data["price"] != "$0":
                products.append(ProductData(**product_data))
        
        print(f"✅ Successfully parsed {len(products)} products")
        return {"parsed_products": products}
        
    except Exception as e:
        error_msg = f"Parsing failed: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {"error_message": error_msg}

# Fixed conditional edge functions
def should_continue_after_load_tools(state: ScrapeProcessState) -> str:
    """Check if we should continue after loading tools"""
    if hasattr(state, 'error_message') and state.error_message:
        return "END"
    return "scrape_with_tools"

def should_continue_after_scraping(state: ScrapeProcessState) -> str:
    """Check if we should continue after scraping"""
    if hasattr(state, 'error_message') and state.error_message:
        return "END"
    return "parse_data"

# Build the workflow
workflow = StateGraph(ScrapeProcessState)

workflow.add_node("load_tools", load_tools_node)
workflow.add_node("scrape_with_tools", scrape_with_tools_node)
workflow.add_node("parse_data", parse_data_node)

workflow.set_entry_point("load_tools")

# Fixed conditional edges
workflow.add_conditional_edges(
    "load_tools",
    should_continue_after_load_tools,
    {"scrape_with_tools": "scrape_with_tools", "END": END}
)

workflow.add_conditional_edges(
    "scrape_with_tools", 
    should_continue_after_scraping,
    {"parse_data": "parse_data", "END": END}
)

workflow.add_edge("parse_data", END)

# Compile the app
langgraph_app = workflow.compile()

# Usage
async def run_scraping(query: str):
    """Run the scraping workflow"""
    initial_state = ScrapeProcessState(user_query=query)
    result = await langgraph_app.ainvoke(initial_state)
    
    if hasattr(result, 'error_message') and result.error_message:
        print(f"❌ Error: {result.error_message}")
        return None
    else:
        print(f"✅ Found {len(result.parsed_products)} products")
        return result.parsed_products


@app.post("/search", response_model=SearchResponse)
async def search_nike_shoes(request: SearchRequest):
    print(f"Received request for query: {request.query}")
    
    initial_state = ScrapeProcessState(user_query=request.query)
    
    try:
        # Execute the async LangGraph workflow
        final_state = await langgraph_app.ainvoke(initial_state)
        
        # Check for errors first - use dictionary access
        if 'error_message' in final_state and final_state['error_message']:
            raise HTTPException(status_code=500, detail=f"Scraping or parsing failed: {final_state['error_message']}")

        # Access parsed_products from the final state - use dictionary access
        if 'parsed_products' in final_state and final_state['parsed_products']:
            products = final_state['parsed_products']  # Fixed: removed extra brackets
            print(f"Found {len(products)} products in final state")
            
            # Validate products format
            if products and isinstance(products, list):
                # Convert products to the expected format if needed
                formatted_products = [
                    ProductData(
                        name=p.name,
                        price=p.price,
                        currency=p.currency,
                        availability=p.availability,
                        product_url=p.product_url,
                        image_url=p.image_url
                    ) for p in products
                ]
                
                return SearchResponse(products=formatted_products)
            else:
                raise HTTPException(status_code=404, detail="No valid products found in the response")
        else:
            raise HTTPException(status_code=404, detail="No products data found in the response")

    except Exception as e:
        error_msg = f"Workflow execution error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=f"Internal server error: Something went wrong while processing your request. Please try again later.")