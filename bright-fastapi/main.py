# Correct imports based on the official documentation
from mcp.client.stdio import stdio_client
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
from mcp_server2 import setup_mcp_connection
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
        for tool in mcp_tools:
            print(f"  - {tool.name}: {tool.description}")
        
        return {"mcp_tools": mcp_tools}
        
    except Exception as e:
        error_msg = f"Failed to load MCP tools: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {"error_message": error_msg}
    
async def optimize_query_node(state: ScrapeProcessState) -> dict:
    print("Step 1: Optimizing user query with Claude...")
    try:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. Rephrase the user's query to give context for a web scraper targeting 'Nike men's shoes'. Make the query more specific and actionable for web scraping."),
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
        # Check if we have tools loaded
        if not state.mcp_tools:
            return {"error_message": "No MCP tools available for scraping"}
        
        # Create agent with the loaded tools
        agent = create_react_agent(llm, state.mcp_tools)

        # Use the optimized query or original query
        query_to_use = state.optimized_query or state.user_query
        
        # Create initial message for the agent
        initial_messages = [HumanMessage(content=f"Search for Nike shoes: {query_to_use}")]
        
        # Run the agent
        response = await agent.ainvoke({"messages": initial_messages})

        # Extract the content from the final AI message
        if "messages" in response and response["messages"]:
            # Get the last message from the agent
            last_message = response["messages"][-1]
            # Extract content from the AI message
            if hasattr(last_message, 'content'):
                scraped_content = last_message.content
            else:
                scraped_content = str(last_message)
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
        # Use Claude to structure the data
        parse_prompt = f"""Parse the following scraped Nike shoe data into a JSON array of products.
        
    Each product should have:
    - name: Product name
    - price: Price as string
    - currency: "USD" 
    - availability: "In Stock", "Out of Stock", or "Limited"
    - product_url: Direct link to product page
    - image_url: Product image URL (if available)

    Return ONLY valid JSON array, no markdown or explanations.

    Scraped data:
    {state.scraped_data}"""

        result = await llm.ainvoke([HumanMessage(content=parse_prompt)])
        
        # Parse the JSON response
        import json
        response_content = result.content.strip()
        
        # Clean markdown if present
        if response_content.startswith("```json"):
            response_content = response_content[7:-3].strip()
        elif response_content.startswith("```"):
            # Handle other code block formats
            lines = response_content.split('\n')
            response_content = '\n'.join(lines[1:-1]).strip()
        
        try:
            parsed_data = json.loads(response_content)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the response
            import re
            json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
            else:
                raise ValueError("Could not extract valid JSON from response")
        
        # Convert to ProductData objects
        products = []
        for item in parsed_data:
            try:
                product = ProductData(**item)
                products.append(product)
            except Exception as e:
                print(f"⚠️ Failed to parse product {item}: {e}")
                continue
        
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
        
        # Check for errors properly
        if hasattr(final_state, 'error_message') and final_state.error_message:
            raise HTTPException(status_code=500, detail=f"Scraping or parsing failed: {final_state.error_message}")
        
        if not hasattr(final_state, 'parsed_products') or not final_state.parsed_products:
            raise HTTPException(status_code=404, detail="No products found. The page might not contain product data or the scraping failed.")

        return SearchResponse(products=final_state.parsed_products)
        
    except Exception as e:
        error_msg = f"Workflow execution error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")