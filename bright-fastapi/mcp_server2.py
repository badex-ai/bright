from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio
from typing import Tuple
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import traceback



async def setup_mcp_connection():
    """Set up MCP connection to Bright Data server"""

    bright_data_api_key = os.environ.get("BRIGHTDATA_API_TOKEN")

    
    try:
        print("🔧 Initializing MCP client...")
        
        # Check if required environment variables are set
        api_token = os.getenv("BRIGHTDATA_API_TOKEN")
        if not api_token:
            raise ValueError("BRIGHTDATA_API_TOKEN environment variable is not set")
        
        print(f"📡 Using API token: {api_token[:10]}..." if api_token else "❌ No API token found")
        
        client = MultiServerMCPClient(
            {
                "brightdata": {
                    "command": "npx",
                    # Replace with absolute path to your math_server.py file
                    "args": ["-y", "@brightdata/mcp"],
                    "env": {
                    "API_TOKEN": bright_data_api_key,
                    "WEB_UNLOCKER_ZONE": "mcp_unlocker",
                    },
                    "transport": "stdio",
                },
            }
        )
        
        print("🔌 Connecting to MCP server...")
        tools = await client.get_tools()
        
        print(f"✅ Successfully connected! Got {len(tools)} tools")
        
        # Return agent and tools for later use
        return tools
        
    except Exception as e:
        print(f"❌ MCP Connection Error: {str(e)}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        raise  # Re-raise the exception so it can be handled upstream