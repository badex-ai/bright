# mcp_server.py
import os
import logging
import asyncio
from typing import Optional, List, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Correct imports based on actual documentation
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_core.tools import BaseTool
except ImportError:
    raise ImportError(
        "langchain-mcp-adapters not found. Install with: pip install langchain-mcp-adapters"
    )

logger = logging.getLogger(__name__)

# Global variables to store MCP client and tools
_mcp_client: Optional[MultiServerMCPClient] = None
_mcp_tools: List[BaseTool] = []

async def initialize_mcp_server():
    """
    Initialize MCP client using the correct langchain-mcp-adapters API.
    Returns the MultiServerMCPClient instance.
    """

    print(f"✅ intializing MCP server...")
    global _mcp_client, _mcp_tools
    
    if _mcp_client:
        logger.info("MCP client already initialized")
        return _mcp_client
    
    try:
        # Validate required environment variables
        bright_data_api_key = os.environ.get("BRIGHTDATA_API_TOKEN")

        print(f"✅ bright data API Key loaded: {bright_data_api_key}...")
        if not bright_data_api_key:
            raise ValueError("BRIGHT_DATA_API_KEY not found in environment variables")
        
        logger.info("Initializing MCP client with Bright Data server...")
        
        # Configuration for Bright Data MCP server
        connections = {
            "bright_data": {
                "command": "npx",
                "args": ["-y", "@brightdata/mcp"],
                "transport": "stdio",
                "env": {
                    "API_TOKEN": bright_data_api_key,
                    "WEB_UNLOCKER_ZONE": "mcp_unlocker",
                    "BROWSER_ZONE": "<optional browser zone name, defaults to mcp_browser>"
                }
            }
        }


        print(f"✅ connections state: {connections}...")
        # Initialize the MultiServerMCPClient with connections
        _mcp_client = MultiServerMCPClient(connections)

        print(f"✅ mcpclient___: {_mcp_client}...")

        
        # Load tools using the correct API
        logger.info("Loading MCP tools...")
        _mcp_tools = await _mcp_client.get_tools()
        
        logger.info(f"✅ MCP client initialized successfully with {len(_mcp_tools)} tools")
        logger.info(f"Available tools: {[tool.name for tool in _mcp_tools]}")
        
        return _mcp_client
        
    except Exception as e:
        logger.error(f"❌ Error initializing MCP client: {e}")
        _mcp_client = None
        _mcp_tools = []
        return None

async def get_mcp_tools() -> List[BaseTool]:
    """
    Get the loaded MCP tools as LangChain tools.
    Returns a list of BaseTool instances.
    """
    global _mcp_tools
    
    if not _mcp_tools and _mcp_client:
        logger.info("Refreshing MCP tools...")
        _mcp_tools = await _mcp_client.get_tools()
    elif not _mcp_client:
        logger.warning("No MCP client initialized. Attempting to initialize...")
        await initialize_mcp_server()
    
    return _mcp_tools

async def get_tool_by_name(tool_name: str) -> Optional[BaseTool]:
    """
    Get a specific MCP tool by name.
    
    Args:
        tool_name (str): Name of the tool to retrieve
        
    Returns:
        Optional[BaseTool]: The tool if found, None otherwise
    """
    tools = await get_mcp_tools()
    
    for tool in tools:
        if tool.name == tool_name:
            return tool
    
    logger.warning(f"Tool '{tool_name}' not found in available MCP tools")
    return None

async def list_available_tools() -> List[str]:
    """
    List all available MCP tool names.
    
    Returns:
        List[str]: List of tool names
    """
    tools = await get_mcp_tools()
    return [tool.name for tool in tools]

async def wait_for_initialization() -> bool:
    """
    Wait for MCP initialization to complete.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    client = await initialize_mcp_server()
    return client is not None

def get_mcp_server():
    """
    Get the current MCP client instance.
    
    Returns:
        Optional[MultiServerMCPClient]: The MCP client if initialized, None otherwise
    """
    return _mcp_client

def is_mcp_initialized() -> bool:
    """
    Check if MCP client is initialized.
    
    Returns:
        bool: True if initialized, False otherwise
    """
    return _mcp_client is not None

async def close_mcp_server():
    """
    Close the MCP client and clean up resources.
    Note: MultiServerMCPClient doesn't have an explicit close method,
    so we just reset our global variables.
    """
    global _mcp_client, _mcp_tools
    
    # Reset global variables
    _mcp_client = None
    _mcp_tools = []
    logger.info("✅ MCP client references cleared")

# Utility function for direct tool execution
async def execute_mcp_tool(tool_name: str, **kwargs) -> Any:
    """
    Execute a specific MCP tool with given arguments.
    
    Args:
        tool_name (str): Name of the tool to execute
        **kwargs: Arguments to pass to the tool
    
    Returns:
        Any: Result from tool execution
    """
    tool = await get_tool_by_name(tool_name)
    
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not found")
    
    try:
        # Execute the tool using LangChain's tool interface
        if hasattr(tool, 'ainvoke'):
            result = await tool.ainvoke(kwargs)
        else:
            # Fallback to synchronous invoke if ainvoke not available
            result = tool.invoke(kwargs)
        return result
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}")
        raise

# Context manager for MCP client lifecycle
class MCPClientManager:
    """Context manager for MCP client lifecycle management."""
    
    async def __aenter__(self):
        await initialize_mcp_server()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await close_mcp_server()
    
    def get_client(self):
        return get_mcp_server()
    
    async def get_tools(self):
        return await get_mcp_tools()

# Helper function to get tools for a specific server (if needed)
async def get_tools_from_server(server_name: str = "bright_data") -> List[BaseTool]:
    """
    Get tools from a specific MCP server.
    
    Args:
        server_name (str): Name of the server to get tools from
        
    Returns:
        List[BaseTool]: List of tools from the specified server
    """
    if not _mcp_client:
        await initialize_mcp_server()
    
    if _mcp_client:
        return await _mcp_client.get_tools(server_name=server_name)
    
    return []