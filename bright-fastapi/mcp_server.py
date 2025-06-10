import asyncio
from typing import Tuple
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

anthropic = Anthropic()


async def connect_to_mcp() -> Tuple[AsyncExitStack, ClientSession]:
    """Connect to an MCP server and return session and context stack."""
    # is_python = server_script_path.endswith('.py')
    # is_js = server_script_path.endswith('.js')

    # if not (is_python or is_js):
    #     raise ValueError("Server script must be a .py or .js file")

    # command = "python" if is_python else "node"

    server_params = StdioServerParameters(
        command="npx",
        args= ["-y", "@brightdata/mcp"],
        env={
            "API_TOKEN": os.getenv("BRIGHT_DATA_API_KEY", ""),
            "WEB_UNLOCKER_ZONE": "mcp_unlocker"
        }
    )

    exit_stack = AsyncExitStack()

    # Establish stdio client and session
    stdio, write = await exit_stack.enter_async_context(stdio_client(server_params))
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))
    await session.initialize()

    response = await session.list_tools()
    print("✅ Connected to MCP with tools:", [tool.name for tool in response.tools])

    return exit_stack, session


async def process_query(session: ClientSession, query: str) -> str:
    """Send a query to the MCP server and return a response."""
    response = await session.run_tool(
        tool_name="scrape_as_markdown",  # <-- replace this
        input={"query": query}
    )
    return response.output

# async def get_mcp_tools

# async def get_mcp_server