import asyncio
import weather_server

async def main():
    # Get the MCP server instance
    mcp_server = weather_server.mcp
    
    print("MCP Server:", mcp_server.name)
    print()
    
    # Get the list of available tools
    print("Getting list of available tools...")
    tools = await mcp_server.list_tools()
    
    print("Available tools:")
    print(tools)
    print()
    
    # Let's also try to call one of the tools
    print("Testing get_weather_tool for Berlin:")
    result = await mcp_server.call_tool("get_weather_tool", {"city": "Berlin"})
    print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main()) 