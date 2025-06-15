import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("OPENAI_API_KEY is not set")
# else:
#     print("OPENAI_API_KEY is set: ", api_key)

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI()
        self.openai.api_key = os.getenv("OPENAI_API_KEY")
    

    async def connect_to_mcp_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")

        if not is_python and not is_js:
            raise ValueError("Invalid server script path. Must end with .py or .js")
        
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        print("Available tools:")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available tools"""

        messages = [
            {
                "role": "user",
                "content": query,
            }
        ]
        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # Initialize OpenAI API call
        openai_response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            tools=available_tools,
        )

        # Process response and handle tool calls
        final_text = []
        
        # Process the initial response
        response_message = openai_response.choices[0].message
        
        # Handle the response content and tool calls
        max_iterations = 3
        for _ in range(max_iterations):
            # Add text content if present
            if response_message.content:
                messages.append({
                    "role": "system",
                    "content": response_message.content
                })
                # print(f"\nSystem message: {response_message.content}")
            
            # Check if there are tool calls to execute
            if not response_message.tool_calls:
                # No more tool calls, we're done
                # print("No more tool calls, we're done")
                final_text.append(response_message.content)

                break
            
            observations = {}

            # Execute each tool call
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_arguments = json.loads(tool_call.function.arguments)
                
                # final_text.append(f"[Calling tool {tool_name} with args {tool_arguments}]")
                
                print(f"Calling tool {tool_name} with args {tool_arguments}")
                # Execute tool call via MCP
                result = await self.session.call_tool(tool_name, tool_arguments)
                observations[tool_call.id] = result.content
                # Add tool result to conversation
                # print(f"\nObservation: {observations}")
                messages.append({
                    "role": "user",
                    "content": f'f"Observation: {observations}"'
                })
            
            # Get next response from OpenAI with tool results
            openai_response = self.openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                tools=available_tools,
            )

            response_message = openai_response.choices[0].message
            # print(f"\nResponse message: {response_message}")

        return "\n".join(final_text)

    async def cleanup(self):
        """Clean up resources"""
        if self.exit_stack:
            await self.exit_stack.aclose()


    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_mcp_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())

        
