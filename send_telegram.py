#!/usr/bin/env python3
"""
Send Telegram Message via MCP

A utility script to send a message to a Telegram chat using the MCP server.
Usage: python send_telegram.py CHAT_ID "Your message here"
"""
import os
import sys
import json
import base64
import asyncio
import mcp
from mcp.client.streamable_http import streamablehttp_client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or "7572235558:AAFKJOlMEEdVfSK9gqCyQp0jBgcIdPVcVdE"
SMITHERY_API_KEY = os.getenv("SMITHERY_API_KEY")

async def send_message(chat_id, message_text):
    """Send a message via the Telegram MCP server"""
    # Telegram config with bot token
    telegram_config = {
        "telegramBotToken": BOT_TOKEN
    }
    
    # Encode config in base64
    config_b64 = base64.b64encode(json.dumps(telegram_config).encode()).decode()
    
    # Create server URL
    url = f"https://server.smithery.ai/@NexusX-MCP/telegram-mcp-server/mcp?config={config_b64}&api_key={SMITHERY_API_KEY}"
    
    print(f"Connecting to Telegram MCP server...")
    
    try:
        # Connect with timeout
        async with asyncio.timeout(20):
            async with streamablehttp_client(url) as (read_stream, write_stream, _):
                async with mcp.ClientSession(read_stream, write_stream) as session:
                    # Initialize the connection
                    await session.initialize()
                    
                    # Check if send_message tool is available
                    tools_result = await session.list_tools()
                    available_tools = [t.name for t in tools_result.tools]
                    
                    if "send_message" in available_tools:
                        print(f"Sending message to {chat_id}...")
                        result = await session.call_tool(
                            "send_message",
                            {
                                "entity": chat_id,
                                "text": message_text
                            }
                        )
                        print("Message sent successfully!")
                        return True
                    else:
                        print("Error: send_message tool is not available")
                        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

async def main():
    # Check if enough arguments are provided
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} CHAT_ID \"Your message here\"")
        print("\nTo find your chat ID:")
        print("1. Add the bot to a chat or start a conversation with it")
        print("2. Use the telegram_mcp_client.py script to search for available chats")
        print("3. Or use Telegram's @userinfobot to find your user ID")
        return False
    
    # Get chat ID and message from command line arguments
    chat_id = sys.argv[1]
    message = sys.argv[2]
    
    # Send the message
    success = await send_message(chat_id, message)
    return success

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())