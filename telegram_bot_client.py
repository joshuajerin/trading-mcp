#!/usr/bin/env python3
"""
Telegram Bot Client

A simple client for connecting to the Telegram Bot API
"""
import os
import asyncio
import logging
from dotenv import load_dotenv
import httpx
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get bot token from environment variables
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Telegram Bot API base URL
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Store the last update_id we processed
last_update_id = 0

async def make_request(method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Make a request to the Telegram Bot API
    
    Args:
        method: Telegram API method
        params: Parameters for the method
        
    Returns:
        API response as a dictionary
    """
    if not BOT_TOKEN:
        raise ValueError("Bot token is not set. Please add TELEGRAM_BOT_TOKEN to your .env file.")
    
    url = f"{API_URL}/{method}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=params or {}, timeout=30.0)
            response.raise_for_status()
            result = response.json()
            
            if not result.get("ok"):
                error_msg = f"Telegram API error: {result.get('description', 'Unknown error')}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
            return result.get("result", {})
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error making request to Telegram API: {e}")
        raise

async def get_me() -> Dict[str, Any]:
    """Get information about the bot"""
    return await make_request("getMe")

async def send_message(
    chat_id: str, 
    text: str, 
    parse_mode: Optional[str] = "HTML",
    disable_web_page_preview: bool = False
) -> Dict[str, Any]:
    """
    Send a message to a chat
    
    Args:
        chat_id: Telegram chat ID
        text: Message text
        parse_mode: Parse mode (HTML, Markdown, MarkdownV2)
        disable_web_page_preview: Whether to disable link previews
        
    Returns:
        API response
    """
    params = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": disable_web_page_preview
    }
    
    if parse_mode:
        params["parse_mode"] = parse_mode
        
    return await make_request("sendMessage", params)

async def get_updates(offset: int = 0, timeout: int = 60) -> List[Dict[str, Any]]:
    """
    Get updates from Telegram
    
    Args:
        offset: Update ID to start from
        timeout: Timeout for long polling
        
    Returns:
        List of updates
    """
    params = {
        "offset": offset,
        "timeout": timeout,
        "allowed_updates": ["message", "callback_query"]
    }
    
    return await make_request("getUpdates", params)

async def process_updates():
    """Process incoming updates using long polling"""
    global last_update_id
    
    try:
        # Get updates with a 60-second timeout
        updates = await get_updates(last_update_id + 1, 60)
        
        for update in updates:
            # Update the last_update_id
            update_id = update.get("update_id", 0)
            if update_id > last_update_id:
                last_update_id = update_id
            
            # Process the message
            message = update.get("message")
            if message:
                await process_message(message)
                
            # Process callback queries (button clicks)
            callback_query = update.get("callback_query")
            if callback_query:
                await process_callback_query(callback_query)
    except Exception as e:
        logger.error(f"Error processing updates: {e}")
        # Sleep briefly before trying again
        await asyncio.sleep(1)

async def process_message(message: Dict[str, Any]):
    """
    Process an incoming message
    
    Args:
        message: Message data from Telegram
    """
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")
    user = message.get("from", {})
    
    if not chat_id:
        return
    
    logger.info(f"Received message from {user.get('username', 'Unknown')}: {text}")
    
    # Handle commands
    if text.startswith("/"):
        await handle_command(chat_id, text, user)
    elif text.upper().startswith("EXECUTE"):
        # This looks like a response to a trading recommendation
        try:
            # Import the process_trading_response function
            # We use a dynamic import to avoid circular imports
            import importlib
            server_module = importlib.import_module("server")
            
            # Process the response
            logger.info(f"Processing trading response: {text}")
            await send_message(chat_id, f"Processing your trading request: <i>{text}</i>\nPlease wait...")
            
            # Call the processing function
            result = await server_module.process_trading_response(str(chat_id), text)
            
            # If the result contains a status error, notify the user
            if result.get("status") == "error":
                await send_message(chat_id, f"<b>⚠️ Error processing your trading request:</b>\n{result.get('message', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error processing trading response: {e}")
            await send_message(chat_id, f"<b>⚠️ Error processing your trading request:</b>\n{str(e)}")
    else:
        # Echo the message for now
        await send_message(chat_id, f"You said: {text}")

async def process_callback_query(callback_query: Dict[str, Any]):
    """
    Process a callback query (button click)
    
    Args:
        callback_query: Callback query data from Telegram
    """
    query_id = callback_query.get("id")
    chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
    data = callback_query.get("data", "")
    
    logger.info(f"Received callback query: {data}")
    
    # Answer the callback query to remove the loading indicator
    await make_request("answerCallbackQuery", {"callback_query_id": query_id})
    
    # Process the button click
    if chat_id:
        await send_message(chat_id, f"You clicked: {data}")

async def handle_command(chat_id: str, command: str, user: Dict[str, Any]):
    """
    Handle a command
    
    Args:
        chat_id: Chat ID
        command: Command text
        user: User data
    """
    command = command.lower()
    
    if command.startswith("/start"):
        await send_message(
            chat_id,
            f"👋 Hello, {user.get('first_name', 'there')}!\n\n"
            f"I'm your Robinhood Trading Bot. Use /help to see available commands."
        )
    elif command.startswith("/help"):
        await send_message(
            chat_id,
            "Available commands:\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/trade - Run autonomous trading analysis\n"
            "/echo <text> - Echo back your text\n"
        )
    elif command.startswith("/echo "):
        text = command[6:]  # Remove "/echo " prefix
        await send_message(chat_id, text)
    elif command.startswith("/trade"):
        # Run the autonomous trading analysis
        try:
            # Import the run_autonomously function
            import importlib
            server_module = importlib.import_module("server")
            
            # Notify the user
            await send_message(chat_id, "<b>🤖 Starting autonomous trading analysis...</b>\n\nAnalyzing your watchlist and generating recommendations. This may take a moment.")
            
            # Run the analysis - this will generate recommendations but NOT execute any trades
            # The user must respond with an EXECUTE command to initiate trades
            result = await server_module.run_autonomously(str(chat_id))
            
            if result.get("status") != "success":
                await send_message(chat_id, f"<b>⚠️ Error running analysis:</b>\n{result.get('message', 'Unknown error')}")
            else:
                # Inform the user that trades have been executed automatically
                await send_message(chat_id, "<b>ℹ️ Trades executed:</b>\nAll recommended trades have been executed automatically. See the execution report above for details.")
                
        except Exception as e:
            logger.error(f"Error running autonomous analysis: {e}")
            await send_message(chat_id, f"<b>⚠️ Error:</b>\n{str(e)}")
    else:
        await send_message(
            chat_id,
            "Unknown command. Use /help to see available commands."
        )

async def main():
    """Main function to start the bot"""
    try:
        # Check if the bot token is set
        if not BOT_TOKEN:
            logger.error("Bot token is not set. Please add TELEGRAM_BOT_TOKEN to your .env file.")
            return
        
        # Get bot information
        bot_info = await get_me()
        logger.info(f"Started bot: @{bot_info.get('username')}")
        
        print(f"Bot started successfully! @{bot_info.get('username')}")
        print("Send a message to your bot on Telegram to test it.")
        print("Press Ctrl+C to stop the bot.")
        
        # Run the update loop
        while True:
            await process_updates()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    asyncio.run(main())