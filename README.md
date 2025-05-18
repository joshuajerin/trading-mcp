# Robinhood MCP & Telegram MCP Client

This project provides:
- An **MCP server** for Robinhood trading (Python, FastMCP)
- A **Telegram MCP client** to connect to Telegram via Smithery MCP bridge

## Setup

1. **Clone the repo**

2. **Install dependencies**
```bash
pip install -r requirements.txt
# or
pip install "mcp[cli]" python-dotenv robin_stocks
```

3. **.env setup**
Create a `.env` file with:
```
TELEGRAM_API_ID=your_telegram_api_id
TELEGRAM_API_HASH=your_telegram_api_hash
SMITHERY_API_KEY=your_smithery_api_key
```

## Running

### Start the Robinhood MCP server
```bash
python server.py
```

### Run the Telegram MCP client
```bash
python telegram_mcp_client.py
```

- The Telegram client will list available tools and let you send messages via MCP.
- Do **not** use a Telegram Bot API token for the MCP client—use your API ID and Hash from [my.telegram.org](https://my.telegram.org).

---
**Tip:**
- For Robinhood MCP, edit `mcp_tools.py` and `mcp_prompts.py` to customize tools and prompts.
- For Telegram MCP, edit `telegram_mcp_client.py` to try different tool calls.
