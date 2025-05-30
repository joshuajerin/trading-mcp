#!/usr/bin/env python3
"""
Robinhood MCP Server

A FastMCP server that exposes Robinhood trading functionality 
to Claude and other LLM clients.

## IMPORTANT USAGE GUIDELINES FOR LLM AGENTS

To efficiently use this server, follow these best practices:

1. TOKEN EFFICIENCY: Financial data contains large amounts of text that can consume tokens rapidly.
   - DO NOT request full raw data dumps (like complete order histories)
   - USE targeted queries (specific dates, specific stocks) 
   - PREFER summary tools over raw data tools whenever possible

2. STEPWISE APPROACH: 
   - First check if a tool works with a small sample (e.g., a single stock or single date)
   - Then build on successful results for deeper analysis
   - Break complex analyses into smaller focused queries

3. CODE-FIRST APPROACH:
   - When analyzing trading data, write code to process the data rather than dumping all raw data
   - Use client-side code to aggregate and analyze data

4. PRIVATE DATA HANDLING:
   - Summarize financial findings rather than displaying full transaction details
   - Focus on trends, patterns, and metrics rather than specific trade IDs or exact timestamps

Following these guidelines will result in faster, more reliable responses and better user experience.
"""
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from fastmcp import FastMCP, Context, Image
import robin_stocks.robinhood as rh

# Initialize the MCP server
mcp = FastMCP(
    "robinhood", 
    dependencies=["robin_stocks", "pydantic"],
    description="A server that provides stock trading functionality through Robinhood"
)

# Global variable to store login state
login_state = {
    "mfa_required": False,
    "username": None,
    "password": None,
    "challenge": None
}

# ----- Models for request/response data -----

class StockOrder(BaseModel):
    """Model for placing a stock order"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL')")
    quantity: int = Field(..., description="Number of shares to trade")
    price: Optional[float] = Field(None, description="Limit price (if applicable)")
    order_type: str = Field("market", description="Order type: 'market' or 'limit'")
    time_in_force: str = Field("gtc", description="Time in force: 'gtc' (good till canceled), 'gfd' (good for day), etc.")
    extended_hours: bool = Field(False, description="Whether to allow trading during extended hours")

class LimitOrder(BaseModel):
    """Model for placing a limit order"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL')")
    quantity: int = Field(..., description="Number of shares to trade")
    price: float = Field(..., description="Limit price for the order")
    time_in_force: str = Field("gtc", description="Time in force: 'gtc' (good till canceled), 'gfd' (good for day)")
    extended_hours: bool = Field(False, description="Whether to allow trading during extended hours")

class LoginCredentials(BaseModel):
    """Model for login credentials"""
    username: str = Field(..., description="Robinhood username (email)")
    password: str = Field(..., description="Robinhood password")
    mfa_code: Optional[str] = Field(None, description="MFA code if required")

class InitialLoginCredentials(BaseModel):
    """Model for initial login without MFA"""
    username: str = Field(..., description="Robinhood username (email)")
    password: str = Field(..., description="Robinhood password")

class MfaCredentials(BaseModel):
    """Model for MFA code submission"""
    mfa_code: str = Field(..., description="MFA code from authenticator app")

class StockInfo(BaseModel):
    """Model for stock information"""
    ticker: str = Field(..., description="Stock ticker symbol")

# ----- Authentication -----

@mcp.tool()
async def initiate_login(credentials: InitialLoginCredentials) -> Dict[str, Any]:
    """
    Initiate login to Robinhood with username and password.
    
    This is the first step of the login process. If MFA is required,
    the response will indicate this and you can then call submit_mfa_code.
    """
    try:
        # Store credentials for MFA step
        login_state["username"] = credentials.username
        login_state["password"] = credentials.password
        login_state["mfa_required"] = False
        login_state["challenge"] = None
        
        # Attempt login without MFA first
        login_response = rh.login(
            username=credentials.username,
            password=credentials.password,
            mfa_code=None
        )
        
        # If login succeeded without MFA
        if login_response and "access_token" in login_response:
            # Clear stored credentials since login completed
            login_state["username"] = None
            login_state["password"] = None
            return {
                "success": True,
                "message": "Successfully logged in to Robinhood",
                "expires_in": login_response.get("expires_in", 86400),
                "scope": login_response.get("scope", "internal"),
                "mfa_required": False
            }
        else:
            # Login failed, likely needs MFA
            login_state["mfa_required"] = True
            return {
                "success": False,
                "message": "MFA code required. Please call submit_mfa_code with your MFA code.",
                "mfa_required": True
            }
            
    except Exception as e:
        error_msg = str(e)
        
        # Check if the error indicates MFA is required
        if "mfa" in error_msg.lower() or "two" in error_msg.lower() or "factor" in error_msg.lower():
            login_state["mfa_required"] = True
            return {
                "success": False,
                "message": "MFA code required. Please call submit_mfa_code with your MFA code.",
                "mfa_required": True
            }
        else:
            # Clear stored credentials on actual login failure
            login_state["username"] = None
            login_state["password"] = None
            return {
                "success": False,
                "message": f"Login failed: {error_msg}",
                "mfa_required": False
            }

@mcp.tool()
async def submit_mfa_code(credentials: MfaCredentials) -> Dict[str, Any]:
    """
    Submit MFA code to complete the login process.
    
    This should be called after initiate_login returns mfa_required: true.
    """
    try:
        # Check if we're in the right state for MFA submission
        if not login_state["mfa_required"] or not login_state["username"] or not login_state["password"]:
            return {
                "success": False,
                "message": "No pending MFA login found. Please call initiate_login first."
            }
        
        # Attempt login with stored credentials and provided MFA code
        login_response = rh.login(
            username=login_state["username"],
            password=login_state["password"],
            mfa_code=credentials.mfa_code
        )
        
        # Clear stored credentials regardless of outcome
        login_state["username"] = None
        login_state["password"] = None
        login_state["mfa_required"] = False
        login_state["challenge"] = None
        
        # Check if login succeeded
        if login_response and "access_token" in login_response:
            return {
                "success": True,
                "message": "Successfully logged in to Robinhood with MFA",
                "expires_in": login_response.get("expires_in", 86400),
                "scope": login_response.get("scope", "internal")
            }
        else:
            return {
                "success": False,
                "message": "Login failed with provided MFA code. Please try again."
            }
            
    except Exception as e:
        # Clear stored credentials on error
        login_state["username"] = None
        login_state["password"] = None
        login_state["mfa_required"] = False
        login_state["challenge"] = None
        
        return {
            "success": False,
            "message": f"MFA login failed: {str(e)}"
        }

@mcp.tool()
async def login(credentials: LoginCredentials) -> Dict[str, Any]:
    """
    Login to Robinhood with the provided credentials (legacy method).
    
    This tool logs in to Robinhood and enables other trading functionalities.
    It stores the authentication token for subsequent requests.
    
    Note: For better MFA handling, use initiate_login followed by submit_mfa_code.
    """
    try:
        login_response = rh.login(
            username=credentials.username,
            password=credentials.password,
            mfa_code=credentials.mfa_code
        )
        
        # Return a sanitized version of the response
        return {
            "success": True,
            "message": "Successfully logged in to Robinhood",
            "expires_in": login_response.get("expires_in", 86400),
            "scope": login_response.get("scope", "internal")
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Login failed: {str(e)}"
        }

@mcp.tool()
async def logout() -> Dict[str, str]:
    """
    Logout from Robinhood and invalidate the current session.
    """
    try:
        rh.logout()
        return {"status": "success", "message": "Successfully logged out from Robinhood"}
    except Exception as e:
        return {"status": "error", "message": f"Logout failed: {str(e)}"}

# ----- Stock Information -----

@mcp.tool()
async def get_stock_quote(stock_info: StockInfo) -> Dict[str, Any]:
    """
    Get the latest quote information for a stock.
    
    Returns latest price, bid/ask, volume, and other quote information.
    """
    try:
        ticker = stock_info.ticker.upper()
        quote_info = rh.stocks.get_quotes(ticker)
        
        if not quote_info or isinstance(quote_info, list) and not quote_info:
            return {"status": "error", "message": f"No quote data found for {ticker}"}
            
        if isinstance(quote_info, list):
            quote_info = quote_info[0]
            
        # Clean up the response to include just the most relevant information
        return {
            "status": "success",
            "ticker": ticker,
            "ask_price": float(quote_info.get("ask_price", 0)),
            "bid_price": float(quote_info.get("bid_price", 0)),
            "last_trade_price": float(quote_info.get("last_trade_price", 0)),
            "previous_close": float(quote_info.get("previous_close", 0)),
            "updated_at": quote_info.get("updated_at", ""),
            "volume": int(float(quote_info.get("volume", 0)))
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get quote for {stock_info.ticker}: {str(e)}"}

@mcp.tool()
async def get_latest_price(stock_info: StockInfo) -> Dict[str, Any]:
    """
    Get the latest price for a stock.
    
    Returns a simple response with just the latest price.
    """
    try:
        ticker = stock_info.ticker.upper()
        price = rh.stocks.get_latest_price(ticker)
        
        if not price or not price[0]:
            return {"status": "error", "message": f"No price data found for {ticker}"}
            
        return {
            "status": "success",
            "ticker": ticker,
            "price": float(price[0])
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get price for {ticker}: {str(e)}"}

# ----- Trading Operations -----

@mcp.tool()
async def buy_stock_market_order(order: StockOrder) -> Dict[str, Any]:
    """
    Place a market order to buy a stock.
    
    Places an order to buy the specified quantity of a stock at the current market price.
    """
    try:
        result = rh.orders.order_buy_market(
            symbol=order.ticker,
            quantity=order.quantity,
            timeInForce=order.time_in_force,
            extendedHours=order.extended_hours
        )
        
        return {
            "status": "success",
            "order_id": result.get("id", ""),
            "state": result.get("state", ""),
            "ticker": order.ticker,
            "quantity": order.quantity,
            "type": "market",
            "side": "buy",
            "created_at": result.get("created_at", "")
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to place buy order: {str(e)}"}

@mcp.tool()
async def sell_stock_market_order(order: StockOrder) -> Dict[str, Any]:
    """
    Place a market order to sell a stock.
    
    Places an order to sell the specified quantity of a stock at the current market price.
    """
    try:
        result = rh.orders.order_sell_market(
            symbol=order.ticker,
            quantity=order.quantity,
            timeInForce=order.time_in_force,
            extendedHours=order.extended_hours
        )
        
        return {
            "status": "success",
            "order_id": result.get("id", ""),
            "state": result.get("state", ""),
            "ticker": order.ticker,
            "quantity": order.quantity,
            "type": "market",
            "side": "sell",
            "created_at": result.get("created_at", "")
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to place sell order: {str(e)}"}

@mcp.tool()
async def buy_stock_limit_order(order: LimitOrder) -> Dict[str, Any]:
    """
    Place a limit order to buy a stock.
    
    Places an order to buy the specified quantity of a stock at or below the specified limit price.
    """
    try:
        result = rh.orders.order_buy_limit(
            symbol=order.ticker,
            quantity=order.quantity,
            limitPrice=order.price,
            timeInForce=order.time_in_force,
            extendedHours=order.extended_hours
        )
        
        return {
            "status": "success",
            "order_id": result.get("id", ""),
            "state": result.get("state", ""),
            "ticker": order.ticker,
            "quantity": order.quantity,
            "limit_price": order.price,
            "type": "limit",
            "side": "buy",
            "created_at": result.get("created_at", "")
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to place buy limit order: {str(e)}"}

@mcp.tool()
async def sell_stock_limit_order(order: LimitOrder) -> Dict[str, Any]:
    """
    Place a limit order to sell a stock.
    
    Places an order to sell the specified quantity of a stock at or above the specified limit price.
    """
    try:
        result = rh.orders.order_sell_limit(
            symbol=order.ticker,
            quantity=order.quantity,
            limitPrice=order.price,
            timeInForce=order.time_in_force,
            extendedHours=order.extended_hours
        )
        
        return {
            "status": "success",
            "order_id": result.get("id", ""),
            "state": result.get("state", ""),
            "ticker": order.ticker,
            "quantity": order.quantity,
            "limit_price": order.price,
            "type": "limit",
            "side": "sell",
            "created_at": result.get("created_at", "")
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to place sell limit order: {str(e)}"}

@mcp.tool()
async def cancel_order(order_id: str) -> Dict[str, Any]:
    """
    Cancel an open order by order ID.
    
    Cancels an open order that hasn't been executed yet.
    """
    try:
        result = rh.orders.cancel_stock_order(order_id)
        return {
            "status": "success" if result else "error",
            "order_id": order_id,
            "message": "Order cancelled successfully" if result else "Failed to cancel order"
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to cancel order: {str(e)}"}

# ----- Portfolio Information -----

@mcp.tool()
async def get_portfolio() -> Dict[str, Any]:
    """
    Get portfolio information including equity value, cash balance, and other account details.
    """
    try:
        portfolio = rh.account.build_portfolio()
        return {
            "status": "success",
            "equity": float(portfolio.get("equity", 0)),
            "extended_hours_equity": float(portfolio.get("extended_hours_equity", 0)),
            "cash": float(portfolio.get("cash", 0)),
            "dividend_total": float(portfolio.get("dividend_total", 0))
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get portfolio: {str(e)}"}

@mcp.tool()
async def get_positions() -> Dict[str, Any]:
    """
    Get current positions in the portfolio.
    
    Returns all stocks currently held in the account, with quantity and cost basis.
    """
    try:
        positions = rh.account.get_open_stock_positions()
        formatted_positions = []
        
        for position in positions:
            instrument_data = rh.stocks.get_instrument_by_url(position.get("instrument", ""))
            ticker = instrument_data.get("symbol", "UNKNOWN")
            quantity = float(position.get("quantity", 0))
            average_buy_price = float(position.get("average_buy_price", 0))
            
            formatted_positions.append({
                "ticker": ticker,
                "quantity": quantity,
                "average_buy_price": average_buy_price,
                "cost_basis": quantity * average_buy_price
            })
        
        return {
            "status": "success",
            "positions": formatted_positions
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get positions: {str(e)}"}

@mcp.tool()
async def get_open_orders() -> Dict[str, Any]:
    """
    Get all open orders.
    
    Returns all orders that are currently open (e.g., unfilled limit orders).
    """
    try:
        orders = rh.orders.get_all_open_stock_orders()
        formatted_orders = []
        
        for order in orders:
            # Parse the instrument URL to get the ticker
            instrument_data = rh.stocks.get_instrument_by_url(order.get("instrument", ""))
            ticker = instrument_data.get("symbol", "UNKNOWN")
            
            formatted_orders.append({
                "order_id": order.get("id", ""),
                "ticker": ticker,
                "side": order.get("side", ""),
                "quantity": float(order.get("quantity", 0)),
                "type": order.get("type", ""),
                "price": float(order.get("price", 0)) if order.get("price") else None,
                "created_at": order.get("created_at", ""),
                "state": order.get("state", "")
            })
        
        return {
            "status": "success",
            "orders": formatted_orders
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get open orders: {str(e)}"}

@mcp.tool()
async def get_orders_by_date(date: str) -> Dict[str, Any]:
    """
    Get all orders placed on a specific date.
    
    Returns all orders (open, filled, canceled, etc.) created on the specified date.
    The date must be in YYYY-MM-DD format (e.g., '2025-05-02').
    """
    try:
        # Get all orders from the Robinhood API
        all_orders = rh.orders.get_all_stock_orders()
        
        # Filter orders by the specified date
        filtered_orders = []
        for order in all_orders:
            if order.get('created_at', '').startswith(date):
                # Parse the instrument URL to get the ticker
                instrument_data = rh.stocks.get_instrument_by_url(order.get("instrument", ""))
                ticker = instrument_data.get("symbol", "UNKNOWN")
                
                # Format the order data
                formatted_order = {
                    "order_id": order.get("id", ""),
                    "ticker": ticker,
                    "side": order.get("side", ""),
                    "quantity": float(order.get("quantity", 0)),
                    "type": order.get("type", ""),
                    "price": float(order.get("price", 0)) if order.get("price") else None,
                    "created_at": order.get("created_at", ""),
                    "state": order.get("state", ""),
                    "executions": order.get("executions", []),
                    "filled_quantity": float(order.get("cumulative_quantity", 0)),
                    "average_price": float(order.get("average_price", 0)) if order.get("average_price") else None
                }
                
                # Convert created_at time from UTC to Eastern Time
                if "created_at" in order and order["created_at"]:
                    # Parse the ISO timestamp
                    utc_time = order["created_at"].replace('Z', '+00:00')
                    # Calculate ET (UTC-4 during daylight saving time)
                    formatted_order["created_at_et"] = f"{utc_time[0:19]}Z (ET: {utc_time[11:16]} ET)"
                
                filtered_orders.append(formatted_order)
        
        # Sort orders by created_at timestamp (newest first)
        filtered_orders.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return {
            "status": "success",
            "date": date,
            "orders_count": len(filtered_orders),
            "orders": filtered_orders
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get orders for date {date}: {str(e)}"}

# ----- Market Data Resources -----

@mcp.resource("robinhood://stocks/{ticker}/info")
async def get_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Get detailed information about a stock.
    """
    try:
        ticker = ticker.upper()
        stock_info = rh.stocks.get_fundamentals(ticker)[0]
        
        return {
            "ticker": ticker,
            "name": stock_info.get("high_52_weeks", ""),
            "sector": stock_info.get("sector", ""),
            "industry": stock_info.get("industry", ""),
            "market_cap": float(stock_info.get("market_cap", 0)),
            "pe_ratio": float(stock_info.get("pe_ratio", 0)),
            "dividend_yield": float(stock_info.get("dividend_yield", 0)),
            "high_52_weeks": float(stock_info.get("high_52_weeks", 0)),
            "low_52_weeks": float(stock_info.get("low_52_weeks", 0))
        }
    except Exception as e:
        return {"error": f"Failed to get stock info for {ticker}: {str(e)}"}

@mcp.resource("robinhood://portfolio/summary")
async def get_portfolio_summary() -> Dict[str, Any]:
    """
    Get a summary of the portfolio's performance and composition.
    """
    try:
        portfolio = rh.account.build_portfolio()
        positions = rh.account.get_open_stock_positions()
        
        # Count number of different stocks
        positions_count = len(positions)
        
        return {
            "equity": float(portfolio.get("equity", 0)),
            "cash": float(portfolio.get("cash", 0)),
            "total_assets": float(portfolio.get("equity", 0)) + float(portfolio.get("cash", 0)),
            "positions_count": positions_count,
            "dividend_total": float(portfolio.get("dividend_total", 0))
        }
    except Exception as e:
        return {"error": f"Failed to get portfolio summary: {str(e)}"}

@mcp.resource("robinhood://account/history/{timespan}")
async def get_account_history(timespan: str) -> Dict[str, Any]:
    """
    Get account history for a specified timespan.
    
    The timespan parameter can be: day, week, month, 3month, year, 5year, all
    """
    try:
        valid_timespans = ["day", "week", "month", "3month", "year", "5year", "all"]
        if timespan not in valid_timespans:
            return {"error": f"Invalid timespan: {timespan}. Must be one of {', '.join(valid_timespans)}"}
            
        history = rh.account.get_historical_portfolio(interval=timespan)
        
        # Extract the key information from the history
        equity_data = []
        for data_point in history.get("equity_historicals", []):
            equity_data.append({
                "date": data_point.get("begins_at", ""),
                "equity": float(data_point.get("equity_close", 0)),
                "adjusted_equity": float(data_point.get("adjusted_equity_close", 0)),
            })
        
        return {
            "timespan": timespan,
            "equity_data": equity_data,
            "start_date": equity_data[0].get("date", "") if equity_data else "",
            "end_date": equity_data[-1].get("date", "") if equity_data else "",
            "total_return_percentage": history.get("total_return", {}).get("percentage", 0)
        }
    except Exception as e:
        return {"error": f"Failed to get account history: {str(e)}"}


# Helper function to get ticker from instrument URL
def get_ticker_from_instrument(instrument_url: str) -> str:
    """Get ticker symbol from instrument URL."""
    try:
        if not instrument_url:
            return "UNKNOWN"
        instrument_data = rh.stocks.get_instrument_by_url(instrument_url)
        return instrument_data.get("symbol", "UNKNOWN")
    except:
        return "UNKNOWN"


@mcp.tool()
async def analyze_trading_profit(date: str) -> Dict[str, Any]:
    """
    Calculate profit/loss from day trading on a specific date.
    
    Analyzes all trades made on the specified date using closest-price matching algorithm,
    which pairs buy/sell orders based on price similarity for more accurate profit calculation.
    """
    try:
        # Get orders for the specified date
        all_orders = rh.orders.get_all_stock_orders()
        filtered_orders = [o for o in all_orders if o.get('created_at', '').startswith(date)]

        # Filter by filled status (completed trades)
        filled_orders = [o for o in filtered_orders if o.get('state') == 'filled']

        # Group by ticker
        ticker_groups = {}
        for order in filled_orders:
            ticker = get_ticker_from_instrument(order.get("instrument", ""))
            if ticker not in ticker_groups:
                ticker_groups[ticker] = {"buys": [], "sells": []}

            # Convert to structured format for matching algorithm
            try:
                price = float(order.get('average_price', 0))
                quantity = float(order.get('quantity', 0))

                # Process executions to extract fees and timestamps
                executions = order.get('executions', [])
                fees = sum([float(e.get('fees', 0)) for e in executions])

                trade_record = {
                    'id': order.get('id'),
                    'price': price,
                    'quantity': quantity,
                    'remaining_qty': quantity,  # For tracking matched portions
                    'fees': fees,
                    'timestamp': order.get('last_transaction_at'),
                    'created_at': order.get('created_at')
                }

                # Add to appropriate category
                if order.get('side') == 'buy':
                    ticker_groups[ticker]["buys"].append(trade_record)
                else:
                    ticker_groups[ticker]["sells"].append(trade_record)
            except (ValueError, TypeError):
                # Skip if conversion fails
                continue

        # Process each ticker with the closest price matching algorithm
        results = []
        total_profit = 0
        total_matched_trades = 0

        for ticker, trades in ticker_groups.items():
            buys = trades["buys"]
            sells = trades["sells"]

            # Sort buys by timestamp (earliest first)
            buys.sort(key=lambda x: x['created_at'])

            # Sort sells by price (highest first to maximize profit)
            sells.sort(key=lambda x: x['price'], reverse=True)

            # Match trades using closest price approach
            matched_pairs = []

            for buy in buys:
                buy_qty = buy['remaining_qty']
                buy_price = buy['price']

                # Continue matching until this buy is fully matched or no more sells
                while buy_qty > 0 and any(s['remaining_qty'] > 0 for s in sells):
                    # Calculate price difference for each available sell
                    price_diffs = []

                    for i, sell in enumerate(sells):
                        if sell['remaining_qty'] > 0:
                            # Only consider sells that happened after this buy
                            if sell['created_at'] > buy['created_at']:
                                price_diff = abs(sell['price'] - buy_price)
                                price_diffs.append((i, price_diff))

                    if not price_diffs:
                        break  # No eligible sells found

                    # Sort by price difference (ascending)
                    price_diffs.sort(key=lambda x: x[1])

                    # Get the sell with closest price
                    sell_idx = price_diffs[0][0]
                    sell = sells[sell_idx]

                    # Determine quantity to match
                    match_qty = min(buy_qty, sell['remaining_qty'])

                    if match_qty > 0:
                        # Calculate profit for this match
                        trade_profit = (sell['price'] - buy_price) * match_qty

                        # Calculate proportional fees
                        buy_fee_portion = buy['fees'] * (match_qty / buy['quantity'])
                        sell_fee_portion = sell['fees'] * (match_qty / sell['quantity'])
                        total_fees = buy_fee_portion + sell_fee_portion

                        # Final profit after fees
                        net_profit = trade_profit - total_fees

                        matched_pairs.append({
                            'buy_id': buy['id'],
                            'sell_id': sell['id'],
                            'quantity': match_qty,
                            'buy_price': buy_price,
                            'sell_price': sell['price'],
                            'profit': round(net_profit, 2),
                            'fees': round(total_fees, 2)
                        })

                        # Update remaining quantities
                        buy['remaining_qty'] -= match_qty
                        sells[sell_idx]['remaining_qty'] -= match_qty

                        buy_qty -= match_qty

            # Calculate ticker profits and stats
            ticker_profit = sum(m['profit'] for m in matched_pairs)
            total_profit += ticker_profit
            total_matched_trades += len(matched_pairs)

            # Calculate total shares
            buy_shares = sum(b['quantity'] for b in buys)
            sell_shares = sum(s['quantity'] for s in sells)
            matched_shares = sum(m['quantity'] for m in matched_pairs)

            avg_profit_per_share = 0
            if matched_shares > 0:
                avg_profit_per_share = ticker_profit / matched_shares

            results.append({
                "ticker": ticker,
                "buy_orders": len(buys),
                "sell_orders": len(sells),
                "matched_trades": len(matched_pairs),
                "buy_shares": buy_shares,
                "sell_shares": sell_shares,
                "matched_shares": matched_shares,
                "fees": round(sum(m['fees'] for m in matched_pairs), 2),
                "profit": round(ticker_profit, 2),
                "avg_profit_per_share": round(avg_profit_per_share, 2),
                "matches": matched_pairs
            })

        # Sort results by profit (highest first)
        results.sort(key=lambda x: x['profit'], reverse=True)

        return {
            "status": "success",
            "date": date,
            "total_profit": round(total_profit, 2),
            "ticker_results": results,
            "matched_trades": total_matched_trades,
            "trade_count": len(filled_orders),
            "algorithm": "closest_price_matching"
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to analyze trading profit: {str(e)}"}

# ----- Stock Analysis -----

# Simple mapping of common company names to tickers
COMPANY_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "facebook": "META",
    "meta": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "disney": "DIS",
    "walmart": "WMT",
    "coca cola": "KO",
    "coca-cola": "KO",
    "verizon": "VZ",
    "at&t": "T",
    "johnson & johnson": "JNJ",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "bank of america": "BAC",
    "intel": "INTC",
    "amd": "AMD"
}

@mcp.tool()
async def stock_analysis(company_name: str) -> Dict[str, Any]:
    """
    Get a comprehensive stock analysis based on Yahoo Finance data.
    
    This tool provides a buy/sell/hold recommendation with supporting rationale
    based on price trends, financial metrics, and analyst consensus.
    
    Args:
        company_name: Name of the company or its ticker symbol
    
    Returns:
        Dictionary with recommendation and detailed analysis
    """
    from yahoo_small import analyze_stock
    
    try:
        # Try to get the ticker from our mapping
        ticker = company_name.strip().lower()
        if ticker in COMPANY_TO_TICKER:
            ticker = COMPANY_TO_TICKER[ticker]
        else:
            # If not in mapping, assume input is already a ticker and convert to uppercase
            ticker = company_name.strip().upper()
            
        # Get the analysis from the yahoo_small module
        analysis = analyze_stock(ticker)
        
        # Ensure consistent status field for API response
        if analysis.get('recommendation') == 'ERROR':
            return {
                "status": "error",
                "message": f"Failed to analyze {company_name}: {analysis.get('reason', 'Unknown error')}"
            }
        
        return {
            "status": "success",
            "ticker": analysis.get('ticker'),
            "company": company_name,
            "price": analysis.get('price'),
            "recommendation": analysis.get('recommendation'),
            "score": analysis.get('score', 0),
            "analysis": analysis.get('reasons', [])
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Failed to analyze {company_name}: {str(e)}"
        }

# ----- Prompts -----

@mcp.prompt()
def trading_assistant() -> str:
    """
    A prompt for the trading assistant.
    """
    return """
    I'll help you manage your Robinhood account and execute trades. I can provide stock information, 
    place orders, check your portfolio, and more.
    
    Here are some things I can do:
    - Get stock quotes and latest prices
    - Place market and limit orders
    - Check your portfolio and positions
    - View and cancel open orders
    
    IMPORTANT USAGE NOTES:
    - When analyzing trading data, I'll use code to process information efficiently rather than showing all raw data
    - For large datasets (like trading history), I'll provide summaries and analysis rather than dumping all transactions
    - I'll focus on actionable insights rather than verbose raw data dumps
    
    Before executing any trades, I'll need you to log in with your Robinhood credentials.
    What would you like to do today?
    """

@mcp.prompt()
def stock_analysis(ticker: str) -> str:
    """
    A prompt for analyzing a stock.
    """
    return f"""
    I'll help you analyze {ticker} stock. I can provide:
    
    1. Current market data and price information
    2. Basic fundamental analysis
    3. Information about your current position in {ticker} if you own it
    4. Help with placing trades for {ticker}
    
    What specific information about {ticker} would you like to know?
    """

@mcp.prompt()
def portfolio_review() -> str:
    """
    A prompt for reviewing a portfolio.
    """
    return """
    I'll help you review your Robinhood portfolio. We can look at:
    
    1. Overall portfolio value and cash balance
    2. Individual positions and their performance
    3. Open orders that haven't been executed
    4. Historical performance
    
    Would you like a complete overview, or should we focus on a specific aspect of your portfolio?
    """

# ----- Additional Trading Tools -----

class TickerRequest(BaseModel):
    """Model for requesting data for a specific ticker"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL')")

@mcp.tool()
async def get_limit_orders_by_ticker(request: TickerRequest) -> Dict[str, Any]:
    """
    Get all open limit orders (both buy and sell) for a specific ticker.
    
    Returns a detailed list of all pending limit orders for the requested ticker,
    including order type, quantity, limit price, order ID, and creation date.
    
    The results are sorted by order type (buys first, then sells) and then by price
    (highest to lowest for sells, lowest to highest for buys).
    """
    try:
        ticker = request.ticker.upper()
        all_open_orders = rh.orders.get_all_open_stock_orders()
        
        # Filter orders for the specified ticker
        ticker_orders = []
        for order in all_open_orders:
            # Get the ticker from the instrument URL
            instrument_data = rh.stocks.get_instrument_by_url(order.get("instrument", ""))
            order_ticker = instrument_data.get("symbol", "UNKNOWN")
            
            if order_ticker == ticker:
                # Format the order data
                formatted_order = {
                    "order_id": order.get("id", ""),
                    "side": order.get("side", ""),
                    "quantity": float(order.get("quantity", 0)),
                    "type": order.get("type", ""),
                    "limit_price": float(order.get("price", 0)) if order.get("price") else None,
                    "time_in_force": order.get("time_in_force", ""),
                    "created_at": order.get("created_at", ""),
                    "state": order.get("state", ""),
                }
                
                # Convert created_at time from UTC to Eastern Time
                if "created_at" in order and order["created_at"]:
                    # Parse the ISO timestamp
                    utc_time = order["created_at"].replace('Z', '+00:00')
                    # Calculate ET (UTC-4 during daylight saving time)
                    formatted_order["created_at_et"] = f"{utc_time[0:19]}Z (ET: {utc_time[11:16]} ET)"
                
                ticker_orders.append(formatted_order)
        
        # Separate buy and sell orders
        buy_orders = [order for order in ticker_orders if order["side"] == "buy"]
        sell_orders = [order for order in ticker_orders if order["side"] == "sell"]
        
        # Sort buy orders by price (lowest to highest)
        buy_orders.sort(key=lambda x: x.get("limit_price", 0))
        
        # Sort sell orders by price (highest to lowest)
        sell_orders.sort(key=lambda x: x.get("limit_price", 0), reverse=True)
        
        # Combine sorted orders with buys first, then sells
        sorted_orders = buy_orders + sell_orders
        
        return {
            "status": "success",
            "ticker": ticker,
            "total_orders": len(sorted_orders),
            "buy_orders_count": len(buy_orders),
            "sell_orders_count": len(sell_orders),
            "orders": sorted_orders
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get limit orders for {ticker}: {str(e)}"}

@mcp.tool()
async def get_all_limit_orders() -> Dict[str, Any]:
    """
    Get all open limit orders across all tickers.
    
    Returns a summary of all pending limit orders organized by ticker,
    including counts of buy vs sell orders and total order value for each ticker.
    """
    try:
        all_open_orders = rh.orders.get_all_open_stock_orders()
        
        # Group orders by ticker
        ticker_groups = {}
        
        for order in all_open_orders:
            # Skip non-limit orders
            if order.get("type") != "limit":
                continue
                
            # Get the ticker from the instrument URL
            instrument_data = rh.stocks.get_instrument_by_url(order.get("instrument", ""))
            ticker = instrument_data.get("symbol", "UNKNOWN")
            
            # Initialize ticker group if needed
            if ticker not in ticker_groups:
                ticker_groups[ticker] = {
                    "buy_orders": [],
                    "sell_orders": [],
                }
            
            # Extract relevant order data
            order_data = {
                "order_id": order.get("id", ""),
                "quantity": float(order.get("quantity", 0)),
                "limit_price": float(order.get("price", 0)) if order.get("price") else 0,
                "created_at": order.get("created_at", ""),
            }
            
            # Add to appropriate list based on side
            if order.get("side") == "buy":
                ticker_groups[ticker]["buy_orders"].append(order_data)
            else:
                ticker_groups[ticker]["sell_orders"].append(order_data)
        
        # Format results for each ticker
        ticker_summaries = []
        for ticker, data in ticker_groups.items():
            buy_orders = data["buy_orders"]
            sell_orders = data["sell_orders"]
            
            # Calculate total values
            total_buy_value = sum(order["limit_price"] * order["quantity"] for order in buy_orders)
            total_sell_value = sum(order["limit_price"] * order["quantity"] for order in sell_orders)
            
            ticker_summaries.append({
                "ticker": ticker,
                "buy_orders_count": len(buy_orders),
                "sell_orders_count": len(sell_orders),
                "total_orders": len(buy_orders) + len(sell_orders),
                "total_buy_value": round(total_buy_value, 2),
                "total_sell_value": round(total_sell_value, 2),
            })
        
        # Sort by total orders (highest first)
        ticker_summaries.sort(key=lambda x: x["total_orders"], reverse=True)
        
        return {
            "status": "success",
            "total_tickers": len(ticker_summaries),
            "total_limit_orders": sum(summary["total_orders"] for summary in ticker_summaries),
            "ticker_summaries": ticker_summaries
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get limit orders: {str(e)}"}

@mcp.tool()
async def get_dividends() -> Dict[str, Any]:
    """
    Get all dividend payments for the account.
    """
    try:
        dividends = rh.account.get_dividends()
        # Optionally, format and summarize
        return {"status": "success", "dividends": dividends}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get dividends: {str(e)}"}

@mcp.tool()
async def get_dividends_by_ticker(stock_info: StockInfo) -> Dict[str, Any]:
    """
    Get all dividend payments for a specific stock.
    """
    try:
        dividends = rh.account.get_dividends()
        # Filter by ticker
        filtered = []
        for d in dividends:
            instrument_url = d.get("instrument")
            if instrument_url:
                symbol = rh.stocks.get_instrument_by_url(instrument_url).get("symbol", "")
                if symbol.upper() == stock_info.ticker.upper():
                    filtered.append(d)
        return {"status": "success", "ticker": stock_info.ticker.upper(), "dividends": filtered}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get dividends for {stock_info.ticker}: {str(e)}"}

@mcp.tool()
async def get_earnings(stock_info: StockInfo) -> Dict[str, Any]:
    """
    Get earnings events for a specific stock.
    """
    try:
        earnings = rh.stocks.get_earnings(stock_info.ticker)
        return {"status": "success", "ticker": stock_info.ticker.upper(), "earnings": earnings}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get earnings for {stock_info.ticker}: {str(e)}"}

@mcp.tool()
async def get_watchlist() -> Dict[str, Any]:
    """
    Get the user's watchlist from Robinhood.
    
    Returns a list of stocks in the user's watchlist.
    """
    try:
        # Fix for the watchlist API - create a default list if none exists
        # First try to get the default watchlist
        watchlist_data = []
        try:
            watchlist_data = rh.account.get_watchlist_by_name()
        except:
            # If it fails, try creating a default watchlist
            print("No watchlist found, creating default list")
            
        # If no watchlists or empty response, use positions as a fallback
        stocks = []
        if not watchlist_data:
            # Use current positions as a fallback watchlist
            positions = await get_positions()
            if positions.get("status") == "success":
                for position in positions.get("positions", []):
                    stocks.append({
                        "ticker": position.get("ticker"),
                        "list_name": "Portfolio Positions"
                    })
        else:
            # Process actual watchlists
            for list_name, list_data in watchlist_data.items():
                for item in list_data:
                    # Extract the ticker symbol from the instrument data
                    instrument_url = item.get("instrument")
                    if instrument_url:
                        instrument_data = rh.stocks.get_instrument_by_url(instrument_url)
                        ticker = instrument_data.get("symbol", "UNKNOWN")
                        stocks.append({
                            "ticker": ticker,
                            "list_name": list_name
                        })
        
        return {
            "status": "success",
            "watchlists_count": 1 if stocks and not watchlist_data else len(watchlist_data),
            "stocks_count": len(stocks),
            "stocks": stocks
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get watchlist: {str(e)}"}

# ----- Autonomous Functions -----

import asyncio
from telegram_bot_client import send_message

# Store recommendations for each user
# This is a simple in-memory store - in a production environment, you would use a database
user_recommendations = {}

@mcp.tool()
async def run_autonomously() -> Dict[str, Any]:
    """
    Run an autonomous trading strategy based on watchlist analysis.
    
    This function:
    1. Gets all stocks from the user's watchlist
    2. Analyzes each stock for buy/sell recommendations
    3. Sends a summary report via Telegram
    4. Automatically executes the trading recommendations without waiting for user confirmation
    
    Returns:
        Summary of analysis and trading actions
    """
    try:
        chat_id = "5964407322"
        # Step 1: Get all stocks from watchlist
        watchlist_result = await get_watchlist()
        if watchlist_result.get("status") != "success":
            return {"status": "error", "message": f"Failed to get watchlist: {watchlist_result.get('message', 'Unknown error')}"}
        
        stocks = watchlist_result.get("stocks", [])
        if not stocks:
            return {"status": "success", "message": "No stocks found in watchlist. Nothing to analyze."}
        
        # Step 2: Analyze each stock
        analysis_results = []
        buy_recommendations = []
        sell_recommendations = []
        hold_recommendations = []
        
        for stock in stocks:
            ticker = stock.get("ticker")
            print(f"Analyzing {ticker}...")
            
            # Get latest price
            price_result = await get_latest_price(StockInfo(ticker=ticker))
            current_price = None
            if price_result.get("status") == "success":
                current_price = price_result.get("price")
            
            # Run analysis on the stock
            from yahoo_small import analyze_stock
            analysis = analyze_stock(ticker)
            
            # Add the result to our list
            analysis_result = {
                "ticker": ticker,
                "price": current_price,
                "recommendation": analysis.get("recommendation"),
                "score": analysis.get("score", 0),
                "reasons": analysis.get("reasons", [])
            }
            analysis_results.append(analysis_result)
            
            # Sort by recommendation
            if analysis.get("recommendation") == "BUY":
                buy_quantity = 5 if analysis.get("score", 0) >= 0.75 else 1  # Strong buy = 5 shares, Light buy = 1 share
                buy_recommendations.append({
                    "ticker": ticker,
                    "price": current_price,
                    "quantity": buy_quantity,
                    "strength": "Strong" if analysis.get("score", 0) >= 0.75 else "Light",
                    "score": analysis.get("score", 0),
                    "reasons": analysis.get("reasons", [])
                })
            elif analysis.get("recommendation") == "SELL":
                sell_quantity = 5 if analysis.get("score", 0) <= -0.75 else 1  # Strong sell = 5 shares, Light sell = 1 share
                sell_recommendations.append({
                    "ticker": ticker,
                    "price": current_price,
                    "quantity": sell_quantity,
                    "strength": "Strong" if analysis.get("score", 0) <= -0.75 else "Light",
                    "score": analysis.get("score", 0),
                    "reasons": analysis.get("reasons", [])
                })
            else:
                hold_recommendations.append({
                    "ticker": ticker,
                    "price": current_price,
                    "score": analysis.get("score", 0)
                })
        
        # Step 3: Store recommendations for this user
        user_recommendations[chat_id] = {
            "buy": buy_recommendations,
            "sell": sell_recommendations,
            "hold": hold_recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        # Format the analysis message
        message = "<b>🤖 Autonomous Trading Report</b>\n\n"
        
        if buy_recommendations:
            message += "<b>🟢 BUY Recommendations:</b>\n"
            for rec in buy_recommendations:
                message += f"• <b>{rec['ticker']}</b> @ ${rec['price']:.2f} - {rec['strength']} Buy ({rec['quantity']} shares)\n"
                message += f"  <i>Score: {rec['score']:.2f}</i>\n"
                # Add top 2 reasons
                top_reasons = rec['reasons'][:2] if len(rec['reasons']) > 2 else rec['reasons']
                for reason in top_reasons:
                    message += f"  - {reason}\n"
                message += "\n"
                
        if sell_recommendations:
            message += "<b>🔴 SELL Recommendations:</b>\n"
            for rec in sell_recommendations:
                message += f"• <b>{rec['ticker']}</b> @ ${rec['price']:.2f} - {rec['strength']} Sell ({rec['quantity']} shares)\n"
                message += f"  <i>Score: {rec['score']:.2f}</i>\n"
                # Add top 2 reasons
                top_reasons = rec['reasons'][:2] if len(rec['reasons']) > 2 else rec['reasons']
                for reason in top_reasons:
                    message += f"  - {reason}\n"
                message += "\n"
                
        if hold_recommendations:
            message += "<b>⚪ HOLD Recommendations:</b>\n"
            for rec in hold_recommendations:
                message += f"• <b>{rec['ticker']}</b> @ ${rec['price']:.2f} (Score: {rec['score']:.2f})\n"
            message += "\n"
        
        # Add message that orders will be automatically executed
        message += "\n<b>🔄 AUTOMATIC EXECUTION:</b> All recommended trades will be executed automatically.\n"
        message += "<i>Please wait for the confirmation message with execution details.</i>"
        
        # Send the analysis message
        try:
            await send_message(chat_id, message)
        except Exception as e:
            return {"status": "error", "message": f"Failed to send Telegram message: {str(e)}"}
        
        # Step 4: Automatically execute orders
        executed_buys = []
        executed_sells = []
        failed_orders = []
        
        # Execute buy orders
        for order in buy_recommendations:
            ticker = order["ticker"]
            quantity = order["quantity"]
            
            try:
                # Place a limit order 1% below current price for better execution
                current_price = order["price"]
                limit_price = round(current_price * 0.99, 2)  # 1% below current price
                
                # Place the order
                result = await buy_stock_limit_order(LimitOrder(
                    ticker=ticker,
                    quantity=quantity,
                    price=limit_price,
                    time_in_force="gtc",
                    extended_hours=False
                ))
                
                if result.get("status") == "success":
                    executed_buys.append({
                        "ticker": ticker,
                        "quantity": quantity,
                        "limit_price": limit_price,
                        "order_id": result.get("order_id")
                    })
                else:
                    failed_orders.append({
                        "ticker": ticker,
                        "side": "buy",
                        "quantity": quantity,
                        "error": result.get("message", "Unknown error")
                    })
            except Exception as e:
                failed_orders.append({
                    "ticker": ticker,
                    "side": "buy",
                    "quantity": quantity,
                    "error": str(e)
                })
        
        # Execute sell orders
        for order in sell_recommendations:
            ticker = order["ticker"]
            quantity = order["quantity"]
            
            try:
                # Place a limit order 1% above current price for better execution
                current_price = order["price"]
                limit_price = round(current_price * 1.01, 2)  # 1% above current price
                
                # Place the order
                result = await sell_stock_limit_order(LimitOrder(
                    ticker=ticker,
                    quantity=quantity,
                    price=limit_price,
                    time_in_force="gtc",
                    extended_hours=False
                ))
                
                if result.get("status") == "success":
                    executed_sells.append({
                        "ticker": ticker,
                        "quantity": quantity,
                        "limit_price": limit_price,
                        "order_id": result.get("order_id")
                    })
                else:
                    failed_orders.append({
                        "ticker": ticker,
                        "side": "sell",
                        "quantity": quantity,
                        "error": result.get("message", "Unknown error")
                    })
            except Exception as e:
                failed_orders.append({
                    "ticker": ticker,
                    "side": "sell",
                    "quantity": quantity,
                    "error": str(e)
                })
        
        # Send confirmation message
        confirmation_message = "<b>🤖 Order Execution Summary</b>\n\n"
        
        if executed_buys:
            confirmation_message += "<b>✅ Buy Orders Placed:</b>\n"
            for order in executed_buys:
                confirmation_message += f"• {order['ticker']}: {order['quantity']} shares @ ${order['limit_price']:.2f}\n"
            confirmation_message += "\n"
            
        if executed_sells:
            confirmation_message += "<b>✅ Sell Orders Placed:</b>\n"
            for order in executed_sells:
                confirmation_message += f"• {order['ticker']}: {order['quantity']} shares @ ${order['limit_price']:.2f}\n"
            confirmation_message += "\n"
            
        if failed_orders:
            confirmation_message += "<b>❌ Failed Orders:</b>\n"
            for order in failed_orders:
                confirmation_message += f"• {order['ticker']} ({order['side']}): {order['error']}\n"
            confirmation_message += "\n"
            
        if not executed_buys and not executed_sells:
            confirmation_message += "No orders were executed. There were no actionable recommendations.\n"
        
        confirmation_message += "\n<b>All orders placed as limit orders for better execution prices.</b>"
        
        # Add detailed explanation section
        confirmation_message += "\n\n<b>📊 DETAILED ANALYSIS EXPLANATION:</b>\n\n"
        confirmation_message += "The autonomous trading system has completed a comprehensive analysis of your watchlist stocks using a sophisticated multi-factor model. This analysis incorporates price momentum indicators, fundamental valuation metrics, and market sentiment to generate actionable trading recommendations.\n\n"
        confirmation_message += "The system evaluated each stock based on short and medium-term price trends, examining 1-month and 6-month price changes along with relationships to key moving averages. Fundamental factors including P/E ratios, revenue growth trajectories, and profit margin analysis were integrated into the scoring algorithm. Additionally, recent market sentiment and analyst consensus data contributed to the final recommendation strength.\n\n"
        confirmation_message += "Buy recommendations were executed at a strategic 1% discount to current market price using limit orders, which provides better execution value and reduces slippage. Similarly, sell recommendations were implemented at a 1% premium to maximize potential returns. The position sizing was dynamically adjusted based on recommendation strength - stronger signals resulted in larger position sizes to optimize capital allocation based on conviction level.\n\n"
        confirmation_message += "This automated approach eliminates emotional decision-making and ensures disciplined execution of a quantitative trading strategy. The system will continue to monitor these positions and provide updated recommendations as market conditions evolve."
        
        # Send the confirmation
        await send_message(chat_id, confirmation_message)
        
        # Return the summary
        return {
            "status": "success",
            "message": "Autonomous analysis completed and orders executed automatically",
            "stocks_analyzed": len(analysis_results),
            "buy_recommendations": len(buy_recommendations),
            "sell_recommendations": len(sell_recommendations),
            "hold_recommendations": len(hold_recommendations),
            "executed_buys": len(executed_buys),
            "executed_sells": len(executed_sells),
            "failed_orders": len(failed_orders),
            "analysis_results": analysis_results
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to run autonomous analysis: {str(e)}"}

@mcp.tool()
async def process_trading_response(response_text: str) -> Dict[str, Any]:
    """
    Process a user's response to trading recommendations and execute orders.
    
    This function:
    1. Parses the user's response text (e.g., "EXECUTE ALL", "EXECUTE BUY", etc.)
    2. Retrieves stored recommendations from the previous analysis
    3. Executes the requested orders based on the command
    4. Sends a confirmation message with order details
    
    Args:
        response_text: User's response text (e.g., "EXECUTE ALL")
        
    Returns:
        Summary of executed orders
    """
    try:
        # Fixed chat ID
        chat_id = "5964407322"
        
        # Check if we have recommendations for this user
        if chat_id not in user_recommendations:
            await send_message(chat_id, "No active trading recommendations found. Please run analysis first.")
            return {"status": "error", "message": "No active recommendations found"}
        
        recommendations = user_recommendations[chat_id]
        buy_recs = recommendations["buy"]
        sell_recs = recommendations["sell"]
        
        # Default: no orders to execute
        to_execute_buy = []
        to_execute_sell = []
        
        # Parse the response
        response_text = response_text.strip().upper()
        
        if response_text == "EXECUTE ALL":
            # Execute all buy and sell recommendations
            to_execute_buy = buy_recs
            to_execute_sell = sell_recs
        elif response_text == "EXECUTE BUY":
            # Execute only buy recommendations
            to_execute_buy = buy_recs
        elif response_text == "EXECUTE SELL":
            # Execute only sell recommendations
            to_execute_sell = sell_recs
        elif response_text.startswith("EXECUTE "):
            # Execute specific tickers
            tickers = response_text[8:].strip().split()
            tickers = [t.upper() for t in tickers]
            
            # Filter buy recommendations
            to_execute_buy = [rec for rec in buy_recs if rec["ticker"].upper() in tickers]
            
            # Filter sell recommendations
            to_execute_sell = [rec for rec in sell_recs if rec["ticker"].upper() in tickers]
        else:
            # Invalid response
            await send_message(chat_id, "Invalid response. Please use one of the suggested commands.")
            return {"status": "error", "message": "Invalid response"}
        
        # Execute orders and collect results
        executed_buys = []
        executed_sells = []
        failed_orders = []
        
        # Execute buy orders
        for order in to_execute_buy:
            ticker = order["ticker"]
            quantity = order["quantity"]
            
            try:
                # Place a limit order 1% below current price for better execution
                current_price = order["price"]
                limit_price = round(current_price * 0.99, 2)  # 1% below current price
                
                # Place the order
                result = await buy_stock_limit_order(LimitOrder(
                    ticker=ticker,
                    quantity=quantity,
                    price=limit_price,
                    time_in_force="gtc",
                    extended_hours=False
                ))
                
                if result.get("status") == "success":
                    executed_buys.append({
                        "ticker": ticker,
                        "quantity": quantity,
                        "limit_price": limit_price,
                        "order_id": result.get("order_id")
                    })
                else:
                    failed_orders.append({
                        "ticker": ticker,
                        "side": "buy",
                        "quantity": quantity,
                        "error": result.get("message", "Unknown error")
                    })
            except Exception as e:
                failed_orders.append({
                    "ticker": ticker,
                    "side": "buy",
                    "quantity": quantity,
                    "error": str(e)
                })
        
        # Execute sell orders
        for order in to_execute_sell:
            ticker = order["ticker"]
            quantity = order["quantity"]
            
            try:
                # Place a limit order 1% above current price for better execution
                current_price = order["price"]
                limit_price = round(current_price * 1.01, 2)  # 1% above current price
                
                # Place the order
                result = await sell_stock_limit_order(LimitOrder(
                    ticker=ticker,
                    quantity=quantity,
                    price=limit_price,
                    time_in_force="gtc",
                    extended_hours=False
                ))
                
                if result.get("status") == "success":
                    executed_sells.append({
                        "ticker": ticker,
                        "quantity": quantity,
                        "limit_price": limit_price,
                        "order_id": result.get("order_id")
                    })
                else:
                    failed_orders.append({
                        "ticker": ticker,
                        "side": "sell",
                        "quantity": quantity,
                        "error": result.get("message", "Unknown error")
                    })
            except Exception as e:
                failed_orders.append({
                    "ticker": ticker,
                    "side": "sell",
                    "quantity": quantity,
                    "error": str(e)
                })
        
        # Send confirmation message
        confirmation_message = "<b>🤖 Order Execution Summary</b>\n\n"
        
        if executed_buys:
            confirmation_message += "<b>✅ Buy Orders Placed:</b>\n"
            for order in executed_buys:
                confirmation_message += f"• {order['ticker']}: {order['quantity']} shares @ ${order['limit_price']:.2f}\n"
            confirmation_message += "\n"
            
        if executed_sells:
            confirmation_message += "<b>✅ Sell Orders Placed:</b>\n"
            for order in executed_sells:
                confirmation_message += f"• {order['ticker']}: {order['quantity']} shares @ ${order['limit_price']:.2f}\n"
            confirmation_message += "\n"
            
        if failed_orders:
            confirmation_message += "<b>❌ Failed Orders:</b>\n"
            for order in failed_orders:
                confirmation_message += f"• {order['ticker']} ({order['side']}): {order['error']}\n"
            confirmation_message += "\n"
            
        if not executed_buys and not executed_sells:
            confirmation_message += "No orders were executed. Check that you selected valid tickers.\n"
        
        confirmation_message += "\n<b>All orders placed as limit orders for better execution prices.</b>"
        
        # Send the confirmation
        await send_message(chat_id, confirmation_message)
        
        # Return the summary
        return {
            "status": "success",
            "message": "Order execution completed",
            "executed_buys": len(executed_buys),
            "executed_sells": len(executed_sells),
            "failed_orders": len(failed_orders),
            "orders": {
                "buys": executed_buys,
                "sells": executed_sells,
                "failed": failed_orders
            }
        }
        
    except Exception as e:
        await send_message(chat_id, f"<b>⚠️ Error processing your response:</b>\n{str(e)}")
        return {"status": "error", "message": f"Failed to process trading response: {str(e)}"}

# Run the server when executed directly
if __name__ == "__main__":
    mcp.run()