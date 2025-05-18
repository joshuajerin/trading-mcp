# Autonomous Trading Bot Guide

This guide explains how to use the autonomous trading functionality of the Robinhood MCP bot.

## Overview

The autonomous trading system:
1. Analyzes all stocks in your Robinhood watchlist
2. Generates buy/sell recommendations based on technical and fundamental analysis
3. Sends the recommendations to you via Telegram
4. Automatically executes trades without requiring confirmation

## Prerequisites

1. A Robinhood account with stocks in your watchlist
2. Telegram Bot set up with a valid API token
3. Your Telegram chat ID

## Commands

### Starting the Analysis

To start the autonomous analysis, send the following command to your Telegram bot:

```
/trade
```

The bot will:
- Fetch all stocks from your Robinhood watchlist
- Analyze each stock for buy/sell recommendations
- Send you a detailed report via Telegram

### Analysis Results

The analysis report includes:
- Buy recommendations (with strength indicators)
- Sell recommendations (with strength indicators)
- Hold recommendations
- Key reasons for each recommendation
- Instructions for order execution

### Order Execution

Orders are now automatically executed immediately after analysis without requiring your confirmation.

The system will:
1. Send you a detailed analysis of your watchlist stocks
2. Automatically execute all recommended buy and sell orders 
3. Send you a confirmation message with details of all executed trades

No manual commands are required - the entire process is fully automated.

### Order Execution Logic

- Buy orders are placed as limit orders at 1% below the current price
- Sell orders are placed as limit orders at 1% above the current price
- "Strong" recommendations result in orders for 5 shares
- "Light" recommendations result in orders for 1 share

## Recommendation Criteria

The stock analysis uses the following criteria:

1. **Price Trends**:
   - Short-term (1 month) and medium-term (6 month) price changes
   - Position relative to 50-day and 200-day moving averages

2. **Valuation Metrics**:
   - P/E ratio
   - Revenue growth
   - Profit margins

3. **Analyst Recommendations**:
   - Recent analyst ratings and consensus

## Recommendation Strength

- **Strong Buy**: Score ≥ 0.75 (results in 5-share order)
- **Light Buy**: 0.5 ≤ Score < 0.75 (results in 1-share order)
- **Hold**: -0.5 < Score < 0.5 (no action)
- **Light Sell**: -0.75 < Score ≤ -0.5 (results in 1-share order)
- **Strong Sell**: Score ≤ -0.75 (results in 5-share order)

## Important Notes

- The bot will only analyze stocks that are in your Robinhood watchlist
- Recommendations are stored for your chat ID until you run a new analysis
- All orders are placed as limit orders for better execution prices
- You'll receive a confirmation message after orders are placed
- If any orders fail, the error details will be included in the confirmation message