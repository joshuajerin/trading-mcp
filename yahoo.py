#!/usr/bin/env python3
"""
Yahoo Finance Analysis Module

This module analyzes stock data from Yahoo Finance to provide
buy/sell/hold recommendations with detailed reasoning.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Tuple
import re
from textblob import TextBlob  # For simple sentiment analysis

# Recommendation result types
STRONG_BUY = "STRONG BUY"
BUY = "BUY"
BARELY_BUY = "BARELY BUY"
HOLD = "HOLD"
BARELY_SELL = "BARELY SELL"
SELL = "SELL"
STRONG_SELL = "STRONG SELL"


def get_stock_data(ticker: str) -> Dict[str, Any]:
    """
    Get comprehensive stock data for analysis
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary containing all relevant stock data
    """
    stock = yf.Ticker(ticker)
    data = {}
    
    # Basic information
    try:
        data['info'] = stock.info
    except Exception as e:
        print(f"Error getting info: {e}")
        data['info'] = {}
    
    # Price history (1 year of daily data)
    try:
        data['history'] = stock.history(period="1y")
    except Exception as e:
        print(f"Error getting price history: {e}")
        data['history'] = pd.DataFrame()
    
    # News
    try:
        data['news'] = stock.news
    except Exception as e:
        print(f"Error getting news: {e}")
        data['news'] = []
    
    # Analyst recommendations
    try:
        data['recommendations'] = stock.recommendations
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        data['recommendations'] = pd.DataFrame()
    
    # Analyst price targets
    try:
        data['price_target'] = stock.get_analyst_price_targets()
    except Exception as e:
        print(f"Error getting price targets: {e}")
        data['price_target'] = pd.DataFrame()
    
    # Earnings
    try:
        data['earnings'] = stock.get_earnings()
    except Exception as e:
        print(f"Error getting earnings: {e}")
        data['earnings'] = pd.DataFrame()
    
    # Financial statements
    try:
        data['financials'] = stock.quarterly_financials
    except Exception as e:
        print(f"Error getting financials: {e}")
        data['financials'] = pd.DataFrame()
    
    # Balance sheet
    try:
        data['balance_sheet'] = stock.quarterly_balance_sheet
    except Exception as e:
        print(f"Error getting balance sheet: {e}")
        data['balance_sheet'] = pd.DataFrame()
    
    return data


def analyze_price_trend(history: pd.DataFrame) -> Tuple[float, str]:
    """
    Analyze price trends
    
    Args:
        history: Historical price DataFrame
        
    Returns:
        Score (-1 to 1) and explanation
    """
    if history.empty:
        return 0, "No price history available."
    
    # Recent price changes
    try:
        # Get closing prices
        close = history['Close']
        
        # Calculate short-term trend (1 month)
        short_term_change = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100 if len(close) >= 21 else 0
        
        # Calculate medium-term trend (3 months)
        medium_term_change = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63] * 100 if len(close) >= 63 else 0
        
        # Calculate long-term trend (6 months)
        long_term_change = (close.iloc[-1] - close.iloc[-126]) / close.iloc[-126] * 100 if len(close) >= 126 else 0
        
        # Simple 50-day and 200-day moving averages
        ma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        ma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else close.mean()
        current_price = close.iloc[-1]
        
        # Score based on trends
        score = 0
        explanation = []
        
        # Short-term trend
        if short_term_change > 10:
            score += 0.3
            explanation.append(f"Strong short-term uptrend: +{short_term_change:.1f}% in last month")
        elif short_term_change > 5:
            score += 0.2
            explanation.append(f"Positive short-term trend: +{short_term_change:.1f}% in last month")
        elif short_term_change > 0:
            score += 0.1
            explanation.append(f"Slight short-term uptrend: +{short_term_change:.1f}% in last month")
        elif short_term_change > -5:
            score -= 0.1
            explanation.append(f"Slight short-term downtrend: {short_term_change:.1f}% in last month")
        elif short_term_change > -10:
            score -= 0.2
            explanation.append(f"Negative short-term trend: {short_term_change:.1f}% in last month")
        else:
            score -= 0.3
            explanation.append(f"Strong short-term downtrend: {short_term_change:.1f}% in last month")
            
        # Medium-term trend
        if medium_term_change > 15:
            score += 0.3
            explanation.append(f"Strong medium-term uptrend: +{medium_term_change:.1f}% in last 3 months")
        elif medium_term_change > 7.5:
            score += 0.2
            explanation.append(f"Positive medium-term trend: +{medium_term_change:.1f}% in last 3 months")
        elif medium_term_change > 0:
            score += 0.1
            explanation.append(f"Slight medium-term uptrend: +{medium_term_change:.1f}% in last 3 months")
        elif medium_term_change > -7.5:
            score -= 0.1
            explanation.append(f"Slight medium-term downtrend: {medium_term_change:.1f}% in last 3 months")
        elif medium_term_change > -15:
            score -= 0.2
            explanation.append(f"Negative medium-term trend: {medium_term_change:.1f}% in last 3 months")
        else:
            score -= 0.3
            explanation.append(f"Strong medium-term downtrend: {medium_term_change:.1f}% in last 3 months")
            
        # Moving average signals
        if current_price > ma_50 and current_price > ma_200:
            score += 0.2
            explanation.append(f"Bullish: Price above both 50-day and 200-day moving averages")
        elif current_price > ma_50 and current_price < ma_200:
            score += 0.1
            explanation.append(f"Mixed: Price above 50-day MA but below 200-day MA (improving)")
        elif current_price < ma_50 and current_price > ma_200:
            score -= 0.1
            explanation.append(f"Mixed: Price below 50-day MA but above 200-day MA (weakening)")
        else:
            score -= 0.2
            explanation.append(f"Bearish: Price below both 50-day and 200-day moving averages")
            
        # Golden/Death cross
        if ma_50 > ma_200 and ma_50 / ma_200 < 1.01:
            score += 0.2
            explanation.append(f"Recent golden cross: 50-day MA just crossed above 200-day MA (bullish)")
        elif ma_50 < ma_200 and ma_200 / ma_50 < 1.01:
            score -= 0.2
            explanation.append(f"Recent death cross: 50-day MA just crossed below 200-day MA (bearish)")
        elif ma_50 > ma_200:
            score += 0.1
            explanation.append(f"Golden cross in effect: 50-day MA above 200-day MA (bullish)")
        elif ma_50 < ma_200:
            score -= 0.1
            explanation.append(f"Death cross in effect: 50-day MA below 200-day MA (bearish)")
            
        # Ensure score doesn't exceed bounds
        score = max(min(score, 1.0), -1.0)
        
        return score, "\n".join(explanation)
    
    except Exception as e:
        return 0, f"Error analyzing price trends: {e}"


def analyze_news_sentiment(news: List[Dict]) -> Tuple[float, str]:
    """
    Analyze sentiment from recent news
    
    Args:
        news: List of news items
        
    Returns:
        Score (-1 to 1) and explanation
    """
    if not news:
        return 0, "No news available."
    
    try:
        sentiments = []
        news_list = []
        
        for item in news[:10]:  # Analyze up to 10 most recent news items
            if 'content' in item and 'title' in item.get('content', {}):
                title = item['content'].get('title', '')
                summary = item['content'].get('summary', '')
                
                # Combine title and summary for sentiment analysis
                text = f"{title}. {summary}"
                
                # Get sentiment using TextBlob
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
                
                sentiments.append(sentiment)
                news_list.append({
                    'title': title,
                    'sentiment': sentiment
                })
        
        if not sentiments:
            return 0, "No valid news content to analyze."
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Build explanation
        explanation = [f"Average news sentiment: {avg_sentiment:.2f} (-1 to 1 scale)"]
        
        # Add sentiment classification
        if avg_sentiment > 0.5:
            explanation.append("News sentiment is very positive")
            score = 0.5
        elif avg_sentiment > 0.2:
            explanation.append("News sentiment is positive")
            score = 0.3
        elif avg_sentiment > 0.05:
            explanation.append("News sentiment is slightly positive")
            score = 0.1
        elif avg_sentiment > -0.05:
            explanation.append("News sentiment is neutral")
            score = 0
        elif avg_sentiment > -0.2:
            explanation.append("News sentiment is slightly negative")
            score = -0.1
        elif avg_sentiment > -0.5:
            explanation.append("News sentiment is negative")
            score = -0.3
        else:
            explanation.append("News sentiment is very negative")
            score = -0.5
            
        # Add some example headlines
        positive_news = [n['title'] for n in sorted(news_list, key=lambda x: x['sentiment'], reverse=True)[:2]]
        negative_news = [n['title'] for n in sorted(news_list, key=lambda x: x['sentiment'])[:2]]
        
        if positive_news:
            explanation.append("Positive headlines:")
            explanation.extend([f"• {news}" for news in positive_news])
        
        if negative_news:
            explanation.append("Negative headlines:")
            explanation.extend([f"• {news}" for news in negative_news])
            
        return score, "\n".join(explanation)
    
    except Exception as e:
        return 0, f"Error analyzing news sentiment: {e}"


def analyze_analyst_recommendations(recommendations: pd.DataFrame, price_targets: pd.DataFrame) -> Tuple[float, str]:
    """
    Analyze analyst recommendations and price targets
    
    Args:
        recommendations: Analyst recommendations DataFrame
        price_targets: Analyst price targets DataFrame
        
    Returns:
        Score (-1 to 1) and explanation
    """
    score = 0
    explanation = []
    
    # First check if we have recommendations
    if recommendations is None or recommendations.empty:
        explanation.append("No analyst recommendations available.")
    else:
        try:
            # Get the most recent recommendations (last 3 months)
            recent_date = recommendations.index.max() - pd.DateOffset(months=3)
            recent_recs = recommendations[recommendations.index >= recent_date]
            
            if not recent_recs.empty:
                # Count different recommendation types
                rec_counts = recent_recs['To Grade'].value_counts()
                
                # Categorize recommendations as buy, hold, or sell
                buy_terms = ['buy', 'outperform', 'overweight', 'strong buy', 'accumulate']
                sell_terms = ['sell', 'underperform', 'underweight', 'strong sell', 'reduce']
                
                buys = sum(rec_counts.get(grade, 0) for grade in rec_counts.index 
                         if any(term in str(grade).lower() for term in buy_terms))
                
                sells = sum(rec_counts.get(grade, 0) for grade in rec_counts.index 
                         if any(term in str(grade).lower() for term in sell_terms))
                
                holds = sum(rec_counts.get(grade, 0) for grade in rec_counts.index 
                          if 'hold' in str(grade).lower() or 'neutral' in str(grade).lower())
                
                total = buys + sells + holds
                
                if total > 0:
                    buy_pct = buys / total * 100
                    sell_pct = sells / total * 100
                    hold_pct = holds / total * 100
                    
                    explanation.append(f"Recent analyst recommendations: {buys} buys, {holds} holds, {sells} sells")
                    
                    if buy_pct > 70:
                        score += 0.5
                        explanation.append(f"Strong bullish consensus: {buy_pct:.1f}% buy ratings")
                    elif buy_pct > 50:
                        score += 0.3
                        explanation.append(f"Bullish consensus: {buy_pct:.1f}% buy ratings")
                    elif buy_pct > 40 and buy_pct > sell_pct:
                        score += 0.1
                        explanation.append(f"Slightly bullish consensus: {buy_pct:.1f}% buy ratings vs {sell_pct:.1f}% sell ratings")
                    elif sell_pct > 70:
                        score -= 0.5
                        explanation.append(f"Strong bearish consensus: {sell_pct:.1f}% sell ratings")
                    elif sell_pct > 50:
                        score -= 0.3
                        explanation.append(f"Bearish consensus: {sell_pct:.1f}% sell ratings")
                    elif sell_pct > 40 and sell_pct > buy_pct:
                        score -= 0.1
                        explanation.append(f"Slightly bearish consensus: {sell_pct:.1f}% sell ratings vs {buy_pct:.1f}% buy ratings")
                    else:
                        explanation.append(f"Mixed or neutral consensus: {buy_pct:.1f}% buy, {hold_pct:.1f}% hold, {sell_pct:.1f}% sell")
        except Exception as e:
            explanation.append(f"Error analyzing recommendations: {e}")
    
    # Check price targets
    if price_targets is None or price_targets.empty:
        explanation.append("No analyst price targets available.")
    else:
        try:
            if "mean" in price_targets.index and price_targets.shape[1] > 0:
                # Get the mean price target
                mean_target = price_targets.loc["mean"].iloc[-1]
                
                # Compare with current price (need to get current price from elsewhere)
                current_price = None
                if 'info' in price_targets.columns.levels[0] and 'price' in price_targets.index:
                    current_price = price_targets.loc["price", "info"].iloc[-1]
                
                if mean_target and current_price:
                    target_pct = (mean_target - current_price) / current_price * 100
                    
                    explanation.append(f"Mean analyst price target: ${mean_target:.2f} (Current: ${current_price:.2f})")
                    
                    if target_pct > 30:
                        score += 0.5
                        explanation.append(f"Very high upside potential: +{target_pct:.1f}%")
                    elif target_pct > 15:
                        score += 0.3
                        explanation.append(f"High upside potential: +{target_pct:.1f}%")
                    elif target_pct > 5:
                        score += 0.1
                        explanation.append(f"Moderate upside potential: +{target_pct:.1f}%")
                    elif target_pct > -5:
                        explanation.append(f"Limited price movement expected: {target_pct:.1f}%")
                    elif target_pct > -15:
                        score -= 0.1
                        explanation.append(f"Moderate downside risk: {target_pct:.1f}%")
                    elif target_pct > -30:
                        score -= 0.3
                        explanation.append(f"High downside risk: {target_pct:.1f}%")
                    else:
                        score -= 0.5
                        explanation.append(f"Very high downside risk: {target_pct:.1f}%")
        except Exception as e:
            explanation.append(f"Error analyzing price targets: {e}")
    
    # Ensure score is within bounds
    score = max(min(score, 1.0), -1.0)
    
    return score, "\n".join(explanation)


def analyze_financials(info: Dict, financials: pd.DataFrame, earnings: pd.DataFrame) -> Tuple[float, str]:
    """
    Analyze financial data, metrics, and growth
    
    Args:
        info: Basic stock info
        financials: Financial statements
        earnings: Earnings data
        
    Returns:
        Score (-1 to 1) and explanation
    """
    score = 0
    explanation = []
    
    try:
        # Valuation metrics
        pe_ratio = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        peg_ratio = info.get('pegRatio')
        price_to_book = info.get('priceToBook')
        
        if pe_ratio is not None:
            explanation.append(f"Trailing P/E Ratio: {pe_ratio:.2f}")
            
            # Evaluate P/E ratio (simplified evaluation)
            if pe_ratio < 10:
                score += 0.3
                explanation.append("Low P/E ratio (potentially undervalued)")
            elif pe_ratio < 20:
                score += 0.1
                explanation.append("Moderate P/E ratio")
            elif pe_ratio < 30:
                score -= 0.1
                explanation.append("High P/E ratio (potentially overvalued)")
            else:
                score -= 0.3
                explanation.append("Very high P/E ratio (potentially significantly overvalued)")
                
        if forward_pe is not None and pe_ratio is not None:
            explanation.append(f"Forward P/E Ratio: {forward_pe:.2f}")
            
            # Compare forward P/E to trailing P/E
            if forward_pe < pe_ratio * 0.8:
                score += 0.2
                explanation.append("Forward P/E significantly lower than trailing P/E (suggests improving earnings)")
            elif forward_pe < pe_ratio:
                score += 0.1
                explanation.append("Forward P/E lower than trailing P/E (suggests some earnings improvement)")
            elif forward_pe > pe_ratio * 1.2:
                score -= 0.2
                explanation.append("Forward P/E significantly higher than trailing P/E (suggests deteriorating earnings)")
            elif forward_pe > pe_ratio:
                score -= 0.1
                explanation.append("Forward P/E higher than trailing P/E (suggests some earnings pressure)")
                
        if peg_ratio is not None:
            explanation.append(f"PEG Ratio: {peg_ratio:.2f}")
            
            # Evaluate PEG ratio
            if peg_ratio < 1:
                score += 0.2
                explanation.append("PEG ratio below 1.0 (potentially undervalued relative to growth)")
            elif peg_ratio < 1.5:
                score += 0.1
                explanation.append("PEG ratio moderately above 1.0 (fairly valued relative to growth)")
            elif peg_ratio < 2:
                score -= 0.1
                explanation.append("PEG ratio high (potentially overvalued relative to growth)")
            else:
                score -= 0.2
                explanation.append("PEG ratio very high (potentially significantly overvalued relative to growth)")
        
        # Growth metrics
        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')
        
        if revenue_growth is not None:
            revenue_growth_pct = revenue_growth * 100
            explanation.append(f"Revenue Growth: {revenue_growth_pct:.1f}%")
            
            # Evaluate revenue growth
            if revenue_growth_pct > 30:
                score += 0.3
                explanation.append("Exceptional revenue growth")
            elif revenue_growth_pct > 15:
                score += 0.2
                explanation.append("Strong revenue growth")
            elif revenue_growth_pct > 5:
                score += 0.1
                explanation.append("Moderate revenue growth")
            elif revenue_growth_pct > 0:
                explanation.append("Slight revenue growth")
            elif revenue_growth_pct > -5:
                score -= 0.1
                explanation.append("Slight revenue decline")
            elif revenue_growth_pct > -15:
                score -= 0.2
                explanation.append("Significant revenue decline")
            else:
                score -= 0.3
                explanation.append("Severe revenue decline")
        
        if earnings_growth is not None:
            earnings_growth_pct = earnings_growth * 100
            explanation.append(f"Earnings Growth: {earnings_growth_pct:.1f}%")
            
            # Evaluate earnings growth
            if earnings_growth_pct > 30:
                score += 0.3
                explanation.append("Exceptional earnings growth")
            elif earnings_growth_pct > 15:
                score += 0.2
                explanation.append("Strong earnings growth")
            elif earnings_growth_pct > 5:
                score += 0.1
                explanation.append("Moderate earnings growth")
            elif earnings_growth_pct > 0:
                explanation.append("Slight earnings growth")
            elif earnings_growth_pct > -5:
                score -= 0.1
                explanation.append("Slight earnings decline")
            elif earnings_growth_pct > -15:
                score -= 0.2
                explanation.append("Significant earnings decline")
            else:
                score -= 0.3
                explanation.append("Severe earnings decline")
                
        # Profitability metrics
        profit_margin = info.get('profitMargins')
        
        if profit_margin is not None:
            profit_margin_pct = profit_margin * 100
            explanation.append(f"Profit Margin: {profit_margin_pct:.1f}%")
            
            # Evaluate profit margin
            if profit_margin_pct > 20:
                score += 0.3
                explanation.append("Excellent profit margin")
            elif profit_margin_pct > 10:
                score += 0.2
                explanation.append("Strong profit margin")
            elif profit_margin_pct > 5:
                score += 0.1
                explanation.append("Good profit margin")
            elif profit_margin_pct > 0:
                explanation.append("Positive but modest profit margin")
            else:
                score -= 0.2
                explanation.append("Negative profit margin (company is losing money)")
                
        # Earnings surprises
        if not earnings.empty and 'Surprise(%)' in earnings.columns:
            # Get the last few earnings surprises
            last_surprises = earnings['Surprise(%)'].dropna().iloc[-4:].tolist()
            
            if last_surprises:
                avg_surprise = sum(last_surprises) / len(last_surprises)
                explanation.append(f"Average Earnings Surprise: {avg_surprise:.1f}%")
                
                # Count positive vs negative surprises
                positive_surprises = sum(1 for s in last_surprises if s > 0)
                negative_surprises = sum(1 for s in last_surprises if s < 0)
                
                # Evaluate earnings surprises
                if avg_surprise > 15:
                    score += 0.3
                    explanation.append("Very strong earnings surprises")
                elif avg_surprise > 5:
                    score += 0.2
                    explanation.append("Positive earnings surprises")
                elif avg_surprise > 0:
                    score += 0.1
                    explanation.append("Slight positive earnings surprises")
                elif avg_surprise > -5:
                    score -= 0.1
                    explanation.append("Slight negative earnings surprises")
                else:
                    score -= 0.2
                    explanation.append("Significant negative earnings surprises")
                    
                if positive_surprises == len(last_surprises):
                    score += 0.1
                    explanation.append("Consistent positive earnings surprises")
                elif negative_surprises == len(last_surprises):
                    score -= 0.1
                    explanation.append("Consistent negative earnings surprises")
    
    except Exception as e:
        explanation.append(f"Error analyzing financials: {e}")
    
    # Ensure score is within bounds
    score = max(min(score, 1.0), -1.0)
    
    return score, "\n".join(explanation)


def get_recommendation(ticker: str) -> Dict[str, Any]:
    """
    Get a comprehensive buy/sell/hold recommendation for a stock
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with recommendation and detailed analysis
    """
    try:
        # Get stock data
        data = get_stock_data(ticker)
        
        # Perform analyses
        price_score, price_explanation = analyze_price_trend(data['history'])
        news_score, news_explanation = analyze_news_sentiment(data['news'])
        analyst_score, analyst_explanation = analyze_analyst_recommendations(data['recommendations'], data['price_target'])
        financial_score, financial_explanation = analyze_financials(data['info'], data['financials'], data['earnings'])
        
        # Calculate overall score (weighted)
        weights = {
            'price': 0.3,      # Price trends
            'news': 0.1,       # News sentiment
            'analyst': 0.3,    # Analyst recommendations
            'financial': 0.3   # Financial metrics
        }
        
        overall_score = (
            price_score * weights['price'] +
            news_score * weights['news'] +
            analyst_score * weights['analyst'] +
            financial_score * weights['financial']
        )
        
        # Determine recommendation
        if overall_score >= 0.6:
            recommendation = STRONG_BUY
        elif overall_score >= 0.3:
            recommendation = BUY
        elif overall_score >= 0.1:
            recommendation = BARELY_BUY
        elif overall_score >= -0.1:
            recommendation = HOLD
        elif overall_score >= -0.3:
            recommendation = BARELY_SELL
        elif overall_score >= -0.6:
            recommendation = SELL
        else:
            recommendation = STRONG_SELL
        
        # Create summary
        summary = f"Recommendation for {ticker}: {recommendation} (Score: {overall_score:.2f})\n\n"
        summary += "ANALYSIS SUMMARY:\n"
        summary += f"• Price Trend Analysis: {price_score:.2f} - {'Bullish' if price_score > 0 else 'Bearish' if price_score < 0 else 'Neutral'}\n"
        summary += f"• News Sentiment: {news_score:.2f} - {'Positive' if news_score > 0 else 'Negative' if news_score < 0 else 'Neutral'}\n"
        summary += f"• Analyst Consensus: {analyst_score:.2f} - {'Bullish' if analyst_score > 0 else 'Bearish' if analyst_score < 0 else 'Neutral'}\n"
        summary += f"• Financial Health: {financial_score:.2f} - {'Strong' if financial_score > 0 else 'Weak' if financial_score < 0 else 'Average'}\n"
        
        # Get current price
        current_price = data['history']['Close'].iloc[-1] if not data['history'].empty else None
        price_str = f"${current_price:.2f}" if current_price else "Unknown"
        
        # Build result
        result = {
            'ticker': ticker,
            'current_price': price_str,
            'recommendation': recommendation,
            'overall_score': round(overall_score, 2),
            'summary': summary,
            'details': {
                'price_trend': {
                    'score': round(price_score, 2),
                    'explanation': price_explanation
                },
                'news_sentiment': {
                    'score': round(news_score, 2),
                    'explanation': news_explanation
                },
                'analyst_consensus': {
                    'score': round(analyst_score, 2),
                    'explanation': analyst_explanation
                },
                'financial_health': {
                    'score': round(financial_score, 2),
                    'explanation': financial_explanation
                }
            }
        }
        
        return result
    
    except Exception as e:
        return {
            'ticker': ticker,
            'recommendation': 'ERROR',
            'error': str(e),
            'summary': f"Error analyzing {ticker}: {str(e)}"
        }


if __name__ == "__main__":
    import sys
    
    # If a ticker is provided as a command-line argument
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        print(f"Analyzing {ticker}...")
        recommendation = get_recommendation(ticker)
        print(recommendation['summary'])
        print("\nDETAILED ANALYSIS:")
        print("\nPRICE TREND ANALYSIS:")
        print(recommendation['details']['price_trend']['explanation'])
        print("\nNEWS SENTIMENT ANALYSIS:")
        print(recommendation['details']['news_sentiment']['explanation'])
        print("\nANALYST CONSENSUS ANALYSIS:")
        print(recommendation['details']['analyst_consensus']['explanation'])
        print("\nFINANCIAL HEALTH ANALYSIS:")
        print(recommendation['details']['financial_health']['explanation'])
    else:
        print("Usage: python yahoo.py TICKER")
        print("Example: python yahoo.py AAPL") 