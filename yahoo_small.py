#!/usr/bin/env python3
"""
Yahoo Finance Enhanced Analysis

Comprehensive stock analyzer that provides detailed buy/sell recommendations
with technical indicators, fundamental analysis, and market sentiment
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timedelta

# Recommendation types
BUY = "BUY"
HOLD = "HOLD"
SELL = "SELL"

def get_basic_stock_data(ticker: str) -> Dict[str, Any]:
    """Get comprehensive stock data for detailed analysis"""
    try:
        stock = yf.Ticker(ticker)
        data = {
            'info': stock.info,
            'history': stock.history(period="1y"),  # Extended to 1 year for better trend analysis
            'recommendations': stock.recommendations,
            'price': stock.history(period="1d")['Close'].iloc[-1] if not stock.history(period="1d").empty else None,
            'earnings': stock.earnings,
            'institutional_holders': stock.institutional_holders,
            'balance_sheet': stock.balance_sheet,
            'cashflow': stock.cashflow,
            'financials': stock.financials
        }
        return data
    except Exception as e:
        print(f"Error getting data for {ticker}: {e}")
        return {'error': str(e)}

def calculate_rsi(prices: pd.Series, window: int = 14) -> float:
    """Calculate Relative Strength Index"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return 50  # Neutral value

def calculate_macd(prices: pd.Series) -> Tuple[float, float, float]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return macd.iloc[-1], signal.iloc[-1], histogram.iloc[-1]
    except Exception as e:
        print(f"Error calculating MACD: {e}")
        return 0, 0, 0

def calculate_bollinger_bands(prices: pd.Series, window: int = 20) -> Tuple[float, float, float]:
    """Calculate Bollinger Bands"""
    try:
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = middle_band + (std * 2)
        lower_band = middle_band - (std * 2)
        
        return lower_band.iloc[-1], middle_band.iloc[-1], upper_band.iloc[-1]
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")
        return 0, 0, 0

def analyze_volume_trend(history: pd.DataFrame) -> Tuple[float, str]:
    """Analyze volume trend"""
    try:
        recent_volume = history['Volume'].iloc[-5:].mean()
        past_volume = history['Volume'].iloc[-20:-5].mean()
        
        volume_change = (recent_volume - past_volume) / past_volume * 100
        
        if volume_change > 30:
            return 1, f"Strong volume increase: +{volume_change:.1f}%"
        elif volume_change > 15:
            return 0.5, f"Moderate volume increase: +{volume_change:.1f}%"
        elif volume_change < -30:
            return -1, f"Strong volume decrease: {volume_change:.1f}%"
        elif volume_change < -15:
            return -0.5, f"Moderate volume decrease: {volume_change:.1f}%"
        else:
            return 0, f"Stable volume pattern: {volume_change:.1f}%"
    except Exception as e:
        print(f"Error analyzing volume trend: {e}")
        return 0, "Could not analyze volume trend"

def analyze_stock(ticker: str) -> Dict[str, Any]:
    """
    Comprehensive analysis of a stock with detailed buy/sell recommendation
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with recommendation and detailed analysis
    """
    # Get data
    data = get_basic_stock_data(ticker)
    if 'error' in data:
        return {'ticker': ticker, 'recommendation': 'ERROR', 'reason': data['error']}
    
    # Set up scores and reasons
    scores = []
    reasons = []
    technical_indicators = []
    fundamental_metrics = []
    market_sentiment = []
    industry_comparison = []
    risk_assessment = []
    
    current_price = data['price']
    
    # 1. Technical Analysis
    try:
        history = data['history']
        if not history.empty and len(history) > 20:
            recent_price = history['Close'].iloc[-1]
            week_ago_price = history['Close'].iloc[-5] if len(history) >= 5 else history['Close'].iloc[0]
            month_ago_price = history['Close'].iloc[-22] if len(history) >= 22 else history['Close'].iloc[0]
            three_month_ago_price = history['Close'].iloc[-66] if len(history) >= 66 else history['Close'].iloc[0]
            six_month_ago_price = history['Close'].iloc[-132] if len(history) >= 132 else history['Close'].iloc[0]
            year_ago_price = history['Close'].iloc[0]
            
            week_change = (recent_price - week_ago_price) / week_ago_price * 100
            month_change = (recent_price - month_ago_price) / month_ago_price * 100
            three_month_change = (recent_price - three_month_ago_price) / three_month_ago_price * 100
            six_month_change = (recent_price - six_month_ago_price) / six_month_ago_price * 100
            year_change = (recent_price - year_ago_price) / year_ago_price * 100
            
            # Detailed price trend analysis
            if month_change > 10 and three_month_change > 20 and six_month_change > 30:
                scores.append(1.5)  # Very strong uptrend
                technical_indicators.append(f"Exceptional uptrend: +{week_change:.1f}% (1wk), +{month_change:.1f}% (1mo), +{three_month_change:.1f}% (3mo), +{six_month_change:.1f}% (6mo), +{year_change:.1f}% (1yr)")
            elif month_change > 5 and three_month_change > 10 and six_month_change > 15:
                scores.append(1)  # Strong uptrend
                technical_indicators.append(f"Strong uptrend: +{week_change:.1f}% (1wk), +{month_change:.1f}% (1mo), +{three_month_change:.1f}% (3mo), +{six_month_change:.1f}% (6mo), +{year_change:.1f}% (1yr)")
            elif month_change > 2 and three_month_change > 5:
                scores.append(0.5)  # Moderate uptrend
                technical_indicators.append(f"Moderate uptrend: +{week_change:.1f}% (1wk), +{month_change:.1f}% (1mo), +{three_month_change:.1f}% (3mo), +{six_month_change:.1f}% (6mo), +{year_change:.1f}% (1yr)")
            elif month_change < -10 and three_month_change < -20 and six_month_change < -30:
                scores.append(-1.5)  # Very strong downtrend
                technical_indicators.append(f"Severe downtrend: {week_change:.1f}% (1wk), {month_change:.1f}% (1mo), {three_month_change:.1f}% (3mo), {six_month_change:.1f}% (6mo), {year_change:.1f}% (1yr)")
            elif month_change < -5 and three_month_change < -10 and six_month_change < -15:
                scores.append(-1)  # Strong downtrend
                technical_indicators.append(f"Strong downtrend: {week_change:.1f}% (1wk), {month_change:.1f}% (1mo), {three_month_change:.1f}% (3mo), {six_month_change:.1f}% (6mo), {year_change:.1f}% (1yr)")
            elif month_change < -2 and three_month_change < -5:
                scores.append(-0.5)  # Moderate downtrend
                technical_indicators.append(f"Moderate downtrend: {week_change:.1f}% (1wk), {month_change:.1f}% (1mo), {three_month_change:.1f}% (3mo), {six_month_change:.1f}% (6mo), {year_change:.1f}% (1yr)")
            else:
                scores.append(0)  # Neutral
                technical_indicators.append(f"Neutral trend: {week_change:.1f}% (1wk), {month_change:.1f}% (1mo), {three_month_change:.1f}% (3mo), {six_month_change:.1f}% (6mo), {year_change:.1f}% (1yr)")
            
            # Moving averages (multiple timeframes)
            if len(history) >= 200:
                ma_20 = history['Close'].rolling(window=20).mean().iloc[-1]
                ma_50 = history['Close'].rolling(window=50).mean().iloc[-1]
                ma_100 = history['Close'].rolling(window=100).mean().iloc[-1]
                ma_200 = history['Close'].rolling(window=200).mean().iloc[-1]
                
                ma_relationships = []
                ma_score = 0
                
                if recent_price > ma_20:
                    ma_relationships.append("Price > 20-day MA")
                    ma_score += 0.25
                else:
                    ma_relationships.append("Price < 20-day MA")
                    ma_score -= 0.25
                    
                if recent_price > ma_50:
                    ma_relationships.append("Price > 50-day MA")
                    ma_score += 0.25
                else:
                    ma_relationships.append("Price < 50-day MA")
                    ma_score -= 0.25
                
                if recent_price > ma_100:
                    ma_relationships.append("Price > 100-day MA")
                    ma_score += 0.25
                else:
                    ma_relationships.append("Price < 100-day MA")
                    ma_score -= 0.25
                    
                if recent_price > ma_200:
                    ma_relationships.append("Price > 200-day MA")
                    ma_score += 0.25
                else:
                    ma_relationships.append("Price < 200-day MA")
                    ma_score -= 0.25
                
                if ma_20 > ma_50 > ma_100 > ma_200:
                    ma_relationships.append("Golden alignment (all MAs in uptrend sequence)")
                    ma_score += 1
                elif ma_20 < ma_50 < ma_100 < ma_200:
                    ma_relationships.append("Death cross alignment (all MAs in downtrend sequence)")
                    ma_score -= 1
                
                # Add to scores
                scores.append(ma_score)
                ma_description = " | ".join(ma_relationships)
                technical_indicators.append(f"Moving averages: {ma_description}")
            
            # RSI Analysis
            rsi = calculate_rsi(history['Close'])
            if rsi > 70:
                scores.append(-0.5)  # Overbought
                technical_indicators.append(f"RSI Overbought: {rsi:.1f} (>70 indicates potential reversal)")
            elif rsi < 30:
                scores.append(0.5)  # Oversold
                technical_indicators.append(f"RSI Oversold: {rsi:.1f} (<30 indicates potential buying opportunity)")
            else:
                technical_indicators.append(f"RSI Neutral: {rsi:.1f}")
            
            # MACD Analysis
            macd, signal, histogram = calculate_macd(history['Close'])
            if macd > signal and histogram > 0 and histogram > histogram.iloc[-2] if isinstance(histogram, pd.Series) else 0:
                scores.append(0.5)  # Bullish MACD
                technical_indicators.append(f"Bullish MACD: {macd:.3f} > Signal {signal:.3f} with rising histogram")
            elif macd < signal and histogram < 0 and histogram < histogram.iloc[-2] if isinstance(histogram, pd.Series) else 0:
                scores.append(-0.5)  # Bearish MACD
                technical_indicators.append(f"Bearish MACD: {macd:.3f} < Signal {signal:.3f} with falling histogram")
            else:
                technical_indicators.append(f"Neutral MACD: {macd:.3f} vs Signal {signal:.3f}")
            
            # Bollinger Bands
            lower_band, middle_band, upper_band = calculate_bollinger_bands(history['Close'])
            band_width = ((upper_band - lower_band) / middle_band) * 100
            
            if recent_price > upper_band:
                scores.append(-0.5)  # Overbought
                technical_indicators.append(f"Above upper Bollinger Band: ${recent_price:.2f} > ${upper_band:.2f} (potential reversal)")
            elif recent_price < lower_band:
                scores.append(0.5)  # Oversold
                technical_indicators.append(f"Below lower Bollinger Band: ${recent_price:.2f} < ${lower_band:.2f} (potential buying opportunity)")
            else:
                position_in_band = ((recent_price - lower_band) / (upper_band - lower_band)) * 100
                technical_indicators.append(f"Within Bollinger Bands: {position_in_band:.1f}% from lower band (width: {band_width:.1f}%)")
            
            # Volume Analysis
            volume_score, volume_reason = analyze_volume_trend(history)
            scores.append(volume_score)
            technical_indicators.append(volume_reason)
            
            # Volatility Analysis
            recent_volatility = history['Close'].pct_change().iloc[-20:].std() * 100
            if recent_volatility > 3:
                risk_assessment.append(f"High volatility: {recent_volatility:.2f}% daily std dev (above average)")
            elif recent_volatility < 1:
                risk_assessment.append(f"Low volatility: {recent_volatility:.2f}% daily std dev (below average)")
            else:
                risk_assessment.append(f"Average volatility: {recent_volatility:.2f}% daily std dev")
                
    except Exception as e:
        technical_indicators.append(f"Could not complete technical analysis: {e}")
    
    # 2. Fundamental Analysis
    try:
        info = data['info']
        
        # Valuation metrics
        pe_ratio = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        peg_ratio = info.get('pegRatio')
        price_to_book = info.get('priceToBook')
        price_to_sales = info.get('priceToSales')
        ev_to_ebitda = info.get('enterpriseToEbitda')
        
        # Financials
        revenue_growth = info.get('revenueGrowth')
        profit_margin = info.get('profitMargins')
        operating_margin = info.get('operatingMargins')
        roi = info.get('returnOnAssets')
        roe = info.get('returnOnEquity')
        debt_to_equity = info.get('debtToEquity')
        current_ratio = info.get('currentRatio')
        
        # Dividend metrics
        dividend_yield = info.get('dividendYield', 0)
        payout_ratio = info.get('payoutRatio', 0)
        
        # Valuation score calculation
        valuation_score = 0
        valuation_reasons = []
        
        # P/E analysis
        if pe_ratio:
            if pe_ratio < 12:
                valuation_score += 1
                valuation_reasons.append(f"Very low P/E ratio: {pe_ratio:.1f} (potential value)")
            elif pe_ratio < 18:
                valuation_score += 0.5
                valuation_reasons.append(f"Below average P/E ratio: {pe_ratio:.1f}")
            elif pe_ratio > 50:
                valuation_score -= 1
                valuation_reasons.append(f"Very high P/E ratio: {pe_ratio:.1f} (potentially overvalued)")
            elif pe_ratio > 30:
                valuation_score -= 0.5
                valuation_reasons.append(f"High P/E ratio: {pe_ratio:.1f}")
            else:
                valuation_reasons.append(f"Average P/E ratio: {pe_ratio:.1f}")
                
        # Forward P/E comparison
        if pe_ratio and forward_pe:
            if forward_pe < pe_ratio * 0.8:
                valuation_score += 0.5
                valuation_reasons.append(f"Forward P/E ({forward_pe:.1f}) much lower than trailing P/E ({pe_ratio:.1f}), suggesting improving earnings")
            elif forward_pe > pe_ratio * 1.2:
                valuation_score -= 0.5
                valuation_reasons.append(f"Forward P/E ({forward_pe:.1f}) much higher than trailing P/E ({pe_ratio:.1f}), suggesting potential earnings decline")
        
        # PEG Ratio
        if peg_ratio:
            if peg_ratio < 1:
                valuation_score += 0.5
                valuation_reasons.append(f"PEG ratio below 1: {peg_ratio:.2f} (growth may be undervalued)")
            elif peg_ratio > 2:
                valuation_score -= 0.5
                valuation_reasons.append(f"PEG ratio above 2: {peg_ratio:.2f} (growth may be overvalued)")
            else:
                valuation_reasons.append(f"PEG ratio: {peg_ratio:.2f}")
        
        # Price-to-Book
        if price_to_book:
            if price_to_book < 1:
                valuation_score += 0.5
                valuation_reasons.append(f"Price-to-Book below 1: {price_to_book:.2f} (trading below book value)")
            elif price_to_book > 5:
                valuation_score -= 0.5
                valuation_reasons.append(f"Price-to-Book above 5: {price_to_book:.2f} (premium to book value)")
            else:
                valuation_reasons.append(f"Price-to-Book: {price_to_book:.2f}")
                
        # Price-to-Sales
        if price_to_sales:
            if price_to_sales < 1:
                valuation_score += 0.5
                valuation_reasons.append(f"Price-to-Sales below 1: {price_to_sales:.2f} (potential value)")
            elif price_to_sales > 10:
                valuation_score -= 0.5
                valuation_reasons.append(f"Price-to-Sales above 10: {price_to_sales:.2f} (high expectations)")
            else:
                valuation_reasons.append(f"Price-to-Sales: {price_to_sales:.2f}")
        
        # EV/EBITDA
        if ev_to_ebitda:
            if ev_to_ebitda < 8:
                valuation_score += 0.5
                valuation_reasons.append(f"EV/EBITDA below 8: {ev_to_ebitda:.2f} (potentially undervalued)")
            elif ev_to_ebitda > 15:
                valuation_score -= 0.5
                valuation_reasons.append(f"EV/EBITDA above 15: {ev_to_ebitda:.2f} (potentially overvalued)")
            else:
                valuation_reasons.append(f"EV/EBITDA: {ev_to_ebitda:.2f}")
        
        # Add valuation score
        scores.append(valuation_score)
        fundamental_metrics.extend(valuation_reasons)
        
        # Financial health score calculation
        financial_score = 0
        financial_reasons = []
        
        # Revenue Growth
        if revenue_growth:
            if revenue_growth > 0.3:  # >30% growth
                financial_score += 1
                financial_reasons.append(f"Exceptional revenue growth: {revenue_growth*100:.1f}%")
            elif revenue_growth > 0.15:  # >15% growth
                financial_score += 0.5
                financial_reasons.append(f"Strong revenue growth: {revenue_growth*100:.1f}%")
            elif revenue_growth > 0.05:  # >5% growth
                financial_score += 0.25
                financial_reasons.append(f"Positive revenue growth: {revenue_growth*100:.1f}%")
            elif revenue_growth < -0.1:  # <-10% growth (significant decline)
                financial_score -= 1
                financial_reasons.append(f"Significant revenue decline: {revenue_growth*100:.1f}%")
            elif revenue_growth < -0.05:  # <-5% growth (moderate decline)
                financial_score -= 0.5
                financial_reasons.append(f"Revenue decline: {revenue_growth*100:.1f}%")
            else:
                financial_reasons.append(f"Flat revenue growth: {revenue_growth*100:.1f}%")
        
        # Profit Margins
        if profit_margin:
            if profit_margin > 0.25:  # >25% margin
                financial_score += 1
                financial_reasons.append(f"Exceptional profit margin: {profit_margin*100:.1f}%")
            elif profit_margin > 0.15:  # >15% margin
                financial_score += 0.5
                financial_reasons.append(f"Strong profit margin: {profit_margin*100:.1f}%")
            elif profit_margin > 0.05:  # >5% margin
                financial_score += 0.25
                financial_reasons.append(f"Positive profit margin: {profit_margin*100:.1f}%")
            elif profit_margin < 0:  # Negative margin
                financial_score -= 0.5
                financial_reasons.append(f"Negative profit margin: {profit_margin*100:.1f}%")
            else:
                financial_reasons.append(f"Thin profit margin: {profit_margin*100:.1f}%")
        
        # Operating Margins
        if operating_margin:
            if operating_margin > 0.25:  # >25% margin
                financial_score += 0.5
                financial_reasons.append(f"Strong operating margin: {operating_margin*100:.1f}%")
            elif operating_margin < 0:  # Negative margin
                financial_score -= 0.5
                financial_reasons.append(f"Negative operating margin: {operating_margin*100:.1f}%")
        
        # Return on Equity
        if roe:
            if roe > 0.2:  # >20% ROE
                financial_score += 0.5
                financial_reasons.append(f"Strong return on equity: {roe*100:.1f}%")
            elif roe < 0:  # Negative ROE
                financial_score -= 0.5
                financial_reasons.append(f"Negative return on equity: {roe*100:.1f}%")
        
        # Debt-to-Equity
        if debt_to_equity:
            if debt_to_equity > 2:
                financial_score -= 0.5
                financial_reasons.append(f"High debt-to-equity ratio: {debt_to_equity:.2f}")
                risk_assessment.append(f"High leverage: Debt-to-equity ratio of {debt_to_equity:.2f} (above average)")
            elif debt_to_equity < 0.5:
                financial_score += 0.5
                financial_reasons.append(f"Low debt-to-equity ratio: {debt_to_equity:.2f}")
                risk_assessment.append(f"Low leverage: Debt-to-equity ratio of {debt_to_equity:.2f} (below average)")
            else:
                financial_reasons.append(f"Moderate debt-to-equity ratio: {debt_to_equity:.2f}")
        
        # Current Ratio
        if current_ratio:
            if current_ratio < 1:
                financial_score -= 0.5
                financial_reasons.append(f"Low current ratio: {current_ratio:.2f} (potential liquidity concerns)")
                risk_assessment.append(f"Potential liquidity risk: Current ratio of {current_ratio:.2f} (below 1.0)")
            elif current_ratio > 2:
                financial_score += 0.25
                financial_reasons.append(f"Strong current ratio: {current_ratio:.2f} (good liquidity)")
            else:
                financial_reasons.append(f"Adequate current ratio: {current_ratio:.2f}")
        
        # Dividend Analysis
        if dividend_yield:
            if dividend_yield > 0.04:  # >4% yield
                financial_score += 0.5
                financial_reasons.append(f"High dividend yield: {dividend_yield*100:.2f}%")
            elif dividend_yield > 0.02:  # >2% yield
                financial_score += 0.25
                financial_reasons.append(f"Moderate dividend yield: {dividend_yield*100:.2f}%")
            else:
                financial_reasons.append(f"Low dividend yield: {dividend_yield*100:.2f}%")
            
            if payout_ratio:
                if payout_ratio > 0.8:
                    financial_score -= 0.25
                    financial_reasons.append(f"High payout ratio: {payout_ratio*100:.1f}% (potential dividend sustainability concerns)")
                elif payout_ratio < 0.4:
                    financial_score += 0.25
                    financial_reasons.append(f"Conservative payout ratio: {payout_ratio*100:.1f}% (room for dividend growth)")
                else:
                    financial_reasons.append(f"Moderate payout ratio: {payout_ratio*100:.1f}%")
        
        # Add financial score
        scores.append(financial_score)
        fundamental_metrics.extend(financial_reasons)
        
        # Company Size & Market Cap
        market_cap = info.get('marketCap')
        if market_cap:
            market_cap_billions = market_cap / 1_000_000_000
            if market_cap_billions > 200:
                market_sentiment.append(f"Mega-cap company: ${market_cap_billions:.1f}B market cap")
                risk_assessment.append("Lower volatility typical of mega-cap stocks")
            elif market_cap_billions > 10:
                market_sentiment.append(f"Large-cap company: ${market_cap_billions:.1f}B market cap")
                risk_assessment.append("Moderate volatility typical of large-cap stocks")
            elif market_cap_billions > 2:
                market_sentiment.append(f"Mid-cap company: ${market_cap_billions:.1f}B market cap")
                risk_assessment.append("Increased volatility typical of mid-cap stocks")
            elif market_cap_billions > 0.3:
                market_sentiment.append(f"Small-cap company: ${market_cap_billions:.1f}B market cap")
                risk_assessment.append("High volatility typical of small-cap stocks")
            else:
                market_sentiment.append(f"Micro-cap company: ${market_cap_billions:.1f}B market cap")
                risk_assessment.append("Very high volatility typical of micro-cap stocks")
        
        # Industry/Sector Information
        sector = info.get('sector')
        industry = info.get('industry')
        if sector and industry:
            market_sentiment.append(f"Sector: {sector} | Industry: {industry}")
            
            # Add industry-specific context if available
            if "Technology" in sector:
                industry_comparison.append("Tech sector showing above-average growth but with high valuations")
            elif "Healthcare" in sector:
                industry_comparison.append("Healthcare sector showing defensive characteristics amid market volatility")
            elif "Energy" in sector:
                industry_comparison.append("Energy sector performance closely tied to commodity prices and global demand")
            elif "Financial" in sector:
                industry_comparison.append("Financial sector sensitive to interest rate changes and economic cycles")
        
    except Exception as e:
        fundamental_metrics.append(f"Could not complete fundamental analysis: {e}")
    
    # 3. Analyst Recommendations
    try:
        recommendations = data['recommendations']
        if not recommendations.empty:
            # Print columns for debugging
            column_info = f"Available columns: {list(recommendations.columns)}"
            print(column_info)
            
            # Get recent recommendations (last 10)
            recent_recs = recommendations.tail(10)
            
            # Try different possible column names for recommendations
            if 'To Grade' in recent_recs.columns:
                grade_column = 'To Grade'
            elif 'toGrade' in recent_recs.columns:
                grade_column = 'toGrade'
            elif 'grade' in recent_recs.columns:
                grade_column = 'grade'
            elif 'action' in recent_recs.columns:
                grade_column = 'action'
            else:
                # Try to find a suitable column for grades
                potential_columns = [col for col in recent_recs.columns if any(
                    keyword in col.lower() for keyword in ['grade', 'recommendation', 'action', 'rating'])]
                
                if potential_columns:
                    grade_column = potential_columns[0]  # Use the first matching column
                    print(f"Using column: {grade_column}")
                else:
                    # If no suitable column found, try to analyze the first string column
                    string_columns = [col for col in recent_recs.columns 
                                     if recent_recs[col].dtype == 'object']
                    
                    if string_columns:
                        grade_column = string_columns[0]
                        print(f"Using first string column: {grade_column}")
                    else:
                        raise ValueError("No suitable column for analyst recommendations found")
            
            # Enhanced buy/hold/sell analysis
            grades = recent_recs[grade_column].astype(str).str.lower()
            # Print sample values for debugging
            print(f"Sample grades: {grades.tolist()[:3]}")
            
            strong_buys = sum(grades.str.contains('strong buy|top pick|outperform|overweight|conviction buy'))
            buys = sum(grades.str.contains('buy|outperform|overweight')) - strong_buys
            holds = sum(grades.str.contains('hold|neutral|market perform|sector perform'))
            sells = sum(grades.str.contains('sell|underperform|underweight'))
            strong_sells = sum(grades.str.contains('strong sell|conviction sell'))
            
            total_ratings = strong_buys + buys + holds + sells + strong_sells
            
            if total_ratings > 0:  # Only add to score if we found something meaningful
                analyst_score = 0
                
                # Calculate weighted score
                weighted_score = (2*strong_buys + buys - holds - 2*strong_sells - sells) / total_ratings
                
                if weighted_score > 0.8:
                    analyst_score = 1
                    market_sentiment.append(f"Very bullish analyst consensus: {strong_buys} strong buys, {buys} buys, {holds} holds, {sells} sells, {strong_sells} strong sells")
                elif weighted_score > 0.4:
                    analyst_score = 0.5
                    market_sentiment.append(f"Bullish analyst consensus: {strong_buys} strong buys, {buys} buys, {holds} holds, {sells} sells, {strong_sells} strong sells")
                elif weighted_score > 0.1:
                    analyst_score = 0.25
                    market_sentiment.append(f"Mildly bullish analyst consensus: {strong_buys} strong buys, {buys} buys, {holds} holds, {sells} sells, {strong_sells} strong sells")
                elif weighted_score < -0.8:
                    analyst_score = -1
                    market_sentiment.append(f"Very bearish analyst consensus: {strong_buys} strong buys, {buys} buys, {holds} holds, {sells} sells, {strong_sells} strong sells")
                elif weighted_score < -0.4:
                    analyst_score = -0.5
                    market_sentiment.append(f"Bearish analyst consensus: {strong_buys} strong buys, {buys} buys, {holds} holds, {sells} sells, {strong_sells} strong sells")
                elif weighted_score < -0.1:
                    analyst_score = -0.25
                    market_sentiment.append(f"Mildly bearish analyst consensus: {strong_buys} strong buys, {buys} buys, {holds} holds, {sells} sells, {strong_sells} strong sells")
                else:
                    analyst_score = 0
                    market_sentiment.append(f"Neutral analyst consensus: {strong_buys} strong buys, {buys} buys, {holds} holds, {sells} sells, {strong_sells} strong sells")
                
                scores.append(analyst_score)
                
                # Target Price Analysis
                if 'Price Target' in recent_recs.columns:
                    target_prices = recent_recs['Price Target'].dropna()
                    if not target_prices.empty:
                        avg_target = target_prices.mean()
                        min_target = target_prices.min()
                        max_target = target_prices.max()
                        upside = ((avg_target / current_price) - 1) * 100 if current_price else 0
                        
                        if upside > 20:
                            scores.append(0.5)
                            market_sentiment.append(f"Strong upside potential: ${avg_target:.2f} avg target ({upside:.1f}% upside), range: ${min_target:.2f}-${max_target:.2f}")
                        elif upside > 10:
                            scores.append(0.25)
                            market_sentiment.append(f"Moderate upside potential: ${avg_target:.2f} avg target ({upside:.1f}% upside), range: ${min_target:.2f}-${max_target:.2f}")
                        elif upside < -20:
                            scores.append(-0.5)
                            market_sentiment.append(f"Significant downside risk: ${avg_target:.2f} avg target ({upside:.1f}% downside), range: ${min_target:.2f}-${max_target:.2f}")
                        elif upside < -10:
                            scores.append(-0.25)
                            market_sentiment.append(f"Moderate downside risk: ${avg_target:.2f} avg target ({upside:.1f}% downside), range: ${min_target:.2f}-${max_target:.2f}")
                        else:
                            market_sentiment.append(f"Price near analyst targets: ${avg_target:.2f} avg target ({upside:.1f}% from current), range: ${min_target:.2f}-${max_target:.2f}")
    except Exception as e:
        market_sentiment.append(f"Could not analyze analyst recommendations: {str(e)}")
    
    # 4. Institutional Ownership
    try:
        institutional_holders = data.get('institutional_holders')
        if institutional_holders is not None and not institutional_holders.empty:
            total_shares_held = institutional_holders['Shares'].sum()
            outstanding_shares = info.get('sharesOutstanding', 0)
            
            if outstanding_shares > 0:
                inst_ownership_pct = (total_shares_held / outstanding_shares) * 100
                
                if inst_ownership_pct > 80:
                    scores.append(0.5)
                    market_sentiment.append(f"Very high institutional ownership: {inst_ownership_pct:.1f}% (strong institutional confidence)")
                elif inst_ownership_pct > 60:
                    scores.append(0.25)
                    market_sentiment.append(f"High institutional ownership: {inst_ownership_pct:.1f}% (positive institutional interest)")
                elif inst_ownership_pct < 20:
                    market_sentiment.append(f"Low institutional ownership: {inst_ownership_pct:.1f}% (limited institutional interest)")
                else:
                    market_sentiment.append(f"Moderate institutional ownership: {inst_ownership_pct:.1f}%")
    except Exception as e:
        market_sentiment.append(f"Could not analyze institutional ownership: {str(e)}")
    
    # 5. Earnings Analysis
    try:
        earnings = data.get('earnings')
        if earnings is not None and not earnings.empty:
            recent_quarters = min(4, len(earnings))
            if recent_quarters > 0:
                earnings_beat_count = 0
                earnings_miss_count = 0
                
                for i in range(recent_quarters):
                    if earnings.iloc[i]['Actual'] > earnings.iloc[i]['Estimate']:
                        earnings_beat_count += 1
                    elif earnings.iloc[i]['Actual'] < earnings.iloc[i]['Estimate']:
                        earnings_miss_count += 1
                
                if earnings_beat_count == recent_quarters:
                    scores.append(0.5)
                    market_sentiment.append(f"Strong earnings performance: Beat expectations in all {recent_quarters} recent quarters")
                elif earnings_beat_count >= recent_quarters - 1:
                    scores.append(0.25)
                    market_sentiment.append(f"Solid earnings performance: Beat expectations in {earnings_beat_count} of {recent_quarters} recent quarters")
                elif earnings_miss_count >= recent_quarters - 1:
                    scores.append(-0.25)
                    market_sentiment.append(f"Weak earnings performance: Missed expectations in {earnings_miss_count} of {recent_quarters} recent quarters")
                else:
                    market_sentiment.append(f"Mixed earnings performance: Beat in {earnings_beat_count}, missed in {earnings_miss_count} of {recent_quarters} recent quarters")
    except Exception as e:
        market_sentiment.append(f"Could not analyze earnings history: {str(e)}")
    
    # 6. Risk Assessment
    if not risk_assessment:
        risk_assessment.append("Standard market risk exposure")
        
    beta = info.get('beta')
    if beta:
        if beta > 1.5:
            risk_assessment.append(f"High market sensitivity: Beta of {beta:.2f} (50%+ more volatile than market)")
        elif beta < 0.7:
            risk_assessment.append(f"Low market sensitivity: Beta of {beta:.2f} (30%+ less volatile than market)")
        else:
            risk_assessment.append(f"Average market sensitivity: Beta of {beta:.2f}")
    
    # Calculate the final score with weighted categories
    category_weights = {
        'technical': 0.35,  # 35% weight for technical analysis
        'fundamental': 0.35,  # 35% weight for fundamentals
        'sentiment': 0.30,   # 30% weight for market sentiment
    }
    
    # Group scores by category
    technical_scores = scores[:len(technical_indicators)]
    fundamental_scores = scores[len(technical_indicators):len(technical_indicators)+len(fundamental_metrics)]
    sentiment_scores = scores[len(technical_indicators)+len(fundamental_metrics):]
    
    # Calculate category averages
    technical_avg = sum(technical_scores) / len(technical_scores) if technical_scores else 0
    fundamental_avg = sum(fundamental_scores) / len(fundamental_scores) if fundamental_scores else 0
    sentiment_avg = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    
    # Calculate weighted final score
    weighted_score = (
        technical_avg * category_weights['technical'] +
        fundamental_avg * category_weights['fundamental'] +
        sentiment_avg * category_weights['sentiment']
    )
    
    # Add all reasons together
    all_reasons = []
    if technical_indicators:
        all_reasons.extend(technical_indicators)
    if fundamental_metrics:
        all_reasons.extend(fundamental_metrics)
    if market_sentiment:
        all_reasons.extend(market_sentiment)
    if industry_comparison:
        all_reasons.extend(industry_comparison)
    if risk_assessment:
        all_reasons.extend(risk_assessment)
    
    # Determine recommendation
    confidence_level = min(abs(weighted_score) * 100, 95)  # Scale confidence with score magnitude, cap at 95%
    
    if weighted_score >= 0.5:
        recommendation = BUY
        confidence = f"Strong Buy ({confidence_level:.0f}% confidence)"
    elif weighted_score >= 0.2:
        recommendation = BUY
        confidence = f"Moderate Buy ({confidence_level:.0f}% confidence)"
    elif weighted_score <= -0.5:
        recommendation = SELL
        confidence = f"Strong Sell ({confidence_level:.0f}% confidence)"
    elif weighted_score <= -0.2:
        recommendation = SELL
        confidence = f"Moderate Sell ({confidence_level:.0f}% confidence)"
    else:
        recommendation = HOLD
        confidence = f"Hold ({confidence_level:.0f}% confidence)"
    
    # Generate price targets
    target_price = None
    target_range = None
    
    try:
        if current_price:
            if recommendation == BUY:
                upside = 0.1 + (weighted_score * 0.1)  # 10% minimum + score-based addition
                target_price = current_price * (1 + upside)
                target_range = (current_price * (1 + upside * 0.7), current_price * (1 + upside * 1.3))
            elif recommendation == SELL:
                downside = 0.1 + (abs(weighted_score) * 0.1)  # 10% minimum + score-based addition
                target_price = current_price * (1 - downside)
                target_range = (current_price * (1 - downside * 1.3), current_price * (1 - downside * 0.7))
            else:
                # For HOLD, smaller range around current price
                target_price = current_price
                target_range = (current_price * 0.95, current_price * 1.05)
    except Exception as e:
        print(f"Error calculating price targets: {e}")
    
    # Format price as string
    price_str = f"{current_price:.2f}" if current_price else "Unknown"
    
    # Format target price
    target_price_str = None
    target_range_str = None
    
    if target_price and target_range:
        target_price_str = f"{target_price:.2f}"
        target_range_str = f"{target_range[0]:.2f}-{target_range[1]:.2f}"
    
    # Compile final result
    result = {
        'ticker': ticker,
        'price': price_str,
        'recommendation': recommendation,
        'confidence': confidence,
        'score': round(weighted_score, 2),
        'reasons': all_reasons,
        'technical_score': round(technical_avg, 2),
        'fundamental_score': round(fundamental_avg, 2),
        'sentiment_score': round(sentiment_avg, 2),
        'target_price': target_price_str,
        'target_range': target_range_str,
        'risk_level': 'High' if len(risk_assessment) >= 3 else 'Moderate' if len(risk_assessment) >= 2 else 'Low'
    }
    
    return result

def print_analysis(result: Dict[str, Any]) -> None:
    """Print analysis in a concise format"""
    ticker = result['ticker']
    price = result['price']
    recommendation = result['recommendation']
    confidence = result.get('confidence', '')
    score = result['score']
    technical_score = result.get('technical_score', 'N/A')
    fundamental_score = result.get('fundamental_score', 'N/A')
    sentiment_score = result.get('sentiment_score', 'N/A')
    target_price = result.get('target_price', 'N/A')
    target_range = result.get('target_range', 'N/A')
    risk_level = result.get('risk_level', 'Unknown')
    reasons = result['reasons']
    
    print(f"======== {ticker} (${price}) ========")
    print(f"RECOMMENDATION: {recommendation} - {confidence}")
    print(f"OVERALL SCORE: {score}")
    print(f"CATEGORY SCORES: Technical: {technical_score} | Fundamental: {fundamental_score} | Sentiment: {sentiment_score}")
    
    if target_price != 'N/A':
        print(f"PRICE TARGET: ${target_price} (Range: ${target_range})")
    
    print(f"RISK LEVEL: {risk_level}")
    print("\nANALYSIS FACTORS:")
    
    # Group reasons by category for better readability
    technical_indicators = [r for r in reasons if any(term in r for term in ['trend', 'RSI', 'MACD', 'Bollinger', 'volume', 'Moving average'])]
    fundamental_metrics = [r for r in reasons if any(term in r for term in ['P/E', 'revenue', 'profit', 'margin', 'debt', 'dividend', 'ratio'])]
    market_sentiment = [r for r in reasons if any(term in r for term in ['analyst', 'consensus', 'institutional', 'earnings', 'cap', 'Sector', 'Industry'])]
    risk_factors = [r for r in reasons if any(term in r for term in ['risk', 'volatility', 'sensitivity'])]
    other_factors = [r for r in reasons if r not in technical_indicators and r not in fundamental_metrics and r not in market_sentiment and r not in risk_factors]
    
    if technical_indicators:
        print("\nTECHNICAL INDICATORS:")
        for reason in technical_indicators:
            print(f"• {reason}")
    
    if fundamental_metrics:
        print("\nFUNDAMENTAL METRICS:")
        for reason in fundamental_metrics:
            print(f"• {reason}")
    
    if market_sentiment:
        print("\nMARKET SENTIMENT:")
        for reason in market_sentiment:
            print(f"• {reason}")
    
    if risk_factors:
        print("\nRISK ASSESSMENT:")
        for reason in risk_factors:
            print(f"• {reason}")
    
    if other_factors:
        print("\nADDITIONAL FACTORS:")
        for reason in other_factors:
            print(f"• {reason}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        print(f"Analyzing {ticker}...")
        result = analyze_stock(ticker)
        print_analysis(result)
    else:
        print("Usage: python yahoo_small.py TICKER")
        print("Example: python yahoo_small.py AAPL")