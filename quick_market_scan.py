import os
import time
import asyncio
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization
from collections import defaultdict
import numpy as np
from clients import KalshiHttpClient, Environment
from exa_py import Exa
from openai import OpenAI
load_dotenv()

MAX_MARKETS = 10
MARKETS_PER_REQUEST = 10
EXA_KEY = os.getenv("EXA_KEY")
KALSHI_KEY = os.getenv("KALSHI_KEY")
PERPLEXITY_KEY = os.getenv("PERPLEXITY_KEY")
N_RESEARCHES = 1

def make_queries(title):
    # Clean the title to make it more search-friendly
    # Remove unusual characters that might cause search issues
    clean_title = ''.join(c if c.isalnum() or c in [' ', '.', ',', '?', '!'] else ' ' for c in title)
    
    # Extract key entities or topics from the market title
    # This helps create more focused search queries
    market_keywords = []
    
    # Extract names of entities (people, places, events)
    if "vs" in title:
        # Handle sports matches
        parts = title.split("vs")
        if len(parts) > 1:
            market_keywords.extend([parts[0].strip(), parts[1].strip()])
    
    # Extract dates if present
    date_markers = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    for marker in date_markers:
        if marker in title:
            # Find any date references like "25NOV10"
            import re
            date_refs = re.findall(r'\d+' + marker + r'\d+', title)
            if date_refs:
                market_keywords.append(date_refs[0])
    
    # Create search queries based on the market information
    # For markets about specific mentions, generalize the search
    if "vs" in title and any(sport in title.lower() for sport in ["tennis", "match", "game", "fight", "bout"]):
        # Sports-focused queries
        players = [kw for kw in market_keywords if len(kw.split()) <= 3]  # Player names are usually short
        queries = [
            f"{' '.join(players)} upcoming match odds",
            f"{' '.join(players)} prediction betting",
            f"{players[0] if players else clean_title} recent performance",
            f"{' '.join(players)} match analysis"
        ]
    else:
        # General market queries
        queries = [
            f"{clean_title} prediction market analysis",
            f"{clean_title} recent news events",
            f"{clean_title} market sentiment",
        ]
    return queries

async def search_market_info(exa_client, market):
    """Search for information about a market using ExaAI."""
    ticker = market['ticker']
    title = market['title']
    
    queries = make_queries(title)
    print(queries)

    all_results = []
    for query in queries:
        try:
            # Search with Exa - include Twitter/X and social media results
            print(f"Searching for: '{query}'...")
            # Use additional parameters based on Exa API documentation
            search_results = exa_client.answer(
                query,
                # include_domains=["twitter.com", "x.com"],
                # content={"highlights": True, "livecrawl": True}
            )
            print(search_results)
            # Safely check if we have results
            if search_results and hasattr(search_results, 'results') and search_results.results:
                safe_results = []
                for result in search_results.results:
                    try:
                        # Check if all required attributes exist and have proper types
                        if (hasattr(result, 'title') and result.title and 
                            hasattr(result, 'url') and result.url and 
                            hasattr(result, 'text') and result.text):
                            
                            # Get content from multiple possible fields with fallbacks
                            text_content = ""
                            
                            # Try multiple fields as fallbacks for content
                            if hasattr(result, 'text') and result.text:
                                text_content = result.text
                            elif hasattr(result, 'highlights') and result.highlights and len(result.highlights) > 0:
                                text_content = " ".join(result.highlights)
                            elif hasattr(result, 'summary') and result.summary:
                                text_content = result.summary
                            
                            # If we still don't have content, create a placeholder
                            if not text_content:
                                text_content = f"[No content available for {result.title}]"
                            
                            # Create the result object with fallbacks for all fields
                            safe_results.append({
                                'title': result.title if result.title else "Untitled",
                                'url': result.url,
                                'content': text_content[:300] + "..." if len(text_content) > 300 else text_content,
                                'source': result.url.split('/')[2] if '/' in result.url else 'unknown',
                                'date': getattr(result, 'published_date', 'Unknown date'),
                                'score': getattr(result, 'score', None),
                                'has_highlights': hasattr(result, 'highlights') and bool(result.highlights)
                            })
                    except Exception as e:
                        print(f"Error processing result from '{query}': {e}")
                        continue
                
                all_results.extend(safe_results)
                print(f"Found {len(safe_results)} valid results for '{query}'")
            else:
                print(f"No valid results found for '{query}'")
                
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
    
    # Process results to extract insights
    insights = {
        'recent_events': [],
        'sentiment': 'Unknown',
        'potential_mispricing': False,
        'mispricing_reason': '',
        'key_sources': []
    }
    
    # Extract key information from search results
    sentiment_keywords = {
        'positive': ['bullish', 'optimistic', 'confident', 'positive', 'surge', 'rise', 'gain', 'likely', 'will', 'expected'],
        'negative': ['bearish', 'pessimistic', 'concerned', 'negative', 'drop', 'fall', 'decline', 'risk', 'unlikely', 'won\'t', 'not expected']
    }
    
    # Handle case where we have no results at all
    if not all_results:
        print("No search results found. Adding fallback analysis.")
        # Add fallback analysis based on the market's title and price
        if 'White House' in market['title'] or 'Press Secretary' in market['title']:
            insights['key_sources'].append({
                'domain': 'kalshi.com',
                'url': f"https://kalshi.com/markets/{market['ticker']}",
                'title': f"Kalshi Market: {market['ticker']}"
            })
            
            # For "will mention X" markets with low prices
            if market['status'] == 'LOW':
                insights['sentiment'] = 'Negative'
                insights['recent_events'].append("No recent mentions or plans detected.")
            else:
                insights['sentiment'] = 'Positive'
                insights['recent_events'].append("High likelihood based on current events.")
            
        return {
            'raw_results': [],
            'insights': insights
        }
    
    for result in all_results:
        # Add to sources
        if result['source'] not in [s.get('domain') for s in insights['key_sources']]:
            insights['key_sources'].append({
                'domain': result['source'],
                'url': result['url'],
                'title': result['title']
            })
        
        # Analyze sentiment
        content_lower = result['content'].lower()
        pos_count = sum(content_lower.count(word) for word in sentiment_keywords['positive'])
        neg_count = sum(content_lower.count(word) for word in sentiment_keywords['negative'])
        
        if pos_count > neg_count:
            result_sentiment = 'Positive'
        elif neg_count > pos_count:
            result_sentiment = 'Negative'
        else:
            result_sentiment = 'Neutral'
            
        # Add sentiment to each result
        result['sentiment'] = result_sentiment
    
    # Determine overall sentiment
    sentiment_counts = {
        'Positive': len([r for r in all_results if r.get('sentiment') == 'Positive']),
        'Negative': len([r for r in all_results if r.get('sentiment') == 'Negative']),
        'Neutral': len([r for r in all_results if r.get('sentiment') == 'Neutral'])
    }
    
    if sentiment_counts['Positive'] > sentiment_counts['Negative']:
        insights['sentiment'] = 'Positive'
    elif sentiment_counts['Negative'] > sentiment_counts['Positive']:
        insights['sentiment'] = 'Negative'
    else:
        insights['sentiment'] = 'Neutral'
    
    # Check for potential mispricing based on sentiment vs price
    if (market['status'] == 'LOW' and insights['sentiment'] == 'Positive') or \
       (market['status'] == 'HIGH' and insights['sentiment'] == 'Negative'):
        insights['potential_mispricing'] = True
        insights['mispricing_reason'] = f"Market priced at {'LOW' if market['status'] == 'LOW' else 'HIGH'} but sentiment is {insights['sentiment']}"
    
    return {
        'raw_results': all_results[:5],  # Return top 5 results
        'insights': insights
    }

async def scan_markets():
    """Quickly scan all markets to find those with prices below 10 or above 90."""
    # Load environment variables
    load_dotenv()
    
    if not EXA_KEY:
        print("Warning: EXA_KEY not found in environment variables. Market analysis will be limited.")
        exa_client = None
    else:
        exa_client = Exa(api_key=EXA_KEY)
        # Test search to verify API key works
    
    # Set up environment and credentials
    env = Environment.PROD  # toggle environment here
    KEYID = ""
    KEYFILE = "pem.key"
    
    try:
        with open(KEYFILE, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None  # Provide the password if your key is encrypted
            )
    except FileNotFoundError:
        raise FileNotFoundError(f"Private key file not found at {KEYFILE}")
    except Exception as e:
        raise Exception(f"Error loading private key: {str(e)}")

    # Initialize the HTTP client
    http_client = KalshiHttpClient(
        key_id=KEYID,
        private_key=private_key,
        environment=env
    )

    try:
        # Get all markets using pagination
        all_markets = []
        cursor = None
        limit = MARKETS_PER_REQUEST  # Maximum limit per request
        
        print("Fetching markets...")
        
        while True:
            params = {'limit': limit}
            if cursor:
                params['cursor'] = cursor
            
            # Make API request
            markets_response = http_client.get(http_client.markets_url, params=params)
            markets = markets_response.get('markets', [])
            
            if not markets:
                break
            
            all_markets.extend(markets)
            if len(all_markets) >= MAX_MARKETS:
                break
            print(f"Fetched {len(all_markets)} markets so far...")
            
            # Check if there are more markets to fetch
            cursor = markets_response.get('cursor')
            if not cursor:
                break
        
        print(f"Total markets found: {len(all_markets)}")
        
        # Filter markets with prices below 10 or above 90
        extreme_markets = []
        
        for market in all_markets:
            ticker = market.get('ticker')
            yes_price = market.get('yes_bid')  # Use yes_bid as an approximation of the current price
            
            if yes_price is not None and (10 <= yes_price <= 25 or 75 <= yes_price <= 90):
                # Check for recent trades to estimate volatility
                trades = http_client.get_trades(ticker=ticker, limit=10)
                
                # Calculate a simple volatility measure
                prices = [trade.get('yes_price', 0) for trade in trades.get('trades', [])]
                volatility = np.std(prices) if prices else 0
                
                # Get market details
                title = market.get('title', '')
                rules = market.get('rules_primary', 'No rules available')
                
                extreme_markets.append({
                    'ticker': ticker,
                    'title': title,
                    'price': yes_price,
                    'volatility': volatility,
                    'status': 'HIGH' if yes_price >= 90 else 'LOW',
                    'rules': rules
                })
        
        # Sort by volatility
        extreme_markets.sort(key=lambda x: x['volatility'], reverse=True)
        
        # Print results
        print(f"\n===== Markets with Extreme Prices (≥90 or ≤10) =====")
        print(f"{'Market Ticker':<15} {'Price':<8} {'Vol':<8} {'Status':<6} {'Market Question/Title'}")
        print("-" * 100)
        
        for market in extreme_markets[:15]:  # Display top 15
            # Truncate title if needed
            display_title = market['title'][:65] + "..." if len(market['title']) > 65 else market['title']
            
            print(f"{market['ticker']:<15} {market['price']:<8} {market['volatility']:<8.2f} {market['status']:<6} {display_title}")
        
        print(f"\nTotal extreme markets found: {len(extreme_markets)}")
        
        # Ask if user wants to research markets with ExaAI
        if extreme_markets and exa_client:
            print("\nAnalyzing volatile markets with ExaAI...")
            
            # Create market insights dictionary
            market_insights = {}
            
            for idx, market in enumerate(extreme_markets[:N_RESEARCHES]):  # Limit to top 5 most volatile
                print(f"Researching {idx+1}/5: {market['ticker']} - {market['title']}")
                # try:
                insights = await search_market_info(exa_client, market)
                market_insights[market['ticker']] = insights
                # except Exception as e:
                #     print(f"Error analyzing market {market['ticker']}: {e}")
                #     # Create a fallback insight object for this market
                #     market_insights[market['ticker']] = {
                #         'raw_results': [],
                #         'insights': {
                #             'recent_events': ["Unable to retrieve data from ExaAI"],
                #             'sentiment': 'Unknown',
                #             'potential_mispricing': False,
                #             'mispricing_reason': '',
                #             'key_sources': [{
                #                 'domain': 'kalshi.com',
                #                 'url': f"https://kalshi.com/markets/{market['ticker']}",
                #                 'title': f"Kalshi Market: {market['ticker']}"
                #             }]
                #         }
                #     }
                
                # Sleep briefly to avoid rate limits
                await asyncio.sleep(1)
            
            # Display analysis results
            print("\n===== Market Analysis Results =====")
            
            for ticker, data in market_insights.items():
                market = next((m for m in extreme_markets if m['ticker'] == ticker), None)
                if not market:
                    continue
                    
                insights = data['insights']
                
                print(f"\n--- {ticker}: {market['title']} ---")
                print(f"Price: {market['price']} ({market['status']})")
                print(f"Volatility: {market['volatility']:.2f}")
                print(f"Sentiment Analysis: {insights['sentiment']}")
                
                if insights['potential_mispricing']:
                    print(f"⚠️ POTENTIAL MISPRICING DETECTED: {insights['mispricing_reason']}")
                
                print("\nKey Sources:")
                for source in insights['key_sources'][:3]:
                    print(f"- {source['domain']}: {source['title']}")
                    print(f"  {source['url']}")
                
                print("\nRelevant Excerpts:")
                if data['raw_results']:
                    for result in data['raw_results'][:2]:
                        print(f"- {result['title']} ({result['sentiment']})")
                        print(f"  {result['content'][:150]}...")
                else:
                    print("  No detailed excerpts available - limited data for this market.")
                    
                # Add market-specific analysis based on market title/rules when no search results
                # if not data['raw_results']:
                #     print("\nMarket-Specific Analysis:")
                #     if "White House Press Secretary" in market['title']:
                #         print("  This is a 'mention market' about specific words in a press briefing.")
                #         if market['status'] == 'LOW':
                #             print("  The low price suggests this term is unlikely to be mentioned.")
                #             print("  Check for: 1) How unusual/specific the term is, 2) Current news relevance")
                #         else:
                #             print("  The high price suggests this term is likely to be mentioned.")
                #             print("  Check for: 1) Term's relevance to current news cycle, 2) Previous mentions")
                #     elif "by" in market['title'] and any(month in market['title'] for month in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]):
                #         print("  This is a date-specific prediction market.")
                #         print("  Check recent developments and timeline feasibility for potential mispricing.")
        
        # Ask if user wants to see details for any market
        # if extreme_markets:
        #     print("\nEnter a ticker for more details (or press Enter to exit):")
        #     ticker_input = input().strip()
            
        #     if ticker_input:
        #         # Find the market with the given ticker
        #         for market in extreme_markets:
        #             if market['ticker'] == ticker_input:
        #                 print("\n===== Market Details =====")
        #                 print(f"Ticker: {market['ticker']}")
        #                 print(f"Title: {market['title']}")
        #                 print(f"Current Price: {market['price']}")
        #                 print(f"Volatility: {market['volatility']}")
        #                 print(f"Status: {market['status']}")
        #                 print("\nRules:")
        #                 print(market['rules'])
                        
        #                 # Show detailed analysis if available
        #                 if exa_client and ticker_input in market_insights:
        #                     insights = market_insights[ticker_input]['insights']
        #                     raw_results = market_insights[ticker_input]['raw_results']
                            
        #                     print("\n===== ExaAI Analysis =====")
        #                     print(f"Overall Sentiment: {insights['sentiment']}")
                            
        #                     if insights['potential_mispricing']:
        #                         print(f"\n⚠️ POTENTIAL MISPRICING DETECTED:")
        #                         print(f"  {insights['mispricing_reason']}")
                            
        #                     print("\nDetailed Search Results:")
        #                     for idx, result in enumerate(raw_results):
        #                         print(f"\n{idx+1}. {result['title']}")
        #                         print(f"   Source: {result['source']} | Sentiment: {result['sentiment']}")
        #                         print(f"   URL: {result['url']}")
        #                         print(f"   Excerpt: {result['content']}")
        #                 break
        #         else:
        #             print(f"No market found with ticker {ticker_input}")
        
    except Exception as e:
        print(f"Error scanning markets: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(scan_markets())
