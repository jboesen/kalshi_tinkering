import os
import asyncio
import re
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization
from clients import KalshiHttpClient, Environment
from openai import AsyncOpenAI  # Change to AsyncOpenAI for async support

# Load environment variables
load_dotenv()

# Constants
MAX_MARKETS = 100_000
MARKETS_PER_REQUEST = 1000
PERPLEXITY_KEY = os.getenv("PERPLEXITY_KEY")
N_RESEARCHES = 50  # Number of markets to research
MAX_CONCURRENT_REQUESTS = 5  # Limit concurrent API calls to avoid rate limits

def construct_market_url(market):
    """
    Construct the proper URL for a Kalshi market based on available data
    """
    ticker = market.get('ticker')
    event_ticker = market.get('event_ticker')
    category = market.get('category')
    
    # If we have both an event_ticker and category, use the full path
    if event_ticker and category:
        return f"https://kalshi.com/events/{category}/{event_ticker}/markets/{ticker}"
    # If we have just an event_ticker
    elif event_ticker:
        return f"https://kalshi.com/events/{event_ticker}/markets/{ticker}"
    # Default case - just use the ticker
    else:
        return f"https://kalshi.com/markets/{ticker}"

async def analyze_market(perplexity_client, market):
    """Get analysis from Perplexity and extract conclusion and Fermi estimate tags"""
    ticker = market['ticker']
    title = market['title']
    price = market['price']
    
    print(f"Researching: {ticker} - {title}")
    
    try:
        # Simple query based on market title
        messages = [
            {
                "role": "user",
                "content": 
                f"""Decide if there is sufficient evidence to conclude that {title} will or won't happen with high certainty. 
                The market is currently priced at {price}.

                First, break this question down into its component parts, identifying what is knowable and what remains uncertain; for short-term events, prioritize short-term trends. 

                For each component:
                - Provide a 90% confidence interval rather than a point estimate
                - Identify which reference class this belongs to and what the base rate is
                - Explicitly state major assumptions being made

                Then, reason through the evidence using Fermi estimation, being careful to:
                - Avoid both overconfidence and underconfidence
                - Consider both inside and outside views
                - Update beliefs appropriately as you work through the evidence

                Finally, conclude with 
                - <conclusion>MISPRICED</conclusion> or <conclusion>UNCERTAIN</conclusion>
                - your Fermi probability estimate (in decimal format) in <fermiEstimate></fermiEstimate> tags. 
                - A calibration score from 1-10 indicating your confidence in this probability estimate being well-calibrated in <calibration></calibration> tags"""
            },
        ]
        
        # Request analysis from Perplexity
        response = await perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=messages,
        )
        response_text = response.choices[0].message.content
        print(f"Received response from Perplexity for {ticker}")
        
        # Extract conclusion and Fermi estimate tags
        conclusion_match = re.search(r'<conclusion>(.*?)</conclusion>', response_text)
        fermi_match = re.search(r'<fermiEstimate>(.*?)</fermiEstimate>', response_text)
        calibration_match = re.search(r'<calibration>(.*?)</calibration>', response_text)
        
        conclusion = conclusion_match.group(1) if conclusion_match else "UNKNOWN"
        fermi_estimate = None
        calibration_score = 0
        
        if fermi_match:
            try:
                fermi_estimate = float(fermi_match.group(1)) 
            except ValueError:
                print(f"Could not convert Fermi estimate to float: {fermi_match.group(1)}")
        
        if calibration_match:
            try:
                calibration_score = int(calibration_match.group(1))
            except ValueError:
                print(f"Could not convert calibration score to int: {calibration_match.group(1)}")
        
        # Calculate price difference (in percentage points)
        fermi_price = fermi_estimate * 100 if fermi_estimate is not None else None
        price_difference = abs(price - fermi_price) if fermi_price is not None else 0
        
        # Create market link using the proper URL construction function
        market_link = construct_market_url(market)
        
        return {
            'ticker': ticker,
            'title': title,
            'price': price,
            'conclusion': conclusion,
            'fermi_estimate': fermi_estimate,
            'fermi_price': fermi_price,
            'price_difference': price_difference,
            'calibration_score': calibration_score,
            'market_link': market_link
        }
        
    except Exception as e:
        print(f"Error analyzing market {ticker}: {e}")
        return None

async def process_market_batch(perplexity_client, markets):
    """Process a batch of markets in parallel using semaphore to limit concurrency"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def analyze_with_semaphore(market):
        async with semaphore:
            return await analyze_market(perplexity_client, market)
    
    # Create tasks for all markets in the batch
    tasks = [analyze_with_semaphore(market) for market in markets]
    
    # Wait for all tasks to complete and collect results
    results = await asyncio.gather(*tasks)
    return [result for result in results if result is not None]

async def scan_markets():
    """Scan markets, analyze them in parallel, and sort by mispricing"""
    # Initialize Perplexity client
    if not PERPLEXITY_KEY:
        print("Error: PERPLEXITY_KEY not found. Exiting.")
        return
    
    perplexity_client = AsyncOpenAI(  # Use AsyncOpenAI instead
        api_key=PERPLEXITY_KEY,
        base_url="https://api.perplexity.ai"
    )
    
    # Set up Kalshi client
    env = Environment.PROD
    KEYID = ""
    KEYFILE = "pem.key"
    
    try:
        with open(KEYFILE, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None
            )
    except FileNotFoundError:
        raise FileNotFoundError(f"Private key file not found at {KEYFILE}")
    
    http_client = KalshiHttpClient(
        key_id=KEYID,
        private_key=private_key,
        environment=env
    )
    
    # Get markets from Kalshi
    all_markets = []
    cursor = None
    
    print("Fetching markets...")
    
    while len(all_markets) < MAX_MARKETS:
        params = {'limit': MARKETS_PER_REQUEST}
        if cursor:
            params['cursor'] = cursor
        
        markets_response = http_client.get(http_client.markets_url, params=params)
        markets = markets_response.get('markets', [])
        
        if not markets:
            break
        
        all_markets.extend(markets)
        print(f"Fetched {len(all_markets)} markets so far...")
        
        cursor = markets_response.get('cursor')
        if not cursor:
            break
    
    # Filter markets with extreme prices (10-25% or 75-90%)
    filtered_markets = []
    
    for market in all_markets:
        yes_price = market.get('yes_bid')  # Use yes_bid as price
        
        if yes_price is not None and (10 <= yes_price <= 25 or 75 <= yes_price <= 90):
            # Keep all original market fields that we need for URL construction
            filtered_market = {
                'ticker': market.get('ticker'),
                'title': market.get('title', ''),
                'price': yes_price,
                'event_ticker': market.get('event_ticker'),
                'category': market.get('category')
            }
            filtered_markets.append(filtered_market)
    
    print(f"Found {len(filtered_markets)} markets with target prices.")
    
    # Process markets in parallel (up to N_RESEARCHES)
    markets_to_research = filtered_markets[:N_RESEARCHES]
    results = await process_market_batch(perplexity_client, markets_to_research)
    
    # Calculate normalized scores for mispriced markets
    mispriced_results = [r for r in results if r['conclusion'] == 'MISPRICED' and r['fermi_price'] is not None]
    
    if mispriced_results:
        # Find max values for normalization
        max_calibration = max(result['calibration_score'] for result in mispriced_results)
        max_price_difference = max(result['price_difference'] for result in mispriced_results)
        
        # Calculate priority score for each result
        for result in mispriced_results:
            # Normalize values (scale from 0 to 1)
            normalized_calibration = result['calibration_score'] / max_calibration if max_calibration > 0 else 0
            normalized_difference = result['price_difference'] / max_price_difference if max_price_difference > 0 else 0
            
            # Calculate priority score: normalized_difference^2 * normalized_calibration
            result['priority_score'] = (normalized_difference ** 2) * normalized_calibration
    
    # Sort results: 
    # 1. MISPRICED first
    # 2. Then by priority score for mispriced markets
    # 3. For UNCERTAIN markets, sort by calibration score
    results.sort(key=lambda x: (
        0 if x['conclusion'] == 'MISPRICED' else 1,
        -x.get('priority_score', 0),  # Sort by priority score (descending)
        -x['calibration_score']       # Then by calibration score (for uncertain markets)
    ))
    
    # Print sorted results
    print("\n===== Mispriced Markets =====")
    print(f"{'Ticker':<20} {'Market Price':<12} {'Fermi Est':<12} {'Diff':<8} {'Calib':<6} {'Priority':<10} {'Title'}")
    print("-" * 150)
    
    for result in results:
        if result['conclusion'] == 'MISPRICED':
            fermi_display = f"{result['fermi_price']:.1f}%" if result['fermi_price'] is not None else "N/A"
            diff_display = f"{result['price_difference']:.1f}" if result['price_difference'] > 0 else "N/A"
            priority_display = f"{result.get('priority_score', 0):.4f}" if 'priority_score' in result else "N/A"
            
            print(f"{result['ticker']:<20} {result['price']:.1f}%{' ':<8} {fermi_display:<12} {diff_display:<8} {result['calibration_score']:<6} {priority_display:<10} {result['title']}")
            print(f"{'  Link:':<20} {result['market_link']}")
    
    print("\n===== Uncertain Markets =====")
    print(f"{'Ticker':<20} {'Market Price':<12} {'Fermi Est':<12} {'Calib':<6} {'Title'}")
    print("-" * 140)
    
    for result in results:
        if result['conclusion'] != 'MISPRICED':
            fermi_display = f"{result['fermi_price']:.1f}%" if result['fermi_price'] is not None else "N/A"
            
            print(f"{result['ticker']:<20} {result['price']:.1f}%{' ':<8} {fermi_display:<12} {result['calibration_score']:<6} {result['title']}")
            print(f"{'  Link:':<20} {result['market_link']}")

if __name__ == "__main__":
    asyncio.run(scan_markets())
