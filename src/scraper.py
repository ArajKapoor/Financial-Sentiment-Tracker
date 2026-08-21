import feedparser
import pandas as pd
from urllib.parse import quote

# Predefined RSS feeds mapped with normalized (lowercase) keys
SECTOR_FEEDS = {
    "top market news": "https://finance.yahoo.com/news/rssindex",
    "technology": "https://news.google.com/rss/search?q=technology+stocks+market+news&hl=en-IN&gl=IN&ceid=IN:en",
    "energy": "https://news.google.com/rss/search?q=energy+oil+gas+stocks+market+news&hl=en-IN&gl=IN&ceid=IN:en",
    "financials & banking": "https://news.google.com/rss/search?q=banking+finance+stocks+market+news&hl=en-IN&gl=IN&ceid=IN:en",
    "crypto market": "https://news.google.com/rss/search?q=crypto+bitcoin+market+news&hl=en-IN&gl=IN&ceid=IN:en"
}

def fetch_financial_news(ticker="AAPL", max_results=15):
    query_str = str(ticker).strip()
    query_norm = query_str.lower()
    
    # 1. Check if the input matches a predefined sector feed
    if query_norm in SECTOR_FEEDS:
        rss_url = SECTOR_FEEDS[query_norm]
    else:
        # 2. Otherwise treat as a ticker, converting spaces/specials to valid URL percent-encoding
        safe_ticker = quote(query_str)
        rss_url = f"https://finance.yahoo.com/rss/headline?s={safe_ticker}"
        
    feed = feedparser.parse(rss_url)
    
    articles = []
    for entry in feed.entries[:max_results]:
        articles.append({
            "title": entry.get("title", ""),
            "published": entry.get("published", ""),
            "link": entry.get("link", "")
        })
        
    return pd.DataFrame(articles)