import feedparser
import urllib.parse
import pandas as pd

def fetch_rss_news(query: str, max_results: int = 10) -> pd.DataFrame:
    """
    Fetches news headlines from Google News RSS feed for a given ticker or topic.
    """
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(rss_url)
    
    articles = []
    for entry in feed.entries[:max_results]:
        articles.append({
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "published": entry.get("published", "")
        })
        
    return pd.DataFrame(articles)


def fetch_financial_news(ticker: str = "AAPL", max_results: int = 15) -> pd.DataFrame:
    """Compatibility wrapper used by `src.pipeline`.

    Keeps the simple behavior of fetching RSS headlines for the given
    ticker or topic.
    """
    return fetch_rss_news(ticker, max_results=max_results)