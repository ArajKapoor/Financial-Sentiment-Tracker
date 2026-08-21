import pandas as pd
from src.scraper import fetch_financial_news
from src.analyzer import analyze_headline_sentiment

def run_sentiment_pipeline(ticker="AAPL"):
    df = fetch_financial_news(ticker)
    if df.empty:
        return df
        
    sentiments = []
    confidences = []
    
    for headline in df["title"]:
        sentiment, conf = analyze_headline_sentiment(headline)
        sentiments.append(sentiment)
        confidences.append(conf)
        
    df["Sentiment"] = sentiments
    df["Confidence"] = confidences
    return df