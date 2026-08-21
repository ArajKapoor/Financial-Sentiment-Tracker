import os
import requests
import pandas as pd
import torch
import yfinance as yf
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging

# Suppress Hugging Face warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.set_verbosity_error()

@st.cache_resource
def load_finbert():
    """
    Loads FinBERT model and tokenizer with caching to avoid reloading on every rerun.
    """
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs FinBERT inference on news headline titles.
    """
    if df.empty or "title" not in df.columns:
        return df

    tokenizer, model = load_finbert()
    titles = df["title"].tolist()

    inputs = tokenizer(titles, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    labels = ["positive", "negative", "neutral"]
    
    results = []
    for p in probs:
        max_idx = torch.argmax(p).item()
        results.append({
            "sentiment": labels[max_idx],
            "confidence": round(p[max_idx].item(), 4)
        })

    sentiment_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)

def fetch_stock_price_data(ticker_symbol: str) -> pd.DataFrame:
    """
    Fetches historical stock prices using yfinance with custom browser headers to prevent throttling.
    """
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        })
        
        ticker = yf.Ticker(ticker_symbol, session=session)
        df = ticker.history(period="1mo")
        
        if df.empty:
            return pd.DataFrame()
            
        return df
    except Exception as e:
        print(f"yfinance fetch error for {ticker_symbol}: {e}")
        return pd.DataFrame()   

def fetch_stock_price_data(ticker_symbol: str) -> pd.DataFrame:
    """
    Fetches historical stock prices using yfinance with safety handling for cloud IP blocks.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="1mo")
        
        if df is None or df.empty:
            return pd.DataFrame()
            
        return df
    except Exception as e:
        return pd.DataFrame()