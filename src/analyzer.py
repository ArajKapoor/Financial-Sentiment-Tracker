import os
import requests
import pandas as pd
import torch
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
    Fetches historical stock prices using Stooq CSV API to bypass Yahoo Finance cloud IP blocks.
    """
    try:
        # Format symbol for US equities on Stooq (e.g. AAPL.US)
        clean_ticker = ticker_symbol.strip().lower()
        if not clean_ticker.endswith(".us"):
            stooq_ticker = f"{clean_ticker}.us"
        else:
            stooq_ticker = clean_ticker

        url = f"https://stooq.com/q/d/l/?s={stooq_ticker}&i=d"
        df = pd.read_csv(url)
        
        if df.empty or "Close" not in df.columns:
            return pd.DataFrame()
            
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").tail(30)
        df.set_index("Date", inplace=True)
        return df
    except Exception as e:
        print(f"Stooq fetch error for {ticker_symbol}: {e}")
        return pd.DataFrame()