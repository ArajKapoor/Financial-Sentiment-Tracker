import os
import streamlit as st
import pandas as pd
import plotly.express as px

# Suppress Hugging Face warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from src.scraper import fetch_rss_news
from src.analyzer import analyze_sentiment, fetch_stock_price_data

st.set_page_config(
    page_title="Financial Sentiment & Market Tracker",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Real-Time Financial Sentiment & Market Tracker")
st.markdown("Analyze live news headlines with **FinBERT** and cross-reference sentiment against historical price movements.")

# Sidebar Controls
st.sidebar.header("Search Configurations")
ticker = st.sidebar.text_input("Enter Ticker or Company (e.g., AAPL, TSLA, MSFT)", value="AAPL")
num_headlines = st.sidebar.slider("Number of Headlines", min_value=5, max_value=20, value=10)

if st.sidebar.button("Run Analysis", type="primary") or ticker:
    st.subheader(f"Results for: {ticker.upper()}")
    
    # 1. Fetch and Analyze News
    with st.spinner("Fetching news and calculating FinBERT sentiment..."):
        news_df = fetch_rss_news(ticker, max_results=num_headlines)
        
        if not news_df.empty:
            analyzed_df = analyze_sentiment(news_df)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("### Latest Headlines & Sentiment")
                st.dataframe(analyzed_df[["title", "sentiment", "confidence"]], width="stretch")
            
            with col2:
                st.write("### Sentiment Distribution")
                sentiment_counts = analyzed_df["sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]
                
                fig = px.pie(
                    sentiment_counts,
                    names="Sentiment",
                    values="Count",
                    color="Sentiment",
                    color_discrete_map={"positive": "#00CC96", "negative": "#EF553B", "neutral": "#AB63FA"}
                )
                st.plotly_chart(fig, width="stretch")
        else:
            st.warning(f"No recent news headlines found for {ticker}.")

    # 2. Fetch and Display Stock Price History
    st.markdown("---")
    st.subheader(f"📊 30-Day Stock Price Movement ({ticker.upper()})")
    
    with st.spinner("Fetching market price data..."):
        price_df = fetch_stock_price_data(ticker)
        
        if not price_df.empty and "Close" in price_df.columns:
            st.line_chart(price_df["Close"], width="stretch")
        else:
            st.warning(
                f"⚠️ Live market price chart for '{ticker}' is temporarily unavailable due to Yahoo Finance rate limiting. "
                "The FinBERT news sentiment analysis above remains fully functional."
            )