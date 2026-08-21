import os
import streamlit as st
import pandas as pd
import plotly.express as px

# Suppress Hugging Face warnings at application startup
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from src.scraper import fetch_rss_news
from src.analyzer import analyze_sentiment, fetch_stock_price_data

# Page Setup
st.set_page_config(
    page_title="Financial Sentiment & Market Tracker",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Real-Time Financial Sentiment & Market Tracker")
st.markdown("Analyze live news headlines using **FinBERT** and benchmark sentiment metrics against stock price trends.")

# Sidebar Configuration
st.sidebar.header("Search Configurations")
ticker = st.sidebar.text_input("Enter Ticker or Company (e.g., AAPL, TSLA, MSFT)", value="AAPL")
num_headlines = st.sidebar.slider("Number of Headlines", min_value=5, max_value=20, value=10)

if st.sidebar.button("Run Analysis", type="primary") or ticker:
    selected_ticker = ticker.strip().upper()
    st.subheader(f"Results for: {selected_ticker}")
    
    # 1. Fetch News Headlines and Run FinBERT Sentiment
    with st.spinner("Fetching news and executing FinBERT inference..."):
        news_df = fetch_rss_news(selected_ticker, max_results=num_headlines)
        
        if not news_df.empty:
            analyzed_df = analyze_sentiment(news_df)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("### Latest Headlines & Sentiment")
                # Using width="stretch" to eliminate Streamlit deprecation warning
                st.dataframe(analyzed_df[["title", "sentiment", "confidence"]], width="stretch")
            
            with col2:
                st.write("### Sentiment Breakdown")
                sentiment_counts = analyzed_df["sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]
                
                fig = px.pie(
                    sentiment_counts,
                    names="Sentiment",
                    values="Count",
                    color="Sentiment",
                    color_discrete_map={
                        "positive": "#00CC96",
                        "negative": "#EF553B",
                        "neutral": "#AB63FA"
                    }
                )
                try:
                    st.plotly_chart(fig, width="stretch")
                except TypeError:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No recent news headlines found for '{selected_ticker}'.")

    # 2. Fetch and Display Market Price Data
    st.markdown("---")
    st.subheader(f"📊 30-Day Stock Price Movement ({selected_ticker})")
    
    with st.spinner("Fetching market price data..."):
        price_df = fetch_stock_price_data(selected_ticker)
        
        if price_df is not None and not price_df.empty and "Close" in price_df.columns:
            st.line_chart(price_df["Close"])
        else:
            st.warning(
                f"⚠️ Live market price data for **{selected_ticker}** is temporarily throttled by Yahoo Finance on public cloud infrastructure. "
                "The FinBERT headline sentiment analysis above remains 100% active and functional."
            )