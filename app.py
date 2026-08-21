import os
import sys
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# Ensure module imports from /src work properly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.pipeline import run_sentiment_pipeline

# Page Setup
st.set_page_config(
    page_title="Financial Sentiment & Market Tracker",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Real-Time Financial Sentiment & Market Tracker")
st.markdown(
    "Analyze live market sentiment from stock headlines or industry sectors using **FinBERT**."
)

# Sidebar Controls
st.sidebar.header("Search Configuration")
mode = st.sidebar.radio(
    "Search Mode:", ["Single Stock Ticker", "Market / Sector News"]
)

if mode == "Single Stock Ticker":
    query = (
        st.sidebar.text_input(
            "Enter Stock Ticker (e.g., AAPL, RELIANCE.NS, TSLA):", value="AAPL"
        )
        .strip()
        .upper()
    )
else:
    query = st.sidebar.selectbox(
        "Select Sector:",
        [
            "Top Market News",
            "Technology",
            "Energy",
            "Financials & Banking",
            "Crypto Market",
        ],
    )

# Run Analysis
if st.sidebar.button("Run Analysis") or query:
    with st.spinner(f"Fetching news and analyzing sentiment for '{query}'..."):
        df = run_sentiment_pipeline(query)

    if df.empty:
        st.error(
            f"No headlines found for '{query}'. Please check the symbol or sector selection."
        )
    else:
        # Initialize price history variable
        hist = pd.DataFrame()

        # -------------------------------------------------------------
        # Feature 1: Composite Sentiment Index (KPI Metric Cards)
        # -------------------------------------------------------------
        score_map = {"Positive": 1, "Negative": -1, "Neutral": 0}
        df["Score"] = df["Sentiment"].map(score_map) * df["Confidence"]
        avg_score = df["Score"].mean()

        m1, m2, m3 = st.columns(3)

        with m1:
            if avg_score > 0.15:
                st.metric(
                    "Market Stance Index",
                    "BULLISH 📈",
                    delta=f"{avg_score*100:.1f}% Index",
                )
            elif avg_score < -0.15:
                st.metric(
                    "Market Stance Index",
                    "BEARISH 📉",
                    delta=f"{avg_score*100:.1f}% Index",
                )
            else:
                st.metric(
                    "Market Stance Index",
                    "NEUTRAL ⚖️",
                    delta=f"{avg_score*100:.1f}% Index",
                )

        with m2:
            st.metric("Total Headlines Analyzed", len(df))

        with m3:
            pos_ratio = (df["Sentiment"] == "Positive").mean() * 100
            st.metric("Positive Sentiment Ratio", f"{pos_ratio:.1f}%")

        st.divider()

        # -------------------------------------------------------------
        # Main Dashboard Layout
        # -------------------------------------------------------------
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Sentiment Breakdown")
            sentiment_counts = df["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#2ecc71",
                    "Negative": "#e74c3c",
                    "Neutral": "#95a5a6",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature 4: Top Financial Keywords
            words = " ".join(df["title"]).lower().split()
            stop_words = {
                "the", "a", "an", "to", "in", "and", "of", "for", "on",
                "with", "at", "by", "is", "are", "from", "as", "it",
                "its", "stock", "stocks", "shares", "market"
            }
            cleaned_words = [
                w.strip(".,!?\"'()")
                for w in words
                if w.strip(".,!?\"'()").isalnum()
                and w.strip(".,!?\"'()") not in stop_words
                and len(w) > 3
            ]

            top_terms = Counter(cleaned_words).most_common(5)
            if top_terms:
                st.markdown(
                    "**🔥 Trending Terms:** "
                    + ", ".join(
                        [f"`{word}` ({count})" for word, count in top_terms]
                    )
                )

        with col2:
            # Feature 2: Fetch and Display yfinance Price Chart
            if mode == "Single Stock Ticker":
                st.subheader(f"30-Day Price Movement ({query})")
                try:
                    stock = yf.Ticker(query)
                    hist = stock.history(period="1mo")
                    if not hist.empty:
                        st.line_chart(hist["Close"])
                    else:
                        st.info(
                            f"No price history found for ticker symbol '{query}'."
                        )
                except Exception:
                    st.info("Could not retrieve stock price history.")
            else:
                st.subheader("Sector News Summary")
                st.info(
                    "Select a specific stock ticker in the sidebar to view historical price action charts."
                )

            # Feature 5: Sentiment-Price Divergence Engine
            if mode == "Single Stock Ticker" and not hist.empty and len(hist) >= 5:
                price_5d_change = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100

                st.markdown("#### ⚡ Quant Anomaly & Divergence Signal")
                
                if avg_score > 0.15 and price_5d_change < -1.0:
                    st.warning(
                        f"**Bullish Divergence Detected:** News sentiment is positive ({avg_score*100:.1f}%), "
                        f"but stock price fell {price_5d_change:.2f}% over 5 days. "
                        "Market may be underreacting to positive news."
                    )
                elif avg_score < -0.15 and price_5d_change > 1.0:
                    st.error(
                        f"**Bearish Divergence Detected:** News sentiment is negative ({avg_score*100:.1f}%), "
                        f"but stock price rose +{price_5d_change:.2f}% over 5 days. "
                        "Rally may be unsupported by underlying market news."
                    )
                else:
                    st.success(
                        f"**Market Alignment:** News sentiment ({avg_score*100:.1f}%) and 5-day price action "
                        f"({price_5d_change:+.2f}%) are moving in the same direction."
                    )

        st.divider()

        # -------------------------------------------------------------
        # Data Table & Feature 3: CSV Export
        # -------------------------------------------------------------
        st.subheader("Latest Analyzed Headlines")

        display_df = df[["title", "Sentiment", "Confidence", "published"]]

        st.dataframe(
            display_df,
            column_config={
                "title": "Headline",
                "published": "Published Date",
                "Confidence": st.column_config.NumberColumn(
                    "Confidence Score", format="%.2f"
                ),
            },
            use_container_width=True,
            hide_index=True,
        )

        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Sentiment Analysis Report (CSV)",
            data=csv_data,
            file_name=f"{query}_sentiment_report.csv",
            mime="text/csv",
        )