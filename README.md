# 📈 Real-Time Financial Sentiment & Market Tracker

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financial-sentiment-tracker.streamlit.app)

> **Live Web Application:** [financial-sentiment-tracker.streamlit.app](https://arajkapoorfinancialsentimenttracker.streamlit.app)

An end-to-end Natural Language Processing (NLP) and Quantitative Analytics web application built with **FinBERT**, **PyTorch**, and **Streamlit**. The system evaluates real-time market sentiment from stock news feeds and benchmarks NLP metrics against historical stock price performance (`yfinance`) to detect market mispricings and quant anomalies.

---

## Key Features

* **FinBERT NLP Engine:** Uses a specialized Financial BERT transformer (`ProsusAI/finbert`) to classify financial news headlines into `Positive`, `Negative`, or `Neutral` stances with token-level confidence scores.
* **Quant Anomaly & Divergence Signal:** Automatically calculates 5-day stock returns and cross-references them against FinBERT sentiment scores to detect:
  * **Bullish Divergence:** High positive sentiment alongside dropping price action (market underreaction).
  * **Bearish Divergence:** Negative sentiment alongside rising price action (unsupported rally).
* **Live Price Trend Overlays:** Integrates historical 30-day stock market price charts powered by `yfinance`.
* **Composite Market Stance Index:** Aggregates headline confidence into a single quantitative Bullish/Bearish index score.
* **Trending Key-Phrase Extraction:** Automatically filters and extracts key financial terms driving news flow.
* **Exportable Data Reports:** Download processed headlines, confidence scores, and timestamps in CSV format for downstream financial analysis.

---

## Tech Stack

* **Language:** Python 3.10+
* **Machine Learning & NLP:** PyTorch, Hugging Face Transformers (`FinBERT`)
* **Frontend / Dashboard:** Streamlit, Plotly Express
* **Financial Data & Scraping:** `yfinance`, `feedparser`, Pandas

---

## Repository Structure

Financial-Sentiment-Tracker/
│
├── app.py                  # Main Streamlit web application & UI logic
├── requirements.txt        # Python package dependencies
├── .gitignore              # Ignored files (virtual environment, cache)
├── README.md               # Project documentation
│
└── src/                    # Core processing scripts
    ├── __init__.py
    ├── scraper.py          # RSS headline fetching module
    ├── analyzer.py         # PyTorch / FinBERT sentiment model pipeline
    └── pipeline.py         # End-to-end execution script

---

**Deployment Notes**

- If you deploy to Streamlit Cloud (or other hosted runners), set the environment variable `TRANSFORMERS_NO_IMAGE_PROCESSING=1` in the app settings. This prevents `transformers` from importing optional image-processing submodules that require `torchvision` at import time and can cause startup failures.
- Ensure `requirements.txt` contains `torch`, `torchvision`, and `transformers` so the deployed environment installs compatible PyTorch wheels. After pushing these changes, restart/redeploy the app in Streamlit Cloud so the platform installs the new packages.

Example Streamlit Cloud steps:

1. Push changes to your repository:

```powershell
git add requirements.txt README.md
git commit -m "Deploy: add torchvision and deployment note"
git push origin main
```

2. In Streamlit Cloud → Your app → Settings → Environment variables: add `TRANSFORMERS_NO_IMAGE_PROCESSING = 1` and restart the app.

These steps will eliminate the `ModuleNotFoundError: No module named 'torchvision'` crash on startup.