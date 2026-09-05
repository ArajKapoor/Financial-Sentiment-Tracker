# Financial Sentiment and Market Tracker

A Streamlit application that combines financial news sentiment analysis with recent stock price data.

## What It Does

- Fetches headlines from Google News RSS for a ticker or company name.
- Classifies each headline with the `ProsusAI/finbert` model.
- Labels headlines as positive, negative, or neutral and displays confidence scores.
- Fetches the latest 30 daily closing prices from the Stooq CSV API.
- Displays the headlines, sentiment breakdown, and price chart in a Streamlit dashboard.

## Technology

- Python 3.10+
- Streamlit
- PyTorch
- Hugging Face Transformers and FinBERT
- Pandas
- Plotly
- Feedparser
- Stooq CSV API for historical prices

The application does not currently use `yfinance`. Although it remains listed in `requirements.txt`, price data is fetched from Stooq in `src/analyzer.py`.

## Project Structure

```text
financial sentiment app/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── src/
    ├── analyzer.py         # FinBERT inference and Stooq price retrieval
    ├── pipeline.py         # Pipeline helper
    └── scraper.py          # Google News RSS retrieval
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the App

```powershell
streamlit run app.py
```

The first sentiment analysis run downloads the FinBERT tokenizer and model from Hugging Face.

## Deployment

For Streamlit Cloud, set this environment variable if the deployment encounters optional image-processing imports:

```text
TRANSFORMERS_NO_IMAGE_PROCESSING=1
```

The application requires network access to Google News RSS, Hugging Face, and Stooq.