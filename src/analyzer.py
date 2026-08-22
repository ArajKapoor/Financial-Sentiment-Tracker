"""Analyzer: provide a minimal, lazy-loading FinBERT sentiment function.

Keep top-level imports lightweight to avoid triggering transformers' image
processing submodules (which require `torchvision`) during app startup.
"""

MODEL_NAME = "ProsusAI/finbert"

# Cached tokenizer/model (loaded on first use)
_tokenizer = None
_model = None
_labels = ["Positive", "Negative", "Neutral"]


def _load_model():
    """Load tokenizer and model; raise a clear error if dependencies missing."""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except Exception as e:
        raise RuntimeError(
            "Missing ML dependencies: install 'transformers' and 'torch' (and optionally 'torchvision')"
        ) from e

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def analyze_headline_sentiment(headline: str):
    """Return (label, confidence) for a single headline string.

    The model/tokenizer are loaded on first call and cached afterward.
    """
    _load_model()

    import torch
    import torch.nn.functional as F

    inputs = _tokenizer(headline, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    max_idx = torch.argmax(probs).item()
    return _labels[max_idx], round(probs[max_idx].item(), 4)


def analyze_sentiment(df: "pd.DataFrame") -> "pd.DataFrame":
    """Run sentiment analysis on a DataFrame with a `title` column.

    Returns a new DataFrame with added `sentiment` and `confidence` columns
    (lowercase labels to match the app UI expectations).
    """
    import pandas as pd

    if df is None or df.empty or "title" not in df.columns:
        return df

    sentiments = []
    confidences = []
    for title in df["title"].astype(str).tolist():
        label, conf = analyze_headline_sentiment(title)
        sentiments.append(label.lower())
        confidences.append(conf)

    out = df.reset_index(drop=True).copy()
    out["sentiment"] = sentiments
    out["confidence"] = confidences
    return out


def fetch_stock_price_data(ticker_symbol: str) -> "pd.DataFrame":
    """Attempt to fetch 30 days of price history for `ticker_symbol`.

    Tries `yfinance` first; on failure falls back to Stooq CSV download.
    Returns a DataFrame indexed by Date with a `Close` column, or an empty
    DataFrame on failure.
    """
    import pandas as pd

    # Try yfinance (preferred)
    try:
        import yfinance as yf

        t = yf.Ticker(ticker_symbol)
        hist = t.history(period="1mo")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            return hist
    except Exception:
        # fallthrough to stooq
        pass

    # Fallback: Stooq (daily CSV)
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker_symbol.lower()}.us&i=d"
        df = pd.read_csv(url)
        if df is None or df.empty or "Close" not in df.columns:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").tail(30)
        df.set_index("Date", inplace=True)
        return df
    except Exception:
        return pd.DataFrame()