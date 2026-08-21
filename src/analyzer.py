from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
labels = ["Positive", "Negative", "Neutral"]

def analyze_headline_sentiment(headline):
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        
    max_idx = torch.argmax(probs).item()
    return labels[max_idx], round(probs[max_idx].item(), 4)