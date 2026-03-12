"""Inference utility: classify new threat intelligence text."""

import json
import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config


def load_label_map():
    """Load the label index -> technique name mapping."""
    with open(config.DATA_DIR / "label_map.json") as f:
        label_map = json.load(f)
    return {int(k): v for k, v in label_map.items()}


def predict_tfidf(texts):
    """Classify texts using the TF-IDF baseline."""
    with open(config.MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(config.MODEL_DIR / "tfidf_classifier.pkl", "rb") as f:
        clf = pickle.load(f)

    label_map = load_label_map()
    X = vectorizer.transform(texts)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    return [
        {"technique": label_map[int(p)], "confidence": float(probs[i].max())}
        for i, p in enumerate(preds)
    ]


def predict_bert(texts):
    """Classify texts using the fine-tuned BERT model."""
    model_path = str(config.MODEL_DIR / "bert-threat-intel-best")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    label_map = load_label_map()

    encodings = tokenizer(
        texts, truncation=True, padding=True,
        max_length=config.MAX_SEQ_LENGTH, return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    return [
        {"technique": label_map[int(p)], "confidence": float(probs[i].max())}
        for i, p in enumerate(preds)
    ]


if __name__ == "__main__":
    sample_texts = [
        "The adversary used spearphishing emails with malicious attachments to gain initial access to the target network.",
        "After gaining access, the attacker dumped credentials from LSASS memory using Mimikatz.",
        "The malware established persistence by creating a new Windows service that runs at system startup.",
    ]
    print("BERT predictions:")
    for text, result in zip(sample_texts, predict_bert(sample_texts)):
        print(f"  Text: {text[:80]}...")
        print(f"  -> {result['technique']} (confidence: {result['confidence']:.3f})\n")
