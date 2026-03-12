# Threat Intelligence Classification Using Fine-Tuned Transformers

Fine-tuned BERT on 25K+ MITRE ATT&CK-labeled threat reports to classify attack techniques across 50+ categories; achieved 91% F1 score, outperforming baseline TF-IDF approach by 23 points.

## Overview

This project builds an end-to-end NLP pipeline for automatically classifying cyber threat intelligence reports into MITRE ATT&CK technique categories. It compares a traditional TF-IDF + Logistic Regression baseline against a fine-tuned BERT transformer model.

### Key Results

| Model | Weighted F1 |
|-------|------------|
| TF-IDF + Logistic Regression | ~0.68 |
| Fine-tuned BERT | ~0.91 |

## Dataset

- **Source**: [Security-TTP-Mapping](https://huggingface.co/datasets/tumeteor/Security-TTP-Mapping) from HuggingFace
- **Size**: 25K+ samples (procedure + expert splits combined)
- **Labels**: Top 50 most frequent MITRE ATT&CK techniques
- **Split**: 70% train / 15% validation / 15% test (stratified)

## Project Structure

```
threat-intel-nlp/
├── config.py                # Hyperparameters and paths
├── main.py                  # Pipeline entrypoint
├── requirements.txt         # Dependencies
└── src/
    ├── data_loader.py       # Dataset loading, filtering, preprocessing
    ├── tfidf_baseline.py    # TF-IDF + Logistic Regression baseline
    ├── bert_classifier.py   # BERT fine-tuning with HuggingFace Trainer
    ├── evaluate.py          # Confusion matrices and model comparison plots
    └── predict.py           # Inference on new threat intelligence text
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Run the full pipeline

```bash
python main.py --stage all
```

### Run individual stages

```bash
python main.py --stage data      # Download and preprocess dataset
python main.py --stage tfidf     # Train TF-IDF baseline
python main.py --stage bert      # Fine-tune BERT
python main.py --stage compare   # Generate comparison chart
```

### Inference on new text

```python
from src.predict import predict_bert

texts = [
    "The adversary used spearphishing emails with malicious attachments to gain initial access.",
    "The attacker dumped credentials from LSASS memory using Mimikatz.",
]

results = predict_bert(texts)
for text, result in zip(texts, results):
    print(f"{result['technique']} (confidence: {result['confidence']:.3f})")
```

## Approach

### TF-IDF Baseline
- 30,000 features with (1,3)-gram range
- Logistic Regression with balanced class weights
- Sublinear TF scaling

### BERT Fine-Tuning
- Base model: `bert-base-uncased`
- Max sequence length: 256 tokens
- Learning rate: 2e-5 with warmup (10%)
- Early stopping with patience of 2 epochs
- Best model selected by validation weighted F1

## References

- [MITRE ATT&CK Framework](https://attack.mitre.org/)
- [TRAM - Threat Report ATT&CK Mapper](https://github.com/center-for-threat-informed-defense/tram/)
- [Security-TTP-Mapping Dataset](https://huggingface.co/datasets/tumeteor/Security-TTP-Mapping)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2019
