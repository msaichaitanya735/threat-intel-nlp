"""BERT fine-tuning for threat intelligence technique classification."""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, classification_report

import config
from src.data_loader import prepare_dataset


class ThreatIntelDataset(Dataset):
    """PyTorch dataset for tokenized threat intelligence text."""

    def __init__(self, texts, labels, tokenizer, max_length=config.MAX_SEQ_LENGTH):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    """Compute weighted F1 for Trainer evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1_weighted = f1_score(labels, preds, average="weighted")
    f1_macro = f1_score(labels, preds, average="macro")
    return {"f1_weighted": f1_weighted, "f1_macro": f1_macro}


def train_bert_classifier():
    """Fine-tune BERT on threat intelligence classification."""
    train_df, val_df, test_df, le = prepare_dataset()
    num_labels = len(le.classes_)

    print("\n" + "=" * 60)
    print("BERT Fine-Tuning")
    print("=" * 60)

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BERT_MODEL_NAME,
        num_labels=num_labels,
    )

    # Create datasets
    train_dataset = ThreatIntelDataset(
        train_df["text"].values, train_df["label"].values, tokenizer
    )
    val_dataset = ThreatIntelDataset(
        val_df["text"].values, val_df["label"].values, tokenizer
    )
    test_dataset = ThreatIntelDataset(
        test_df["text"].values, test_df["label"].values, tokenizer
    )

    # Training arguments
    output_dir = str(config.MODEL_DIR / "bert-threat-intel")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print(f"\nTraining on {len(train_dataset)} samples...")
    trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test weighted F1: {test_results['eval_f1_weighted']:.4f}")
    print(f"Test macro F1: {test_results['eval_f1_macro']:.4f}")

    # Detailed classification report
    test_preds = trainer.predict(test_dataset)
    pred_labels = np.argmax(test_preds.predictions, axis=-1)
    true_labels = test_df["label"].values

    print("\nClassification Report (Test Set):")
    report_str = classification_report(
        true_labels, pred_labels, target_names=le.classes_
    )
    print(report_str)

    report_dict = classification_report(
        true_labels, pred_labels, target_names=le.classes_, output_dict=True
    )

    # Save the best model
    best_model_path = config.MODEL_DIR / "bert-threat-intel-best"
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))

    # Save results
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model": f"Fine-tuned {config.BERT_MODEL_NAME}",
        "test_f1_weighted": float(test_results["eval_f1_weighted"]),
        "test_f1_macro": float(test_results["eval_f1_macro"]),
        "num_labels": num_labels,
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "num_epochs": config.NUM_EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "train_samples": len(train_dataset),
    }
    with open(config.RESULTS_DIR / "bert_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_bert_classifier()
