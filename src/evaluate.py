"""Evaluation utilities and comparison between models."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from pathlib import Path

import config


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    # Normalize
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm_norm, annot=False, fmt=".1f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_model_comparison(save_path=None):
    """Bar chart comparing TF-IDF baseline vs BERT F1 scores."""
    results_dir = config.RESULTS_DIR
    if save_path is None:
        save_path = results_dir / "model_comparison.png"

    tfidf_path = results_dir / "tfidf_results.json"
    bert_path = results_dir / "bert_results.json"

    models = []
    f1_scores = []

    if tfidf_path.exists():
        with open(tfidf_path) as f:
            tfidf = json.load(f)
        models.append("TF-IDF + LR")
        f1_scores.append(tfidf["test_f1_weighted"])

    if bert_path.exists():
        with open(bert_path) as f:
            bert = json.load(f)
        models.append("Fine-tuned BERT")
        f1_scores.append(bert["test_f1_weighted"])

    if not models:
        print("No results found to compare.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, f1_scores, color=["#5B9BD5", "#ED7D31"], width=0.5)
    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{score:.3f}", ha="center", va="bottom", fontweight="bold",
        )
    ax.set_ylabel("Weighted F1 Score")
    ax.set_title("Threat Intelligence Classification: Model Comparison")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison chart to {save_path}")

    if len(f1_scores) == 2:
        improvement = (f1_scores[1] - f1_scores[0]) * 100
        print(f"\nBERT improvement over TF-IDF: {improvement:+.1f} F1 points")


if __name__ == "__main__":
    plot_model_comparison()
