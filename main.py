"""Main pipeline: data prep -> TF-IDF baseline -> BERT fine-tuning -> comparison."""

import argparse
import json

import config
from src.data_loader import prepare_dataset
from src.tfidf_baseline import train_tfidf_baseline
from src.bert_classifier import train_bert_classifier
from src.evaluate import plot_model_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Threat Intelligence Classification using Fine-Tuned Transformers"
    )
    parser.add_argument(
        "--stage",
        choices=["data", "tfidf", "bert", "compare", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    if args.stage in ("data", "all"):
        print("\n>>> Stage 1: Data Preparation")
        train_df, val_df, test_df, le = prepare_dataset()
        print(f"Dataset ready: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    if args.stage in ("tfidf", "all"):
        print("\n>>> Stage 2: TF-IDF Baseline")
        tfidf_results = train_tfidf_baseline()
        print(f"TF-IDF test F1: {tfidf_results['test_f1_weighted']:.4f}")

    if args.stage in ("bert", "all"):
        print("\n>>> Stage 3: BERT Fine-Tuning")
        bert_results = train_bert_classifier()
        print(f"BERT test F1: {bert_results['test_f1_weighted']:.4f}")

    if args.stage in ("compare", "all"):
        print("\n>>> Stage 4: Model Comparison")
        plot_model_comparison()

    # Print final summary
    results_dir = config.RESULTS_DIR
    tfidf_path = results_dir / "tfidf_results.json"
    bert_path = results_dir / "bert_results.json"

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    if tfidf_path.exists():
        with open(tfidf_path) as f:
            t = json.load(f)
        print(f"TF-IDF + LR       | Test F1: {t['test_f1_weighted']:.4f}")

    if bert_path.exists():
        with open(bert_path) as f:
            b = json.load(f)
        print(f"Fine-tuned BERT    | Test F1: {b['test_f1_weighted']:.4f}")

    if tfidf_path.exists() and bert_path.exists():
        improvement = (b["test_f1_weighted"] - t["test_f1_weighted"]) * 100
        print(f"\nBERT improvement: {improvement:+.1f} F1 points over TF-IDF baseline")


if __name__ == "__main__":
    main()
