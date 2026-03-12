"""TF-IDF + Logistic Regression baseline for threat intelligence classification."""

import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

import config
from src.data_loader import prepare_dataset


def train_tfidf_baseline():
    """Train and evaluate TF-IDF + Logistic Regression baseline."""
    train_df, val_df, test_df, le = prepare_dataset()

    print("\n" + "=" * 60)
    print("TF-IDF + Logistic Regression Baseline")
    print("=" * 60)

    # Build TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # Train Logistic Regression with class balancing
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate on validation set
    val_preds = clf.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average="weighted")
    print(f"\nValidation weighted F1: {val_f1:.4f}")

    # Evaluate on test set
    test_preds = clf.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average="weighted")
    print(f"Test weighted F1: {test_f1:.4f}")

    report = classification_report(
        y_test, test_preds,
        target_names=le.classes_,
        output_dict=True,
    )
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=le.classes_))

    # Save model and results
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(config.MODEL_DIR / "tfidf_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    results = {
        "model": "TF-IDF + Logistic Regression",
        "val_f1_weighted": float(val_f1),
        "test_f1_weighted": float(test_f1),
        "num_features": config.TFIDF_MAX_FEATURES,
        "ngram_range": list(config.TFIDF_NGRAM_RANGE),
    }
    with open(config.RESULTS_DIR / "tfidf_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_tfidf_baseline()
