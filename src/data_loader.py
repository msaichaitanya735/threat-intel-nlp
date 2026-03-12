"""Load and preprocess the Security-TTP-Mapping dataset from HuggingFace."""

import pandas as pd
from collections import Counter
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config


def load_raw_dataset():
    """Load the dataset and combine procedure + expert splits."""
    ds = load_dataset(config.HF_DATASET)
    splits_to_combine = []
    for split_name in ds:
        splits_to_combine.append(ds[split_name])
    combined = concatenate_datasets(splits_to_combine)
    df = combined.to_pandas()
    print(f"Combined dataset size: {len(df)} samples")
    print(f"Available columns: {list(df.columns)}")
    return df


def filter_top_k_techniques(df, text_col, label_col, k=config.TOP_K_TECHNIQUES):
    """Keep only the top-k most frequent technique labels."""
    label_counts = Counter(df[label_col])
    top_k_labels = [label for label, _ in label_counts.most_common(k)]
    df_filtered = df[df[label_col].isin(top_k_labels)].copy()
    print(f"Filtered to top {k} techniques: {len(df_filtered)} samples "
          f"(dropped {len(df) - len(df_filtered)})")
    return df_filtered


def prepare_dataset():
    """Full data preparation pipeline.

    Returns (train_df, val_df, test_df, label_encoder) with columns
    'text' and 'label' (integer-encoded).
    """
    df = load_raw_dataset()

    # Identify text and label columns
    text_col = None
    label_col = None
    for col in df.columns:
        col_lower = col.lower()
        if "procedure" in col_lower or "text" in col_lower or "sentence" in col_lower:
            if text_col is None:
                text_col = col
        if "technique" in col_lower or "label" in col_lower or "ttp" in col_lower or "tid" in col_lower:
            if label_col is None:
                label_col = col

    if text_col is None or label_col is None:
        # Fallback: use first string-like columns
        str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if len(str_cols) >= 2:
            text_col = text_col or str_cols[0]
            label_col = label_col or str_cols[1]
        else:
            raise ValueError(
                f"Cannot identify text/label columns from: {list(df.columns)}"
            )

    print(f"Using text column: '{text_col}', label column: '{label_col}'")

    # Drop rows with missing text or labels
    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].str.strip().astype(bool)]

    # Filter to top-k techniques
    df = filter_top_k_techniques(df, text_col, label_col)

    # Encode labels
    le = LabelEncoder()
    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label_name"]
    df["label"] = le.fit_transform(df["label_name"])
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    # Stratified split: train / val / test
    train_df, temp_df = train_test_split(
        df, test_size=config.VAL_SIZE + config.TEST_SIZE,
        stratify=df["label"], random_state=config.RANDOM_SEED,
    )
    relative_test = config.TEST_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["label"], random_state=config.RANDOM_SEED,
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Save label mapping
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    label_map = {int(i): name for i, name in enumerate(le.classes_)}
    pd.Series(label_map).to_json(config.DATA_DIR / "label_map.json")

    return train_df, val_df, test_df, le


if __name__ == "__main__":
    train_df, val_df, test_df, le = prepare_dataset()
    print("\nSample training data:")
    print(train_df[["text", "label_name"]].head())
