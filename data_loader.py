import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    # 🔹 Load text embeddings
    X_text = np.load("data/processed/embeddings.npy")

    # 🔹 Load structured features + labels
    df = pd.read_csv("data/processed/drug_reviews_clean.csv")

    X_struct = df[[
        "effectiveness_enc",
        "sideEffects_enc",
        "condition_enc"
    ]].values

    y = df["label"].values

    # Ensure all arrays match length
    min_len = min(len(X_text), len(X_struct), len(y))
    X_text = X_text[:min_len]
    X_struct = X_struct[:min_len]
    y = y[:min_len]

    return train_test_split(
        X_text, X_struct, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def load_train_data():
    X_text_train, _, X_struct_train, _, y_train, _ = load_data()

    return (
        torch.tensor(X_text_train, dtype=torch.float32),
        torch.tensor(X_struct_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )


def load_test_data():
    _, X_text_test, _, X_struct_test, _, y_test = load_data()

    return (
        torch.tensor(X_text_test, dtype=torch.float32),
        torch.tensor(X_struct_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
