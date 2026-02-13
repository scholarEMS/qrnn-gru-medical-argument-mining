import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import LabelEncoder

nltk.download("punkt")

COLUMN_NAMES = [
    "reviewID","urlDrugName","rating","effectiveness","sideEffects",
    "condition","benefitsReview","sideEffectsReview","commentsReview"
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 🔹 Weak supervision for argument component labels
def weak_argument_label(text):
    text = text.lower()

    # Premise indicators
    if any(w in text for w in ["because", "since", "due to", "as a result"]):
        return 1  # Premise

    # Claim indicators
    if any(w in text for w in ["i believe", "i think", "this drug", "it works", "it helps"]):
        return 0  # Claim

    return 2  # Non-argument


def preprocess(input_path, output_path):
    df = pd.read_csv(input_path, sep="\t", names=COLUMN_NAMES, header=0)

    # Combine review fields
    df["full_review"] = (
        df["benefitsReview"].fillna("") + " " +
        df["sideEffectsReview"].fillna("") + " " +
        df["commentsReview"].fillna("")
    )

    # 🔹 Split reviews into sentences (argument units)
    df["sentences"] = df["full_review"].apply(sent_tokenize)
    df = df.explode("sentences")

    df["clean"] = df["sentences"].apply(clean_text)
    df["tokens"] = df["clean"].apply(word_tokenize)
    df["stemmed"] = df["tokens"]

    # 🔹 Create argument-style labels
    df["label"] = df["clean"].apply(weak_argument_label)

    # Encode structured features (optional but useful)
    le_eff = LabelEncoder()
    le_side = LabelEncoder()
    le_cond = LabelEncoder()

    df["effectiveness_enc"] = le_eff.fit_transform(df["effectiveness"].astype(str))
    df["sideEffects_enc"] = le_side.fit_transform(df["sideEffects"].astype(str))
    df["condition_enc"] = le_cond.fit_transform(df["condition"].astype(str))

    df.to_csv(output_path, index=False)
    print("Preprocessing complete with ARGUMENT-STYLE labels.")


if __name__ == "__main__":
    preprocess("data/raw/drugLibTrain_raw.tsv",
               "data/processed/drug_reviews_clean.csv")
