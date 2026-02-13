import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/processed/drug_reviews_clean.csv")
sentences = df["stemmed"].apply(eval).tolist()

print("Training Word2Vec embeddings...")

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=300,
    window=7,
    min_count=2,
    workers=4,
    sg=1,
    negative=10,
    epochs=15
)

print("Word2Vec training complete.")

embedding_dim = 300
max_len = 120   # 🔹 Reduced from 200 to prevent memory crash

def sentence_to_embedding(tokens):
    vecs = []
    for word in tokens[:max_len]:
        vecs.append(w2v_model.wv[word] if word in w2v_model.wv else np.zeros(embedding_dim, dtype=np.float32))
    while len(vecs) < max_len:
        vecs.append(np.zeros(embedding_dim, dtype=np.float32))
    return np.array(vecs, dtype=np.float32)   # 🔹 Force float32

print("Converting sentences to embedding matrices...")
embeddings = np.array([sentence_to_embedding(tokens) for tokens in sentences], dtype=np.float32)

np.save("data/processed/embeddings.npy", embeddings)
print("Saved embeddings to data/processed/embeddings.npy")
