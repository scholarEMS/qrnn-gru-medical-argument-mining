import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
import numpy as np

from qrnn_gru import QRNN_GRU
from data_loader import load_test_data

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)

    model = QRNN_GRU(300, 256, 256, 3).to(device)
    model.load_state_dict(torch.load("results/models/qrnn_gru.pt", map_location=device))
    model.eval()

    X_text_test, X_struct_test, y_test = load_test_data()
    X_text_test, X_struct_test, y_test = X_text_test.to(device), X_struct_test.to(device), y_test.to(device)

    with torch.no_grad():
        outputs = model(X_text_test, X_struct_test)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    y_true = y_test.cpu().numpy()
    y_pred = preds.cpu().numpy()
    y_scores = probs.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    with open("results/metrics/qrnn_gru_metrics.json", "w") as f:
        json.dump({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}, f, indent=4)

    np.save("results/predictions/y_true.npy", y_true)
    np.save("results/predictions/y_pred.npy", y_pred)
    np.save("results/predictions/y_scores.npy", y_scores)

if __name__ == "__main__":
    evaluate()
