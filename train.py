import torch
import os
import csv
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from qrnn_gru import QRNN_GRU
from utils import set_seed
from data_loader import load_train_data

def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    X_text_train, X_struct_train, y_train = load_train_data()
    train_dataset = TensorDataset(X_text_train, X_struct_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = QRNN_GRU(300, 256, 256, 3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Class balancing
    classes = np.unique(y_train.numpy())
    weights = compute_class_weight('balanced', classes=classes, y=y_train.numpy())
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    best_loss = float("inf")

    with open("results/logs/training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])

        for epoch in range(1, 6):   # 🔥 Increased to 100 epochs
            model.train()
            epoch_loss = 0.0

            for x_text, x_struct, labels in train_loader:
                x_text = x_text.to(device)
                x_struct = x_struct.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(x_text, x_struct)
                loss = criterion(outputs, labels)
                loss.backward()

                # 🔹 Prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            writer.writerow([epoch, avg_loss])
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")

            # 🔹 Save best model only
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "results/models/qrnn_gru.pt")

    print("Training complete. Best model saved.")

if __name__ == "__main__":
    train()
