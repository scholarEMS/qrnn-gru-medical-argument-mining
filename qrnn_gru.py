import torch
import torch.nn as nn
from qrnn import QRNN


class QRNN_GRU(nn.Module):
    def __init__(self, input_dim, qrnn_dim, gru_dim, num_classes):
        super().__init__()

        # 🔹 Text pathway (QRNN → GRU)
        self.qrnn = QRNN(input_dim, qrnn_dim)
        self.gru = nn.GRU(qrnn_dim, gru_dim, batch_first=True)

        self.text_dropout = nn.Dropout(0.5)

        # 🔹 Structured feature pathway (3 encoded features)
        self.struct_fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 🔹 Final classifier
        self.fc = nn.Linear(gru_dim + 32, num_classes)

    def forward(self, x_text, x_struct):
        # ----- Text branch -----
        q = self.qrnn(x_text)             # (batch, seq_len, qrnn_dim)
        g, _ = self.gru(q)                # (batch, seq_len, gru_dim)

        # Mean pooling across time steps
        g = torch.mean(g, dim=1)          # (batch, gru_dim)
        g = self.text_dropout(g)

        # ----- Structured branch -----
        s = self.struct_fc(x_struct.float())  # (batch, 32)

        # ----- Fusion -----
        combined = torch.cat([g, s], dim=1)   # (batch, gru_dim + 32)

        return self.fc(combined)
