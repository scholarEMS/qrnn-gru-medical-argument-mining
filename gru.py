import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    """
    Standalone GRU model for ablation comparison.
    Input: (batch_size, seq_len, embedding_dim)
    Output: logits for argument classes
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(GRUClassifier, self).__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)   # Last hidden state
        out = self.fc(h_n[-1]) # Use final layer hidden state
        return out
