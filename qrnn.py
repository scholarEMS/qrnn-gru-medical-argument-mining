import torch
import torch.nn as nn

class QRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel=2):
        super().__init__()
        self.conv_z = nn.Conv1d(input_dim, hidden_dim, kernel)
        self.conv_f = nn.Conv1d(input_dim, hidden_dim, kernel)
        self.conv_o = nn.Conv1d(input_dim, hidden_dim, kernel)

    def forward(self, x):
        x = x.transpose(1, 2)
        z = torch.tanh(self.conv_z(x))
        f = torch.sigmoid(self.conv_f(x))
        o = torch.sigmoid(self.conv_o(x))

        h = []
        c = torch.zeros_like(z[:, :, 0])
        for t in range(z.size(2)):
            c = f[:, :, t] * c + (1 - f[:, :, t]) * z[:, :, t]
            h.append(o[:, :, t] * c)

        return torch.stack(h, dim=1)
