import torch.nn as nn

class Model(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super(Model, self).__init__()

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        # For the type of dataset is TSDataset, the size of x is [N, T, F]
        N, T, F = x.size()
        out, _ = self.rnn(x)
        fc_out = self.fc_out(out[:, -1, :])
        out = fc_out.reshape(N, -1).squeeze(dim=1)
        return out