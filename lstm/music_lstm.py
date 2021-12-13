import torch.nn as nn


class MusicLSTM(nn.Module):
    def __init__(self, hidden_dim):
        super(MusicLSTM, self).__init__()

        self.input_dim = 15
        self.hidden_dim = hidden_dim
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=self.input_dim, hidden_size=hidden_dim,
            num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Linear(in_features=hidden_dim,
                            out_features=self.input_dim)

        self.double()

    def forward(self, timestep):
        # print(f'timestep: {timestep}')
        lstm_out, _ = self.lstm(timestep)
        lstm_out = self.fc(lstm_out)
        # print(f'lstm_out: {lstm_out}')

        return lstm_out
