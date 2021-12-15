import torch.nn as nn


class MusicLSTM(nn.Module):
    def __init__(self, hidden_dim):
        super(MusicLSTM, self).__init__()

        self.input_dim = 15
        self.hidden_dim = hidden_dim
        self.num_layers = 2

        self.lstm = nn.LSTM(
            input_size=self.input_dim, hidden_size=64,
            num_layers=self.num_layers, batch_first=True)

        self.dense_1 = nn.Linear(in_features=64,
                                 out_features=1024)

        self.lstm_2 = nn.LSTM(
            input_size=1024, hidden_size=256,
            num_layers=self.num_layers, batch_first=True
        )

        self.fc = nn.Linear(in_features=256,
                            out_features=self.input_dim)

        self.call_counter = 0

        self.double()

    def forward(self, timestep):
        lstm_out, _ = self.lstm(timestep)
        lstm_out = self.dense_1(lstm_out)
        lstm_out, _ = self.lstm_2(lstm_out)
        lstm_out = self.fc(lstm_out)
        self.call_counter += 1
        # if self.call_counter % 24 == 0:
        # print(f'lstm_out: {lstm_out}')
        # print(f'hidden_state: {hs}')
        # print(f'cell_state: {cs}')

        return lstm_out
