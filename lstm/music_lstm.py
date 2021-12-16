import torch.nn as nn


class MusicLSTM(nn.Module):
    def __init__(self, hidden_dim):
        super(MusicLSTM, self).__init__()

        self.call_counter = 0
        self.input_dim = 15
        self.hidden_dim = hidden_dim
        self.num_layers = 1

        self.lstm_0 = nn.LSTM(
            input_size=self.input_dim, hidden_size=512,
            num_layers=self.num_layers, batch_first=True)

        self.dropout_0 = nn.Dropout(p=0.3)

        self.lstm_1 = nn.LSTM(
            input_size=512, hidden_size=512,
            num_layers=self.num_layers, batch_first=True
        )

        self.dropout_1 = nn.Dropout(p=0.3)

        self.lstm_2 = nn.LSTM(
            input_size=512, hidden_size=512,
            num_layers=self.num_layers, batch_first=True
        )

        self.dense_0 = nn.Linear(in_features=512,
                                 out_features=256)

        self.dropout_2 = nn.Dropout(p=0.3)

        self.dense_1 = nn.Linear(in_features=256,
                                 out_features=self.input_dim)

        self.double()

    def forward(self, timestep):
        out, _ = self.lstm_0(timestep)
        out = self.dropout_0(out)
        out, _ = self.lstm_1(out)
        out = self.dropout_1(out)
        out, _ = self.lstm_2(out)
        out = self.dense_0(out)
        out = self.dropout_2(out)
        out = self.dense_1(out)
        self.call_counter += 1

        # print(f'lstm_out: {out}')
        return out
