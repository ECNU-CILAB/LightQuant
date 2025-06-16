import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1, dropout=0.2, batch_first=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first

        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout)
        self.bn = nn.BatchNorm1d(self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)


        out, _ = self.rnn(x, (h0, c0))


        if self.batch_first:
            out = out[:, -1, :]
        else:
            out = out[-1, :, :]


        out = self.bn(out)


        out = self.linear(out)


        out = self.sigmoid(out)

        return out
