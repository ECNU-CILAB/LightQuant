# author:Liu Yu
# time:2025/2/11 18:31
import torch.nn as nn
import torch.nn.functional as F
import torch

import torch.nn as nn
import torch

class lstm(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1, dropout=0.2, batch_first=True):
        super(lstm, self).__init__()
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
        self.sigmoid = nn.Sigmoid()  # 添加Sigmoid激活层

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播LSTM
        out, _ = self.rnn(x, (h0, c0))

        # 取最后一个时间步的输出
        if self.batch_first:
            out = out[:, -1, :]
        else:
            out = out[-1, :, :]

        # 批归一化
        out = self.bn(out)

        # 线性层
        out = self.linear(out)

        # 应用Sigmoid激活函数
        out = self.sigmoid(out)

        return out
