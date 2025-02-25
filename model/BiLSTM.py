# author:Liu Yu
# time:2025/2/24 17:57
import torch.nn as nn
import torch.nn.functional as F
import torch

class BiLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, output_size=1, dropout=0.3, batch_first=True):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first

        # 设置bidirectional=True以创建BiLSTM
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout, bidirectional=True)
        # 由于是双向LSTM，线性层的输入大小需要乘以2
        self.bn = nn.BatchNorm1d(self.hidden_size * 2)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(self.hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 乘以2以适应双向LSTM
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 乘以2以适应双向LSTM

        # 前向传播BiLSTM
        out, _ = self.rnn(x, (h0, c0))

        # 取最后一个时间步的输出
        if self.batch_first:
            out = out[:, -1, :]
        else:
            out = out[-1, :, :]

        # 批归一化
        out = self.bn(out)

        # 添加Dropout层
        out = self.dropout_layer(out)

        # 线性层和ReLU激活函数
        out = self.linear1(out)
        out = self.relu(out)

        # 添加Dropout层
        out = self.dropout_layer(out)

        # 线性层
        out = self.linear2(out)

        # 应用Sigmoid激活函数
        out = self.sigmoid(out)

        return out
