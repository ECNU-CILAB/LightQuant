# author:Liu Yu
# time:2025/3/6 20:52
import torch.nn as nn
import torch

import torch.nn as nn
import torch

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, batch_first):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # 双向 GRU 层
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0  # 只有层数大于1时才应用 dropout
        )

        # 全连接层
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # 第一层全连接
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # 第二层全连接
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)  # 第三层全连接
        self.fc4 = nn.Linear(hidden_size // 4, output_size)  # 输出层

        # 非线性激活函数
        self.relu = nn.ReLU()

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirection

        # GRU 前向传播
        out, _ = self.bigru(x, h0)

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层 + Dropout + 激活函数
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc4(out)  # 输出层不需要激活函数
        return out
