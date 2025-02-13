import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(ResidualBlock, self).__init__()

        # 如果输入和输出的特征数不同，则需要一个线性层来调整维度
        self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SimpleMLPModel(nn.Module):
    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            # 使用 Xavier 正态分布初始化
            torch.nn.init.xavier_normal_(layer.weight)
        return layer

    def __init__(self, input_size=6, hidden_size=350, output_size=2, num_blocks=4):
        super(SimpleMLPModel, self).__init__()

        # 输入层
        self.input_layer = nn.Sequential(
            self.init_weights(nn.Linear(input_size, hidden_size)),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 定义多个残差块
        self.residual_layers = nn.ModuleList([ResidualBlock(hidden_size, hidden_size) for _ in range(num_blocks)])

        # 输出层
        self.output_layer = nn.Sequential(
            self.init_weights(nn.Linear(hidden_size, output_size))  # output_size=2， 直接输出 logits
        )

    def forward(self, x):
        x = x.squeeze(1)  # 假设输入是 [batch_size, 1, input_size]
        x = self.input_layer(x)

        # 通过所有残差块
        for residual_layer in self.residual_layers:
            x = residual_layer(x)

        # 输出 logits（不加 softmax 或 sigmoid）
        x = self.output_layer(x)
        return x
