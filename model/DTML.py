import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, accuracy_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback



class TimeAxisAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False)
        self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, rt_attn=False):
        # x: (batch_size, window_size, input_size)
        o, (h, _) = self.lstm(x)


        h_last = h[-1]  # 取最后一层的隐藏状态 (batch_size, hidden_size)
        h_last = h_last.unsqueeze(-1)  # 增加维度 → (batch_size, hidden_size, 1)


        score = torch.bmm(o, h_last)  # (batch_size, window_size, hidden_size) x (batch_size, hidden_size, 1)
        score = score.squeeze(-1)     # → (batch_size, window_size)

        tx_attn = torch.softmax(score, dim=1)


        context = torch.bmm(
            tx_attn.unsqueeze(1),  # (batch_size, 1, window_size)
            o                       # (batch_size, window_size, hidden_size)
        ).squeeze(1)                # → (batch_size, hidden_size)

        normed_context = self.lnorm(context)
        if rt_attn:
            return normed_context, tx_attn
        else:
            return normed_context, None


class DataAxisAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, drop_rate=0.1):
        super().__init__()
        self.multi_attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.lnorm1 = nn.LayerNorm(hidden_size)
        self.lnorm2 = nn.LayerNorm(hidden_size)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, hm: torch.Tensor, rt_attn=False):
        # Forward Multi-head Attention
        residual = hm
        hm_hat, dx_attn = self.multi_attn(hm, hm, hm)
        hm_hat = self.lnorm1(residual + self.drop_out(hm_hat))

        # Forward FFN
        residual = hm_hat
        hp = torch.tanh(hm + hm_hat + self.mlp(hm + hm_hat))
        hp = self.lnorm2(residual + self.drop_out(hp))

        if rt_attn:
            return hp, dx_attn
        else:
            return hp, None


class DTML(pl.LightningModule):
    def __init__(self, input_size = 7, hidden_size = 64, num_layers = 2, n_heads = 4):
        super().__init__()
        self.input_size = input_size  # 5 价格特征 + 2 时间特征
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.tx_attention = TimeAxisAttention(input_size, hidden_size, num_layers)
        self.dx_attention = DataAxisAttention(hidden_size, n_heads)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"x.shape:{x.shape}")
        batch_size, seq_len, n_stocks, n_features = x.shape
        x = x.permute(0, 2, 1, 3)  # (batch, n_stocks, seq_len, features)

        stock_features = []
        for stock_idx in range(n_stocks):
            stock_seq = x[:, stock_idx, :, :]
            c_stock, _ = self.tx_attention(stock_seq)
            stock_features.append(c_stock)

        stock_features = torch.stack(stock_features, dim=1)
        context, _ = self.dx_attention(stock_features)
        logits = self.classifier(context).squeeze(-1)

        return logits
