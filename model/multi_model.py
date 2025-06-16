import torch
import torch.nn as nn
import os
import sys
from utils.news_process import NewsEmbeddingModel
from model.LSTM import LSTM
from model.ALSTM import ALSTM
from model.BiLSTM import BiLSTM
from model.BiGRU import BiGRU

class MainModel(nn.Module):
    def __init__(self, market_input_size=5, news_emb_dim=20, hidden_size=64, num_layers=2,
                  output_size=2, dropout=0.2, attention_size=0, model=None):
        super(MainModel, self).__init__()

        # embedding model
        self.news_embedding = NewsEmbeddingModel(
            encoder_type="gru",
            input_dim=768,
            hidden_dim=256,
            output_dim=news_emb_dim
        )
        if model == "lstm":
            self.model = LSTM(
                input_size=market_input_size + news_emb_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout=dropout,
                batch_first=True
            )
        elif model == "alstm":
            self.model = ALSTM(
                input_size=market_input_size + news_emb_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout=dropout,
                batch_first=True,
                attention_size=attention_size
            )
        elif model == "bi_lstm":
            self.model = BiLSTM(
                input_size=market_input_size + news_emb_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout=dropout,
                batch_first=True
            )
        self.fusion_layer = AttentionFusion()

    def forward(self, market_feat, news_feat):
        """
        market: (batch, seq_len, market_input_size)
        news:   (batch, seq_len, 768)
        """
        # 新闻 embedding -> (batch, seq_len, news_emb_dim)
        news_feat = self.news_embedding(news_feat)
        fussed_feat = self.fusion_layer(market_feat, news_feat)

        # 拼接 market + news_feat 在 feature 维度
        # x = torch.cat([market_feat, news_feat], dim=-1)  # shape: (batch, seq_len, market_input_size + news_emb_dim)

        x = torch.cat([fussed_feat, market_feat], dim=-1)
        output = self.model(x)

        return output

class AttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 将市场和新闻映射到统一维度
        # self.market_proj = nn.Linear(market_dim, 64)
        # self.news_proj = nn.Linear(news_dim, 64)
        # 计算注意力权重
        self.attn = nn.Linear(10, 2)  # 两个模态的得分
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, market_feat, news_feat):
        # 投影到相同维度
        # market_proj = self.market_proj(market_feat)  # (batch, seq, 64)
        # news_proj = self.news_proj(news_feat)        # (batch, seq, 64)
        # 拼接
        concat_feat = torch.cat([market_feat, news_feat], dim=-1)  # (batch, seq, 128)
        scores = self.attn(concat_feat)  # (batch, seq, 2)
        attn_weights = self.softmax(scores)  # (batch, seq, 2)
        # 加权融合
        market_weight = attn_weights[..., 0:1]
        news_weight = attn_weights[..., 1:2]
        fused_market = market_weight * market_feat  # (batch, seq, market_dim)
        fused_news = news_weight * news_feat        # (batch, seq, news_dim)
        # 最终融合
        fused_feat = fused_market + fused_news  # 或者用平均、拼接等
        return fused_feat