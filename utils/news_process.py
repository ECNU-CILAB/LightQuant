import time
from transformers import BertTokenizer, BertModel
from modelscope import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import warnings
import os
import pandas as pd
import json
import csv
import re
import numpy as np
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings("ignore")


class BiGRUEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, bidirectional=True):
        super(BiGRUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 双向 GRU
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        output, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim * 2)
        return output

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, bidirectional=True):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 双向 LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        output, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)
        return output


class ScaledDotProductAttentionWithSeq(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_q = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.W_k = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.W_v = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim * 2)
        returns: (batch_size, seq_len, hidden_dim * 2)
        """
        Q = self.W_q(x)  # (b, s, h*2)
        K = self.W_k(x)  # (b, s, h*2)
        V = self.W_v(x)  # (b, s, h*2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_dim * 2)
        attn_weights = self.softmax(attn_scores)  # (b, s, s)
        context_vector = torch.matmul(attn_weights, V)  # (b, s, h*2)
        return context_vector

class NewsEmbeddingModel(nn.Module):
    def __init__(self, encoder_type="gru", use_attention=True, input_dim=768, hidden_dim=256, output_dim=20):
        super(NewsEmbeddingModel, self).__init__()

        # 选择 Bi-GRU 或 Bi-LSTM
        if encoder_type == "gru":
            self.encoder = BiGRUEncoder(input_dim, hidden_dim)
        elif encoder_type == "lstm":
            self.encoder = BiLSTMEncoder(input_dim, hidden_dim)
        else:
            raise ValueError("Invalid encoder_type. Choose 'gru' or 'lstm'.")

        self.use_attention = use_attention
        if use_attention:
            self.attention = ScaledDotProductAttentionWithSeq(hidden_dim)
        else:
            self.attention = BiLSTMEncoder(input_dim=hidden_dim * 2, hidden_dim=hidden_dim)


        self.dim_reducer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ELU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )

    def forward(self, x):
        encoded_output = self.encoder(x)  # (batch_size, seq_len, hidden_dim * 2)

        if self.use_attention:

            context_vector = self.attention(encoded_output)
        else:
            context_vector = encoded_output

        # reduce dimension
        news_emb = self.dim_reducer(context_vector)  # (batch_size, seq_len, output_dim)
        return news_emb


