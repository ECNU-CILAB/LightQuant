# author:Liu Yu
# time:2025/3/4 9:59
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
        """
        x: (batch_size, seq_len, input_dim)
        output: (batch_size, seq_len, hidden_dim * 2)  # 双向拼接
        """
        output, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim * 2)
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_dim = hidden_dim

        # 定义 Q, K, V 的线性变换
        self.W_q = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.W_k = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.W_v = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        self.softmax = nn.Softmax(dim=-1)  # 对 attention score 进行 softmax 归一化

    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim * 2)
        output: (batch_size, hidden_dim * 2)  # 注意力加权的最终表示
        """
        Q = self.W_q(x)  # (batch_size, seq_len, hidden_dim * 2)
        K = self.W_k(x)  # (batch_size, seq_len, hidden_dim * 2)
        V = self.W_v(x)  # (batch_size, seq_len, hidden_dim * 2)

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(
            self.hidden_dim * 2)  # (batch_size, seq_len, seq_len)

        # 计算 softmax 归一化的 attention 权重
        attn_weights = self.softmax(attn_scores)  # (batch_size, seq_len, seq_len)

        # 用 attention 权重加权 V
        context_vector = torch.matmul(attn_weights, V)  # (batch_size, seq_len, hidden_dim * 2)

        # 取 Attention 加权和
        output = torch.sum(context_vector, dim=1)  # (batch_size, hidden_dim * 2)

        return output

class NewsEmbeddingModel(nn.Module):
    def __init__(self, use_attention=True, input_dim=768, hidden_dim=256, output_dim=20):
        super(NewsEmbeddingModel, self).__init__()

        self.encoder = BiGRUEncoder(input_dim, hidden_dim)

        self.use_attention = use_attention
        if use_attention:
            self.attention = ScaledDotProductAttention(hidden_dim)

        self.dim_reducer = nn.Linear(hidden_dim * 2, output_dim)  # MLP降维
    def forward(self, x):
        encoded_output = self.encoder(x)  # (batch_size, seq_len, hidden_dim * 2)

        if self.use_attention:
            context_vector = self.attention(encoded_output)  # (batch_size, hidden_dim * 2)
        else:
            context_vector = encoded_output[:, -1, :]  # 取最后时间步的隐藏状态

        return self.dim_reducer(context_vector)  # (batch_size, output_dim)
#生成零向量并保存
def generate_zero_vector_and_save(ticker, date, embedding_path):
    zero_vector = np.zeros(20)
    embedding_file_path = os.path.join(embedding_path, ticker, f"{date}.npy")
    os.makedirs(os.path.dirname(embedding_file_path), exist_ok=True)
    np.save(embedding_file_path, zero_vector)

def get_news_embedding(csv_news_path, embedding_path, local_model_path, use_attention=True, look_back_days=10):

    device = torch.device("cuda:1")

    # 加载 BERT
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    bert_model = AutoModel.from_pretrained(local_model_path).to(device)
    bert_model.eval()

    # 加载 News Embedding 模型
    news_model = NewsEmbeddingModel(use_attention=use_attention).to(device)
    news_model.eval()

    # 读取交易日期
    df_date = pd.read_csv("/home/users/liuyu/Framework/dataset/trading_date_list.csv")

    for date in tqdm(df_date['Date']):
        for ticker in os.listdir(csv_news_path):
            old_ticker_path = os.path.join(csv_news_path, ticker)
            new_ticker_path = os.path.join(embedding_path, ticker)
            os.makedirs(new_ticker_path, exist_ok=True)

            embedding_file_path = os.path.join(new_ticker_path, f"{date}.npy")

            # 存储过去 `look_back_days` 天的新闻
            past_news_embeddings = []
            for i in range(look_back_days, -1, -1):
                past_date_index = df_date[df_date['Date'] == date].index[0] - i
                if past_date_index < 0:
                    continue
                past_date = df_date.iloc[past_date_index]['Date']
                csv_news_date_path = os.path.join(old_ticker_path, f"{past_date}.csv")

                if os.path.exists(csv_news_date_path):
                    df = pd.read_csv(csv_news_date_path)
                    sentence = '. '.join(df['text'].dropna().astype(str))
                    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = bert_model(**inputs)
                    sentence_embedding = outputs.last_hidden_state[:, 0, :]
                    past_news_embeddings.append(sentence_embedding.cpu().numpy())

            if len(past_news_embeddings) == 0:
                generate_zero_vector_and_save(ticker, date, embedding_path)
                continue

            past_news_embeddings = np.stack(past_news_embeddings, axis=1)
            past_news_embeddings = torch.tensor(past_news_embeddings, dtype=torch.float32).to(device)

            with torch.no_grad():
                final_embedding = news_model(past_news_embeddings)

            np.save(embedding_file_path, final_embedding.cpu().numpy())

# 生成向量并保存
get_news_embedding(csv_news_path="/home/users/liuyu/Framework/dataset/csi300_origional/news/",
                   embedding_path="/home/users/liuyu/Framework/dataset/csi300_origional/news_embedding/",
                   local_model_path="/home/users/liuyu/.cache/modelscope/hub/GTE_Base/")


