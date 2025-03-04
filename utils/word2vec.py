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
warnings.filterwarnings("ignore")

local_model_path = "/home/users/liuyu/.cache/modelscope/hub/bert-base-cased/"
csv_news_path = "/home/users/liuyu/Framework/dataset/csi50_origional/news/"
embedding_path = "/home/users/liuyu/Framework/dataset/csi50_origional/news_embedding/"

#生成零向量并保存
def generate_zero_vector_and_save(ticker, date):
    zero_vector = np.zeros(3584)
    embedding_file_path = os.path.join(embedding_path, ticker, f"{date}.npy")
    os.makedirs(os.path.dirname(embedding_file_path), exist_ok=True)
    np.save(embedding_file_path, zero_vector)

def get_news_embedding():
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModel.from_pretrained(local_model_path)

    device = "cuda:1"

    model.to(device)

    # 将模型设置为评估模式
    model.eval()

    df_date = pd.read_csv("/home/users/liuyu/Framework/dataset/trading_date_list.csv")
    for date in df_date['Date']:
        for ticker in os.listdir(csv_news_path):
            old_ticker_path = os.path.join(csv_news_path, ticker)
            new_ticker_path = os.path.join(embedding_path, ticker)
            if not os.path.exists(new_ticker_path):
                os.makedirs(new_ticker_path)
            csv_news_date_path = os.path.join(old_ticker_path, f"{date}.csv")
            if not os.path.exists(csv_news_date_path):
                generate_zero_vector_and_save(ticker, date)
            else:
                sentence = None
                sentence_embedding = None
                df = pd.read_csv(csv_news_date_path)
                # 拼接text列中的所有行
                sentence = ' '.join(df['text'].dropna().astype(str))
                # 对句子进行分词并截断
                inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
                # 将输入数据移动到指定设备
                inputs = {k: v.to(device) for k, v in inputs.items()}
                # 通过模型进行前向传播
                with torch.no_grad():
                    outputs = model(**inputs)

                # 获取 [CLS] token 对应的向量作为句子表示
                sentence_embedding = outputs.last_hidden_state[:, 0, :]
                sentence_embedding = sentence_embedding.squeeze(0)
                sentence_embedding = sentence_embedding.detach()
                embedding_file_path = os.path.join(embedding_path, ticker, f"{date}.npy")
                os.makedirs(os.path.dirname(embedding_file_path), exist_ok=True)
                np.save(embedding_file_path, sentence_embedding.cpu().numpy())


# 生成向量并保存
get_news_embedding()


