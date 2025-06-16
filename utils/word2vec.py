from modelscope import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import warnings
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
warnings.filterwarnings("ignore")

def generate_zero_vector_and_save(ticker, date, embedding_path):
    zero_vector = np.zeros(20)
    embedding_file_path = os.path.join(embedding_path, ticker, f"{date}.npy")
    os.makedirs(os.path.dirname(embedding_file_path), exist_ok=True)
    np.save(embedding_file_path, zero_vector)

def get_news_embedding(csv_news_path, embedding_path, local_model_path, use_attention=True, look_back_days=7, trading_date_list=None):

    device = torch.device("cuda:1")

    # 加载 BERT
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    bert_model = AutoModel.from_pretrained(local_model_path).to(device)
    bert_model.eval()

    # 读取交易日期
    df_date = pd.read_csv("../dataset/trading_date_list.csv")

    for date in tqdm(df_date['Date']):
        for ticker in os.listdir(csv_news_path):
            old_ticker_path = os.path.join(csv_news_path, ticker)
            new_ticker_path = os.path.join(embedding_path, ticker)
            os.makedirs(new_ticker_path, exist_ok=True)

            embedding_file_path = os.path.join(new_ticker_path, f"{date}.npy")


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


            np.save(embedding_file_path, past_news_embeddings.cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="CSMD50", type=str)
    parser.add_argument("--csv_news_path", type=str, default="../dataset/{}/news/".format(parser.get_default('dataset')))
    parser.add_argument("--embedding_path", type=str, default="../dataset/{}/news_embedding/".format(parser.get_default('dataset')))
    parser.add_argument("--local_model_path", type=str, default="your model path")
    parser.add_argument("--trading_date_list", type=str, default="../dataset/trading_date_list.csv")
    args = parser.parse_args()

    get_news_embedding(csv_news_path=args.csv_news_path,
                       embedding_path=args.embedding_path,
                       local_model_path=args.local_model_path,
                       trading_date_list=args.trading_date_list)


