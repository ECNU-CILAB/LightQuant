import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm

def load_csv_file(csv_file, feature_columns, label_columns):
    dataframe = pd.read_csv(csv_file)

    standardized_dataframe = standardize_dataframe(dataframe, feature_columns)

    feature = torch.tensor(standardized_dataframe[feature_columns].values, dtype=torch.float32)
    label = torch.tensor(standardized_dataframe[label_columns].values, dtype=torch.float32)
    return feature, label


def split_sequence(sequence, length):
    sequences = []
    for i in range(len(sequence) - length + 1):
        sequences.append(sequence[i: i + length])
    sequences = torch.stack(sequences)
    return sequences


def standardize_dataframe(dataframe, feature_columns):

    for col in feature_columns:
        mean = dataframe[col].mean()
        std = dataframe[col].std()
        dataframe[col] = (dataframe[col] - mean) / std
    return dataframe


class StockDataset(Dataset):
    def __init__(self, price_dir, news_dir, look_back_window):
        self.samples = []  # 存储 (市场特征, 新闻特征, 标签)

        for stock_file in tqdm(os.listdir(price_dir)):
            if not stock_file.endswith(".csv"):
                continue

            stock_id = stock_file.replace(".csv", "")
            price_path = os.path.join(price_dir, stock_file)
            price_data = pd.read_csv(price_path)

            if "Date" not in price_data.columns or "Label" not in price_data.columns:
                raise ValueError(f"{stock_file} 缺少 'Date' 或 'Label' 列")

            price_data = price_data.sort_values(by="Date").reset_index(drop=True)

            # 预加载所有新闻数据
            news_dict = {}
            news_folder = os.path.join(news_dir, stock_id)
            if os.path.exists(news_folder):
                for news_file in os.listdir(news_folder):
                    if news_file.endswith(".npy"):
                        news_date = news_file.replace(".npy", "")
                        news_path = os.path.join(news_folder, news_file)
                        news_dict[news_date] = np.load(news_path).astype(np.float32)

            for i in range(look_back_window, len(price_data)):
                date = price_data.iloc[i]["Date"]

                # 市场数据窗口
                market_window = price_data.iloc[i - look_back_window:i].drop(["Date", "Label", "Volume"], axis=1).values.astype(np.float32)  # (look_back_window, market_feature_dim)

                # 新闻数据窗口
                news_window = []
                for j in range(i - look_back_window, i):
                    news_date = price_data.iloc[j]["Date"]
                    news = np.zeros(20, dtype=np.float32)
                    if news_date in news_dict:
                        news = news_dict[news_date]
                        news = news.flatten()
                    news_window.append(news)
                news_window = np.array(news_window)  # (look_back_window, news_feature_dim)

                label = int(price_data.iloc[i]["Label"])
                self.samples.append((market_window, news_window, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        market_features, news_features, label = self.samples[idx]
        return {
            "market": torch.tensor(market_features, dtype=torch.float32),  # (look_back_window, market_feature_dim)
            "news": torch.tensor(news_features, dtype=torch.float32),      # (look_back_window, news_feature_dim)
            "label": torch.tensor(label, dtype=torch.long),
        }

class StockDatasetFromPickle(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            self.samples = pickle.load(f)  # 加载 samples 数据

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        market_features, news_features, label = self.samples[idx]
        return {
            "market": torch.tensor(market_features, dtype=torch.float32),
            "news": torch.tensor(news_features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }

def create_dataloader(dataset, batch_size=32, shuffle=True, drop_last=False):

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader

def create_dataset():
    for mode in ["train", "val", "test"]:
        price_dir = f"/home/users/liuyu/Framework/dataset/csi300/{mode}/price"
        news_dir = f"/home/users/liuyu/Framework/dataset/csi300/{mode}/news_embedding"
        dataset = StockDataset(price_dir, news_dir, look_back_window=10)

        with open(f"/home/users/liuyu/Framework/dataset/csi300/{mode}/dataset.pkl", "wb") as f:
            pickle.dump(dataset.samples, f)

# create_dataset()