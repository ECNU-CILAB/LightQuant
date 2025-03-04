# author:Liu Yu
# time:2025/3/4 14:56
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, price_dir, news_dir, look_back_window=10):
        self.samples = []  # 存储 (合并特征, 标签)

        for stock_file in os.listdir(price_dir):
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

                # **保留时间维度**
                price_window = price_data.iloc[i - look_back_window:i].drop(["Date", "Label"], axis=1).values.astype(
                    np.float32)
                news_window = []
                for j in range(i - look_back_window, i):
                    news_date = price_data.iloc[j]["Date"]
                    if news_date in news_dict:
                        news_window.append(news_dict[news_date])
                    else:
                        news_window.append(np.zeros(10, dtype=np.float32))

                news_window = np.array(news_window)  # 变成 (10, 20)

                # **不要 flatten，保持时间维度**
                combined_features = np.concatenate([price_window, news_window], axis=-1)  # (10, 36)

                label = int(price_data.iloc[i]["Label"])
                self.samples.append((combined_features, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label = self.samples[idx]
        return {
            "feature": torch.tensor(features, dtype=torch.float32),  # (10, 36)
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