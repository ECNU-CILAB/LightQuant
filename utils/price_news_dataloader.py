# author:Liu Yu
# time:2025/3/4 14:56
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, price_dir, news_dir):
        """
        :param price_dir: 存储股票价格 CSV 文件的文件夹路径
        :param news_dir: 存储新闻 npy 文件的文件夹路径
        """
        self.samples = []  # 用于存储 (合并特征, 标签)

        # 遍历所有股票的价格 CSV 文件
        for stock_file in os.listdir(price_dir):
            if stock_file.endswith(".csv"):
                stock_id = stock_file.replace(".csv", "")
                price_path = os.path.join(price_dir, stock_file)

                # 读取股票价格 CSV 文件
                price_data = pd.read_csv(price_path)

                # 确保 CSV 至少包含 'date' 和 'label' 列
                if "Date" not in price_data.columns or "Label" not in price_data.columns:
                    raise ValueError(f"{stock_file} 缺少 'Date' 或 'Label' 列")

                # 遍历每天的数据，查找对应的新闻 NPY 文件
                for i, row in price_data.iterrows():
                    date = row["Date"]
                    news_path = os.path.join(news_dir, f"{stock_id}/{date}.npy")

                    if os.path.exists(news_path):  # 只有匹配到新闻数据的日期才添加
                        price_features = row.drop(["Date", "Label"]).values.astype(np.float32)
                        news_features = np.load(news_path).astype(np.float32)
                        combined_features = np.concatenate([price_features, news_features])  # 合并特征
                        label = int(row["Label"])
                        self.samples.append((combined_features, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label = self.samples[idx]

        return {
            "feature": torch.tensor(features, dtype=torch.float32),  # 16 + 20 = 36 维
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