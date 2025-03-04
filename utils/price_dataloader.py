import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


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


class StockDataset(torch.utils.data.Dataset):
    def __init__(self, csv_files=[], look_back=1):
        super().__init__()
        self.load_csv_files(csv_files)

    def load_csv_files(
            self,
            csv_files=[],
            look_back=1,
            features_columns=['Open', 'High', 'Low', 'Close', 'Adj Close'],
            label_columns=['Label'],
    ):
        self.features, self.labels = [], []
        for csv_file in csv_files:
            feature, label = load_csv_file(csv_file, features_columns, label_columns)
            features = split_sequence(feature, look_back)
            labels = split_sequence(label, look_back)
            self.features.append(features)
            self.labels.append(labels)

        self.features = torch.concat(self.features, dim=0)
        self.labels = torch.concat(self.labels, dim=0)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label


def create_dataset(train_folder=None, val_folder=None, test_folder=None, look_back=1):
    def load_dataset(folder, look_back):

        if folder is not None:
            csv_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".csv")]
            return StockDataset(csv_files, look_back)
        return None

    # 分别加载训练、验证、测试数据集
    train_dataset = load_dataset(train_folder, look_back)
    val_dataset = load_dataset(val_folder, look_back)
    test_dataset = load_dataset(test_folder, look_back)

    return train_dataset, val_dataset, test_dataset

def create_dataloader(dataset, batch_size=32, shuffle=True, drop_last=False):

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader
