import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_csv_file(csv_file, feature_columns, label_columns):
    dataframe = pd.read_csv(csv_file)
    # print(csv_file)
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


class Normal_Dataset(Dataset):
    def __init__(self, csv_files=[], look_back_window=1):
        super().__init__()
        self.load_csv_files(csv_files)
        self.mean = None
        self.std = None

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

class Backtest_Dataset(Dataset):
    def __init__(self, csv_file, look_back_window=7, feature_columns=None, label_column='Label'):
        super().__init__()
        self.look_back_window = look_back_window
        self.feature_columns = feature_columns or ['Open', 'High', 'Low', 'Close', 'Adj Close']
        self.label_column = label_column

        df = pd.read_csv(csv_file)
        self.dates = df['Date'].values[look_back_window - 1:]


        raw_close = df['Close'].values
        self.raw_close_prices = raw_close

        # 特征
        raw_features = df[self.feature_columns].values
        labels = df[label_column].values


        standardized_df = standardize_dataframe(df.copy(), self.feature_columns)
        normalized_features = standardized_df[self.feature_columns].values


        self.normalized_sequences = []
        self.raw_sequences = []
        self.labels = []

        for i in range(len(raw_features) - look_back_window + 1):
            norm_seq = normalized_features[i:i + look_back_window]
            raw_seq = raw_features[i:i + look_back_window]
            label = labels[i + look_back_window - 1]

            self.normalized_sequences.append(norm_seq)
            self.raw_sequences.append(raw_seq)
            self.labels.append(label)

    def __len__(self):
        return len(self.normalized_sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.normalized_sequences[idx], dtype=torch.float32),
            torch.tensor(self.raw_sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

    def get_dates(self):
        return self.dates[:len(self.normalized_sequences)]

def create_dataset(train_folder=None, val_folder=None, test_folder=None, backtest_file=None, look_back_window=1):
    def load_dataset_from_folder(folder, look_back_window):

        if folder is None:
            return None
        csv_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".csv")]
        return Normal_Dataset(csv_files, look_back_window)

    def load_dataset_from_csv(csv_file, look_back_window):
        if csv_file is None:
            return None
        return Backtest_Dataset(csv_file, look_back_window)

    if backtest_file is not None:
        return load_dataset_from_csv(backtest_file, look_back_window)
    else:
        train_dataset = load_dataset_from_folder(train_folder, look_back_window)
        val_dataset = load_dataset_from_folder(val_folder, look_back_window)
        test_dataset = load_dataset_from_folder(test_folder, look_back_window)

        return train_dataset, val_dataset, test_dataset

class DTML_Dataset(Dataset):
    def __init__(self, data_folder, look_back_window=7, n_stocks=5):
        self.data_folder = data_folder
        self.seq_len = look_back_window
        self.n_stocks = n_stocks


        self.csv_files = [
            os.path.join(data_folder, f)
            for f in os.listdir(data_folder)
            if f.endswith('.csv')
        ]
        if not self.csv_files:
            raise ValueError(f"No CSV files found in {data_folder}")


        self.processed_data = []
        for csv_file in self.csv_files:
            self._process_single_stock(csv_file)


        np.random.shuffle(self.processed_data)

    def _process_single_stock(self, csv_path):

        df = pd.read_csv(csv_path)


        df['Date'] = pd.to_datetime(df['Date'])
        df['weekday'] = df['Date'].dt.weekday
        df['month'] = df['Date'].dt.month


        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        scaler = StandardScaler()
        df[price_cols] = scaler.fit_transform(df[price_cols])


        features = df[price_cols + ['weekday', 'month']].values
        closes = df['Close'].values
        labels = [int(closes[i] > closes[i - 1]) for i in range(1, len(closes))]
        labels = [0] + labels

        # 分割序列
        for i in range(len(df) - self.seq_len):
            seq_feat = features[i: i + self.seq_len]
            label = labels[i + self.seq_len]
            self.processed_data.append((seq_feat, label))

    def __len__(self):
        return len(self.processed_data) // self.n_stocks

    def __getitem__(self, idx):

        sampled_indices = np.random.choice(
            len(self.processed_data),
            self.n_stocks,
            replace=False
        )


        features, labels = [], []
        for i in sampled_indices:
            feat, lbl = self.processed_data[i]
            features.append(feat)
            labels.append(lbl)


        features = np.stack(features, axis=1)
        return (
            torch.FloatTensor(features),
            torch.FloatTensor(labels)
        )


class SCINet_Dataset(Dataset):
    def __init__(self, data_folder, seq_len, pred_len):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len


        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"NO such file or directory: {data_folder}")


        csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
                     if f.endswith('.csv')]


        if not csv_files:
            raise ValueError(f"no such file or directory: {data_folder}")

        self.features, self.labels = self._process_files(csv_files)


        if len(self.features) == 0 or len(self.labels) == 0:
            raise RuntimeError("dataset is empty")

    def _process_files(self, csv_files):
        all_features, all_labels = [], []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df = self._add_time_features(df)


                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close']
                if not all(col in df.columns for col in required_columns):
                    continue


                features, labels = self._generate_sequences(df)
                if features.size(0) == 0 or labels.size(0) == 0:
                    continue

                all_features.append(features)
                all_labels.append(labels)
            except Exception as e:
                continue

        if not all_features or not all_labels:
            raise RuntimeError("failed")

        return torch.cat(all_features), torch.cat(all_labels)

    def _add_time_features(self, df):
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df['day'] = df['Date'].dt.day
            df['weekday'] = df['Date'].dt.weekday
            df['month'] = df['Date'].dt.month
            return df
        except KeyError:
            raise KeyError("No 'Date' column")

    def _generate_sequences(self, df):

        min_required_length = self.seq_len + self.pred_len
        if len(df) < min_required_length:

            return torch.empty(0), torch.empty(0)


        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        scaler = StandardScaler()
        df[price_cols] = scaler.fit_transform(df[price_cols])


        feature_cols = price_cols + ['day', 'weekday', 'month']
        features = torch.tensor(df[feature_cols].values, dtype=torch.float32)


        close_prices = df['Close'].values
        labels = []
        for i in range(len(close_prices) - self.pred_len):
            future_closes = close_prices[i + 1: i + self.pred_len + 1]
            lbls = [1 if future_closes[j] > future_closes[j - 1] else 0
                    for j in range(1, len(future_closes))]
            labels.append(lbls)
        labels = torch.tensor(labels, dtype=torch.float32)


        seq_features, seq_labels = [], []
        for i in range(len(features) - self.seq_len - self.pred_len + 1):
            seq_feat = features[i: i + self.seq_len]
            seq_lbl = labels[i + self.seq_len - 1]
            seq_features.append(seq_feat)
            seq_labels.append(seq_lbl)

        return torch.stack(seq_features), torch.stack(seq_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class Backtest_SCINet_Dataset(Dataset):
    def __init__(self, csv_file, seq_len=90, pred_len=7, feature_columns=None, label_column='Label'):

        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_columns = feature_columns or ['Open', 'High', 'Low', 'Close', 'Adj Close', 'day', 'weekday', 'month']
        self.label_column = label_column

        df = pd.read_csv(csv_file)

        df['Date'] = pd.to_datetime(df['Date'])
        df['day'] = df['Date'].dt.day
        df['weekday'] = df['Date'].dt.weekday
        df['month'] = df['Date'].dt.month

        self.raw_close_prices = df['Close'].values


        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        scaler = StandardScaler()
        df[price_cols] = scaler.fit_transform(df[price_cols])


        features = df[self.feature_columns].values
        labels = []
        for i in range(len(df) - pred_len):
            future_closes = df['Close'].values[i+1:i+pred_len+1]
            direction = [1 if future_closes[j] > future_closes[j-1] else 0 for j in range(1, len(future_closes))]
            labels.append(direction)
        labels = np.array(labels)


        self.normalized_sequences = []
        self.labels = []

        for i in range(seq_len, len(features) - pred_len + 1):
            start_idx = max(i - seq_len, 0)
            end_idx = i
            seq_feat = features[start_idx:end_idx]
            seq_label = labels[i - 1] if i <= len(labels) else [0]

            self.normalized_sequences.append(seq_feat)
            self.labels.append(seq_label[0] if len(seq_label) > 0 else 0)


        self.normalized_sequences = torch.tensor(self.normalized_sequences, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.normalized_sequences)

    def __getitem__(self, idx):
        return (
            self.normalized_sequences[idx],
            torch.tensor(self.raw_close_prices[idx + self.seq_len], dtype=torch.float32),
            self.labels[idx]
        )

    def get_dates(self):

        df = pd.read_csv(self.csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df['Date'].values[self.seq_len:]
def create_dataloader(dataset, batch_size=32, shuffle=True, drop_last=False):

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader
