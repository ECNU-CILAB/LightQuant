import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np
from transformers import AutoTokenizer, BertTokenizer
from pathlib import Path
from typing import Tuple, List

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


class StockNet_Dataset(Dataset):
    def __init__(self, price_dir, news_dir, look_back_window):
        self.samples = []

        for stock_file in tqdm(os.listdir(price_dir)):
            if not stock_file.endswith(".csv"):
                continue

            stock_id = stock_file.replace(".csv", "")
            price_path = os.path.join(price_dir, stock_file)
            price_data = pd.read_csv(price_path)

            if "Date" not in price_data.columns or "Label" not in price_data.columns:
                raise ValueError()

            price_data = price_data.sort_values(by="Date").reset_index(drop=True)


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


                market_window = price_data.iloc[i - look_back_window:i].drop(["Date", "Label", "Volume"], axis=1).values.astype(np.float32)  # (look_back_window, market_feature_dim)


                news_window = []
                for j in range(i - look_back_window, i):
                    news_date = price_data.iloc[j]["Date"]
                    news = np.zeros(768, dtype=np.float32)
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
            self.samples = pickle.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        market_features, news_features, label = self.samples[idx]
        return {
            "market": torch.tensor(market_features, dtype=torch.float32),
            "news": torch.tensor(news_features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


class HAN_Dataset(Dataset):
    def __init__(self, x_path, y_path, days, max_num_tweets, max_num_tokens, num_class=2):
        # read dataset
        self.x = torch.tensor(np.loadtxt(x_path, delimiter=',').reshape(-1, days, max_num_tweets, 3, max_num_tokens), dtype=torch.int64)
        self.y = torch.tensor(np.loadtxt(y_path, delimiter=','), dtype=torch.int64)
        self.class_weights = self.y.shape[0] / (num_class * torch.bincount(self.y.int()))

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def HAN_Dataset_Prepare(args):

    price_filenames = os.listdir(f"./dataset/{args.dataset}/price/")
    news_stock_folders = os.listdir(f"./dataset/{args.dataset}/news/")
    stock_name_price = set([filename.split('.')[0] for filename in price_filenames])
    stock_name_news = set(news_stock_folders)
    stock_names = set.intersection(stock_name_news, stock_name_price)


    start = datetime.strptime(args.train_start_date, '%Y-%m-%d')
    end = datetime.strptime(args.test_end_date, '%Y-%m-%d')
    date_list = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    y = pd.DataFrame(index=date_list, columns=list(stock_names))


    for filename in price_filenames:
        stock_name = filename.split(".")[0]
        if stock_name not in stock_names:
            continue
        filepath = os.path.join(f"./dataset/{args.dataset}/price/", filename)
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        df['move_per'] = df['Adj Close'].pct_change().shift(-1)
        for index, move_per in zip(df.index, df['move_per']):
            if index in y.index:
                y.at[index, stock_name] = move_per

    # 4. 标签离散化
    y[(-0.005 <= y) & (y <= 0.0055)] = float('nan')
    y[y > 0.0055] = 1
    y[y < -0.005] = 0

    # 5. 加载BERT分词器，处理新闻数据
    BERT_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    news_data = dict()
    for stock_name in stock_names:
        print(stock_name + ' token')
        stock_news_dir = os.path.join(f"./dataset/{args.dataset}/news/", stock_name)
        if not os.path.isdir(stock_news_dir):
            continue
        for file_name in os.listdir(stock_news_dir):
            if not file_name.endswith('.csv'):
                continue
            file_path = os.path.join(stock_news_dir, file_name)

            news_date = file_name.replace('.csv', '')
            try:
                news_date_obj = datetime.strptime(news_date, '%Y-%m-%d').date()
            except Exception as e:
                print(f"文件 {file_path} 命名不是日期格式，已跳过。")
                continue
            df_news = pd.read_csv(file_path)
            text_data = [str(row['text']) for idx, row in df_news.iterrows()]

            text_data = text_data[:args.max_num_tweets_len]
            text_data += [''] * (args.max_num_tweets_len - len(text_data))
            tokens = BERT_tokenizer(
                text_data,
                max_length=args.max_num_tokens_len,
                truncation=True,
                padding='max_length',
            )
            key = stock_name + ' + ' + str(news_date_obj)
            news_data[key] = tokens


    train_x = pd.DataFrame()
    train_y = pd.DataFrame()
    val_x = pd.DataFrame()
    val_y = pd.DataFrame()
    test_x = pd.DataFrame()
    test_y = pd.DataFrame()

    train_start_date = datetime.strptime(args.train_start_date, '%Y-%m-%d')
    train_end_date = datetime.strptime(args.train_end_date, '%Y-%m-%d')
    val_start_date = datetime.strptime(args.val_start_date, '%Y-%m-%d')
    val_end_date = datetime.strptime(args.val_end_date, '%Y-%m-%d')
    test_start_date = datetime.strptime(args.test_start_date, '%Y-%m-%d')
    test_end_date = datetime.strptime(args.test_end_date, '%Y-%m-%d')

    num_filtered_samples = 0
    for stock_name in stock_names:
        print(stock_name)
        for target_date in date_list:
            if y.at[target_date, stock_name] not in (0, 1):
                continue
            sample = np.zeros((args.days, args.max_num_tweets_len, 3, args.max_num_tokens_len))
            num_no_news_days = 0
            for lag in range(args.days, 0, -1):
                news_date = target_date - timedelta(days=lag)
                key = stock_name + ' + ' + str(news_date.date())
                if key in news_data:
                    news_ids = news_data[key]
                    sample[args.days - lag, :, 0, :] = np.array(news_ids['input_ids'])
                    sample[args.days - lag, :, 1, :] = np.array(news_ids['token_type_ids'])
                    sample[args.days - lag, :, 2, :] = np.array(news_ids['attention_mask'])
                else:
                    num_no_news_days += 1
                    if num_no_news_days > 1:
                        break
            if num_no_news_days > 1:
                num_filtered_samples += 1
                continue
            label = y.at[target_date, stock_name]
            if train_start_date <= target_date <= train_end_date:
                train_x = pd.concat(
                    [train_x, pd.DataFrame(np.expand_dims(np.ravel(sample), axis=0), index=[target_date])])
                train_y = pd.concat([train_y, pd.DataFrame([label], index=[target_date])])
            elif val_start_date <= target_date <= val_end_date:
                val_x = pd.concat([val_x, pd.DataFrame(np.expand_dims(np.ravel(sample), axis=0), index=[target_date])])
                val_y = pd.concat([val_y, pd.DataFrame([label], index=[target_date])])
            elif test_start_date <= target_date <= test_end_date:
                test_x = pd.concat(
                    [test_x, pd.DataFrame(np.expand_dims(np.ravel(sample), axis=0), index=[target_date])])
                test_y = pd.concat([test_y, pd.DataFrame([label], index=[target_date])])

    save_path = f"./dataset/{args.dataset}"
    dir = Path(save_path)
    dir.mkdir(parents=True, exist_ok=True)

    train_x.to_csv(os.path.join(save_path, 'train_x.csv'), index=False, header=False)
    train_y.to_csv(os.path.join(save_path, 'train_y.csv'), index=False, header=False)
    val_x.to_csv(os.path.join(save_path, 'val_x.csv'), index=False, header=False)
    val_y.to_csv(os.path.join(save_path, 'val_y.csv'), index=False, header=False)
    test_x.to_csv(os.path.join(save_path, 'test_x.csv'), index=False, header=False)
    test_y.to_csv(os.path.join(save_path, 'test_y.csv'), index=False, header=False)


class PEN_Dataset(Dataset):
    def __init__(
            self,
            text_data_path: str,
            price_data_path: str,
            days: int = 5,
            max_num_tweets: int = 20,
            max_num_tokens: int = 30,
            pretrained_model: str = "bert-base-uncased",
            normalize_price: bool = True,
            label_col: str = "Label"
    ):

        super(PEN_Dataset, self).__init__()
        self.days = days
        self.max_num_tweets = max_num_tweets
        self.max_num_tokens = max_num_tokens
        self.normalize_price = normalize_price
        self.label_col = label_col


        self.text_data = self._load_text_data(text_data_path)
        self.price_data, self.labels = self._load_price_and_labels(price_data_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)


        assert len(self.text_data) == len(self.price_data) == len(self.labels), "文本、价格和标签数据样本数不一致"

    def _load_text_data(self, path: str) :

        text_data = []
        if path.endswith('.csv'):
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            date_groups = df.groupby(df['Date'])
            dates = sorted(date_groups.groups.keys())
            for i in range(len(dates) - self.days + 1):
                sample_texts = []
                for d in dates[i:i + self.days]:
                    day_texts = date_groups.get_group(d)['text'].tolist()
                    sample_texts.append(day_texts[:self.max_num_tweets])
                text_data.append(sample_texts)
        else:
            raise ValueError(f"不支持的文本数据格式: {path}")
        return text_data

    def _load_price_and_labels(self, path: str) -> Tuple[np.ndarray, np.ndarray]:

        if not os.path.exists(path):
            raise FileNotFoundError(f"价格数据文件不存在: {path}")

        df = pd.read_csv(path)
        if self.label_col not in df.columns:
            raise ValueError(f"价格CSV中缺少标签列: {self.label_col}")

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('Date')


        price_cols = ['Adj_close', 'High', 'Low']
        if not all(col in df.columns for col in price_cols):
            raise ValueError(f"价格数据缺少必要列，需包含: {price_cols}")


        price_samples = []
        label_samples = []
        for i in range(len(df) - self.days + 1):

            day_prices = df[i:i + self.days][price_cols].values
            price_samples.append(day_prices)


            day_labels = []
            for day in range(self.days):
                if day == 0:
                    day_labels.append(0)
                else:

                    label = df.iloc[i + day][self.label_col]
                    day_labels.append(1 if label > 0 else 0)
            label_samples.append(day_labels)


        if self.normalize_price:
            price_samples = self._normalize_prices(price_samples)

        return np.array(price_samples, dtype=np.float32), np.array(label_samples, dtype=np.int64)

    def _normalize_prices(self, price_samples: List[np.ndarray]):

        normalized_samples = []
        for sample in price_samples:
            normalized_day = []
            for day in range(self.days):
                if day == 0:
                    normalized_day.append(sample[day] / 1 - 1)
                else:
                    normalized_day.append(sample[day] / sample[day - 1] - 1)
            normalized_samples.append(np.array(normalized_day))
        return normalized_samples

    def _tokenize_and_pad(self, text: str):

        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_num_tokens,
            padding="max_length",
            truncation=True
        )
        return tokens

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        text_sample = self.text_data[idx]
        tokenized_texts = []
        for day_texts in text_sample:
            day_tokens = []
            for text in day_texts:
                day_tokens.append(self._tokenize_and_pad(text))
            while len(day_tokens) < self.max_num_tweets:
                day_tokens.append([0] * self.max_num_tokens)
            tokenized_texts.append(day_tokens)
        text_inputs = torch.tensor(tokenized_texts, dtype=torch.long)
        price_series = torch.tensor(self.price_data[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return text_inputs, price_series, labels

    def __len__(self) -> int:

        return len(self.labels)



def create_dataloader(dataset, batch_size=32, shuffle=True, drop_last=False):

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader


def create_dataset(price_file=None, news_folder=None, look_back_window=7):

    if price_file is not None and news_folder is not None:
        return Backtest_Dataset(price_file, news_folder, look_back_window)


class Backtest_Dataset(Dataset):
    def __init__(
        self,
        price_file: str,
        news_folder: str,
        look_back_window: int = 7,
        feature_columns: list = None,
        label_column: str = "Label",
        news_dim: int = 768
    ):
        """
        Args:
            price_file (str): 单个股票的 CSV 文件路径。
            news_folder (str): 存放该股票新闻嵌入的目录，内含.npy文件。
            look_back_window (int): 时间窗口长度。
            feature_columns (list): 用于模型输入的价格特征列。
            label_column (str): 标签列名。
            news_dim (int): 新闻嵌入维度（默认为 BERT 的 768）
        """
        super().__init__()
        self.look_back_window = look_back_window
        self.feature_columns = feature_columns or ['Open', 'High', 'Low', 'Close', 'Adj Close']
        self.label_column = label_column
        self.news_dim = news_dim
        self.samples = []

        # 加载价格数据
        stock_id = os.path.splitext(os.path.basename(price_file))[0]
        price_data = pd.read_csv(price_file)

        if "Date" not in price_data.columns or label_column not in price_data.columns:
            raise ValueError(f"{stock_id}.csv 缺少必要列")

        price_data = price_data.sort_values(by="Date").reset_index(drop=True)

        # 构建新闻字典
        news_dict = {}
        if os.path.exists(news_folder):
            for news_file in os.listdir(news_folder):
                if news_file.endswith(".npy"):
                    news_date = news_file.replace(".npy", "")
                    news_path = os.path.join(news_folder, news_file)
                    news = np.load(news_path).astype(np.float32)
                    news_dict[news_date] = news.flatten()  # shape: (news_dim, )

        # 构建样本
        for i in range(look_back_window, len(price_data)):
            date = price_data.iloc[i]["Date"]

            raw_window = price_data.iloc[i - look_back_window:i].copy()
            market_window = raw_window[self.feature_columns].values.astype(np.float32)
            normalized_window = standardize_dataframe(raw_window.copy(), self.feature_columns)[
                self.feature_columns].values.astype(np.float32)

            # 新闻数据：按日期匹配并填充
            news_window = []
            for j in range(i - look_back_window, i):
                news_date = price_data.iloc[j]["Date"]
                if news_date in news_dict:
                    news = news_dict[news_date]
                else:
                    news = np.zeros(self.news_dim, dtype=np.float32)
                news_window.append(news)
            news_window = np.array(news_window)  # shape: (look_back_window, news_dim)

            label = int(price_data.iloc[i][self.label_column])

            self.samples.append((
                torch.tensor(normalized_window, dtype=torch.float32),
                torch.tensor(market_window, dtype=torch.float32),
                torch.tensor(news_window, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# def create_dataset(dataset, look_back_window):
#     for mode in ["train", "val", "test"]:
#         price_dir = f"../dataset/{dataset}/{mode}/price"
#         news_dir = f"../dataset/{dataset}/{mode}/news_embedding"
#         stock_dataset = StockNet_Dataset(price_dir, news_dir, look_back_window)
#
#         with open(f"../dataset/{dataset}/{mode}/dataset.pkl", "wb") as f:
#             pickle.dump(stock_dataset.samples, f)

