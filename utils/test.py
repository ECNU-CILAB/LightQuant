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
                news_date = price_data.iloc[j]["Date"].strftime("%Y-%m-%d")
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