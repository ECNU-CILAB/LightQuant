# author:Liu Yu
# time:2025/2/11 15:13
import os
import shutil
from datetime import datetime
import pandas as pd

def split_data():
    #划分训练集,验证集,测试集
    # 定义日期范围
    train_end_date = datetime.strptime("2024-03-14", "%Y-%m-%d")
    val_end_date = datetime.strptime("2024-08-07", "%Y-%m-%d")

    # 定义数据集路径
    news_path = "/home/users/liuyu/Framework/dataset/csi300_original/news/"
    price_path = "/home/users/liuyu/Framework/dataset/csi300_original/price/"
    output_path = "/home/users/liuyu/Framework/dataset/csi300"

    # 创建输出文件夹
    for dataset in ['train', 'val', 'test']:
        for data_type in ['news', 'price']:
            os.makedirs(os.path.join(output_path, dataset, data_type), exist_ok=True)

    # 处理新闻数据
    for ticker_name in os.listdir(news_path):
        for date_csv in os.listdir(os.path.join(news_path, ticker_name)):
            if date_csv.endswith(".csv"):
                date_str = date_csv.split(".")[0]
                date = datetime.strptime(date_str, "%Y-%m-%d")
                source_path = os.path.join(news_path, ticker_name, date_csv)
                if date <= train_end_date:
                    destination_path = os.path.join(output_path, 'train', 'news', ticker_name, date_csv)
                elif date <= val_end_date:
                    destination_path = os.path.join(output_path, 'val', 'news', ticker_name, date_csv)
                else:
                    destination_path = os.path.join(output_path, 'test', 'news', ticker_name, date_csv)
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.move(source_path, destination_path)

    # 处理价格数据
    for ticker_csv in os.listdir(price_path):
        source_path = os.path.join(price_path, ticker_csv)
        df = pd.read_csv(source_path)
        df['Date'] = pd.to_datetime(df['Date'])

        train_df = df[df['Date'] <= train_end_date]
        val_df = df[(df['Date'] > train_end_date) & (df['Date'] <= val_end_date)]
        test_df = df[df['Date'] > val_end_date]

        for dataset, dataset_df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
            if not dataset_df.empty:
                new_csv = os.path.join(output_path, dataset, 'price', ticker_csv)
                dataset_df.to_csv(new_csv, index=False)


#制作label
def create_label():
    for ticker_csv in os.listdir("/home/users/liuyu/Framework/dataset/csi300/train/price/"):
        source_path = os.path.join("/home/users/liuyu/Framework/dataset/train/price/", ticker_csv)
        data = pd.read_csv(source_path)
        data['Label'] = (data['Close'].shift(-1) - data['Close'] > 0).astype(int)
        data.to_csv(source_path, index=False)


    for ticker_csv in os.listdir("/home/users/liuyu/Framework/dataset/csi300/val/price/"):
        source_path = os.path.join("/home/users/liuyu/Framework/dataset/csi300/val/price/", ticker_csv)
        data = pd.read_csv(source_path)
        data['Label'] = (data['Close'].shift(-1) - data['Close'] > 0).astype(int)
        data.to_csv(source_path, index=False)

    for ticker_csv in os.listdir("/home/users/liuyu/Framework/dataset/csi300/test/price/"):
        source_path = os.path.join("/home/users/liuyu/Framework/dataset/csi300/test/price/", ticker_csv)
        data = pd.read_csv(source_path)
        data['Label'] = (data['Close'].shift(-1) - data['Close'] > 0).astype(int)
        data.to_csv(source_path, index=False)

#计算查看数据集中的None值和空值数量
def count_null_and_na_values():
    count = 0
    null_columns = {}

    for ticker_csv in os.listdir("/home/users/liuyu/Framework/dataset/train/price/"):
        source_path = os.path.join("/home/users/liuyu/Framework/dataset/train/price/", ticker_csv)
        data = pd.read_csv(source_path)

        # 计算空值数量
        null_count = data.isnull().sum().sum()
        na_count = data.isna().sum().sum()
        count += null_count + na_count

        # 记录空值所在的列
        if null_count > 0 or na_count > 0:
            # 只保留空值数量大于零的列
            null_columns[ticker_csv] = {
                'null_counts': {col: count for col, count in data.isnull().sum().to_dict().items() if count > 0},
                'na_counts': {col: count for col, count in data.isna().sum().to_dict().items() if count > 0}
            }

            print(f"Total null and na values in {ticker_csv}: {null_count + na_count}")
            print(f"Null values in columns: {null_columns[ticker_csv]['null_counts']}")
            print(f"Na values in columns: {null_columns[ticker_csv]['na_counts']}")

    print(f"Total null and na values across all files: {count}")
    print(f"Null and na values in each file: {null_columns}")


split_data()
create_label()



