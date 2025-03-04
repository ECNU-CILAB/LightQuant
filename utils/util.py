# author:Liu Yu
# time:2025/2/11 15:13
import os
import shutil
from datetime import datetime
import pandas as pd
import numpy as np

def split_data():
    #划分训练集,验证集,测试集
    # 定义日期范围
    train_end_date = datetime.strptime("2024-03-14", "%Y-%m-%d")
    val_end_date = datetime.strptime("2024-08-07", "%Y-%m-%d")

    # 定义数据集路径
    news_path = "/home/users/liuyu/Framework/dataset/csi50_origional/news/"
    price_path = "/home/users/liuyu/Framework/dataset/csi50_origional/price/"
    news_embedding_path = "/home/users/liuyu/Framework/dataset/csi50_origional/news_embedding/"
    output_path = "/home/users/liuyu/Framework/dataset/csi50"

    # 创建输出文件夹
    for dataset in ['train', 'val', 'test']:
        for data_type in ['news', 'price', 'news_embedding']:
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

    #处理新闻嵌入数据
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
    for ticker_csv in os.listdir("/home/users/liuyu/Framework/dataset/csi50/train/price/"):
        source_path = os.path.join("/home/users/liuyu/Framework/dataset/csi50/train/price/", ticker_csv)
        data = pd.read_csv(source_path)
        data['Label'] = (data['Close'].shift(-1) - data['Close'] > 0).astype(int)
        data.to_csv(source_path, index=False)


    for ticker_csv in os.listdir("/home/users/liuyu/Framework/dataset/csi50/val/price/"):
        source_path = os.path.join("/home/users/liuyu/Framework/dataset/csi50/val/price/", ticker_csv)
        data = pd.read_csv(source_path)
        data['Label'] = (data['Close'].shift(-1) - data['Close'] > 0).astype(int)
        data.to_csv(source_path, index=False)

    for ticker_csv in os.listdir("/home/users/liuyu/Framework/dataset/csi50/test/price/"):
        source_path = os.path.join("/home/users/liuyu/Framework/dataset/csi50/test/price/", ticker_csv)
        data = pd.read_csv(source_path)
        data['Label'] = (data['Close'].shift(-1) - data['Close'] > 0).astype(int)
        data.to_csv(source_path, index=False)

#计算查看数据集中的None值和空值数量
def count_null_and_na_values():
    count = 0
    null_columns = {}

    for ticker_csv in os.listdir("/home/users/liuyu/Framework/dataset/csi50_origional/price/"):
        source_path = os.path.join("/home/users/liuyu/Framework/dataset/csi50_origional/price/", ticker_csv)
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

def generate_trading_date_list():
    price_path = "/home/users/liuyu/Framework/dataset/csi50_origional/price/中国石化.csv"
    df = pd.read_csv(price_path)

    ticker_csv_list = df['Date'].tolist()

    df = pd.DataFrame(ticker_csv_list, columns=['Date'])
    df.to_csv("/home/users/liuyu/Framework/dataset/trading_date_list.csv", index=False)
    print(len(ticker_csv_list))


def check_embedding():
    path = "/home/users/liuyu/Framework/dataset/csi50_origional/news_embedding/中国建筑/2024-04-29.npy"
    embedding = np.load(path)
    print(embedding)
    print(embedding.shape)


#计算数据集中None值和空值数量
# count_null_and_na_values()
#划分训练集,验证集,测试集
# split_data()
#创建label
# create_label()
# 生成交易日期列表
# generate_trading_date_list()
#查看embedding
check_embedding()



