# author:Liu Yu
# time:2025/2/11 15:13

import os
import shutil
from datetime import datetime
import pandas as pd
#划分训练集和测试集
# for ticker_name in os.listdir("/home/users/liuyu/Framework/dataset/CSI300/news_factor_preprocessed/"):
#     new_file_name = "/home/users/liuyu/Framework/dataset/data/news/" + ticker_name
#     if not os.path.exists(new_file_name):
#         os.mkdir(new_file_name)
#     for date_csv in os.listdir("/home/users/liuyu/Framework/dataset/CSI300/news_factor_preprocessed/" + ticker_name):
#         if(date_csv.endswith(".csv")):
#             date_str = date_csv.split(".")[0]
#             date = datetime.strptime(date_str, "%Y-%m-%d")
#             if date >= datetime.strptime("2024-3-14", "%Y-%m-%d"):
#                 source_path = os.path.join("/home/users/liuyu/Framework/dataset/CSI300/news_factor_preprocessed/",
#                                            ticker_name, date_csv)
#                 destination_path = os.path.join(new_file_name, date_csv)
#                 shutil.move(source_path, destination_path)
#
# for ticker_csv in os.listdir("/home/users/liuyu/Framework/dataset/CSI300/price/"):
#     source_path = os.path.join("/home/users/liuyu/Framework/dataset/CSI300/price/", ticker_csv)
#     new_csv = os.path.join("/home/users/liuyu/Framework/dataset/data/price/", ticker_csv)
#
#     if not os.path.exists(new_csv):
#         # 读取原始CSV文件
#         df = pd.read_csv(source_path)
#
#         # 假设日期列名为'Date'
#         df['Date'] = pd.to_datetime(df['Date'])
#
#         # 筛选出日期大于2024-3-14的行
#         filtered_df = df[df['Date'] >= pd.to_datetime("2024-03-14")]
#
#         # 将筛选后的数据写入新的CSV文件
#         filtered_df.to_csv(new_csv, index=False)

#制作label
# for ticker_csv in os.listdir("/home/users/liuyu/Framework/dataset/train/price/"):
#     source_path = os.path.join("/home/users/liuyu/Framework/dataset/train/price/", ticker_csv)
#     data = pd.read_csv(source_path)
#     data['Label'] = (data['Close'].shift(-1) - data['Close'] > 0).astype(int)
#     data.to_csv(source_path, index=False)
#
#
# for ticker_csv in os.listdir("/home/users/liuyu/Framework/dataset/val/price/"):
#     source_path = os.path.join("/home/users/liuyu/Framework/dataset/val/price/", ticker_csv)
#     data = pd.read_csv(source_path)
#     data['Label'] = (data['Close'].shift(-1) - data['Close'] > 0).astype(int)
#     data.to_csv(source_path, index=False)

#计算数据集中的None值和空值数量
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






