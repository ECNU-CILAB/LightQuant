import os
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
import argparse

# Calculate the Number of None and Empty Values in the Dataset
def count_null_and_na_values(dataset):
    count = 0
    null_columns = {}

    for ticker_csv in os.listdir(f"../dataset/{dataset}/price/"):
        source_path = os.path.join(f"../dataset/{dataset}/price/", ticker_csv)
        data = pd.read_csv(source_path)


        null_count = data.isnull().sum().sum()
        na_count = data.isna().sum().sum()
        count += null_count + na_count


        if null_count > 0 or na_count > 0:

            null_columns[ticker_csv] = {
                'null_counts': {col: count for col, count in data.isnull().sum().to_dict().items() if count > 0},
                'na_counts': {col: count for col, count in data.isna().sum().to_dict().items() if count > 0}
            }

            print(f"Total null and na values in {ticker_csv}: {null_count + na_count}")
            print(f"Null values in columns: {null_columns[ticker_csv]['null_counts']}")
            print(f"Na values in columns: {null_columns[ticker_csv]['na_counts']}")

    print(f"Total null and na values across all files: {count}")
    print(f"Null and na values in each file: {null_columns}")

# Fill Missing Values Using Linear Interpolation
def linear_interpolation(dataset):
    data_dir = f"../dataset/{dataset}/price/"
    for ticker_csv in os.listdir(data_dir):
        source_path = os.path.join(data_dir, ticker_csv)

        data = pd.read_csv(source_path)


        if data.isnull().values.any():

            data.interpolate(method='linear', inplace=True)


            data.fillna(method='ffill', inplace=True)
            data.fillna(method='bfill', inplace=True)
            print(f"finished: {ticker_csv}")


        data.to_csv(source_path, index=False)

# Generate Trading Date List
def generate_trading_date_list(dataset):
    price_path = f"../dataset/{dataset}/price/中国石化.csv"
    df = pd.read_csv(price_path)

    ticker_csv_list = df['Date'].tolist()

    df = pd.DataFrame(ticker_csv_list, columns=['Date'])
    df.to_csv("../dataset/trading_date_list.csv", index=False)
    print(len(ticker_csv_list))

# split dataset
def split_data(dataset):

    train_end_date = datetime.strptime("2024-03-14", "%Y-%m-%d")
    val_end_date = datetime.strptime("2024-08-07", "%Y-%m-%d")


    news_path = f"../dataset/{dataset}/news/"
    price_path = f"../dataset/{dataset}/price/"
    news_embedding_path = f"../dataset/{dataset}/news_embedding/"
    output_path = f"../dataset/{dataset}"


    for dataset in ['train', 'val', 'test']:
        for data_type in ['news', 'price', 'news_embedding']:
            os.makedirs(os.path.join(output_path, dataset, data_type), exist_ok=True)

    # split news  data
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
                shutil.copy(source_path, destination_path)

    # split news embedding data
    for ticker_name in os.listdir(news_embedding_path):
        for date_csv in os.listdir(os.path.join(news_embedding_path, ticker_name)):
            if date_csv.endswith(".npy"):
                date_str = date_csv.split(".")[0]
                date = datetime.strptime(date_str, "%Y-%m-%d")
                source_path = os.path.join(news_embedding_path, ticker_name, date_csv)
                if date <= train_end_date:
                    destination_path = os.path.join(output_path, 'train', 'news_embedding', ticker_name, date_csv)
                elif date <= val_end_date:
                    destination_path = os.path.join(output_path, 'val', 'news_embedding', ticker_name, date_csv)
                else:
                    destination_path = os.path.join(output_path, 'test', 'news_embedding', ticker_name, date_csv)
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.copy(source_path, destination_path)

    # split price data
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

# Create Label
def create_label(dataset):
    for ticker_csv in os.listdir(f"../dataset/{dataset}/train/price/"):
        source_path = os.path.join(f"../dataset/{dataset}/train/price/", ticker_csv)
        data = pd.read_csv(source_path)
        data['Label'] = (data['Close'].shift(-1) - data['Close'] > 0).astype(int)
        data.to_csv(source_path, index=False)


    for ticker_csv in os.listdir(f"../dataset/{dataset}/val/price/"):
        source_path = os.path.join(f"../dataset/{dataset}/val/price/", ticker_csv)
        data = pd.read_csv(source_path)
        data['Label'] = (data['Close'].shift(-1) - data['Close'] > 0).astype(int)
        data.to_csv(source_path, index=False)

    for ticker_csv in os.listdir(f"../dataset/{dataset}/test/price/"):
        source_path = os.path.join(f"../dataset/{dataset}/test/price/", ticker_csv)
        data = pd.read_csv(source_path)
        data['Label'] = (data['Close'].shift(-1) - data['Close'] > 0).astype(int)
        data.to_csv(source_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CSMD50")
    args = parser.parse_args()

    # Calculate the Number of None and Empty Values in the Dataset
    count_null_and_na_values(dataset = args.dataset)

    # Fill Missing Values Using Linear Interpolation
    linear_interpolation(dataset = args.dataset)

    # Calculate again and check
    count_null_and_na_values(dataset = args.dataset)

    # generate trading date list
    generate_trading_date_list(dataset = args.dataset)

    # split dataset
    split_data(dataset = args.dataset)

    # create label
    create_label(dataset = args.dataset)




