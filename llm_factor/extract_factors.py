import os
import json
import re
import csv
from datetime import datetime
import pandas as pd
import argparse

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_stock_names(file_path):
    """从CSV文件加载股票代码和名称的映射"""
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    return dict(zip(df['code'], df['code_name']))



def load_news_from_json(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        news_data = json.load(file)
    return news_data


def group_news_by_date(news_data):

    grouped_news = {}
    for news in news_data:
        date = news['time'].split(' ')[0]
        if date not in grouped_news:
            grouped_news[date] = []
        grouped_news[date].append(news)
    return grouped_news


def construct_daily_prompt(daily_news, stock_name, max_news_length=5000):

    daily_news_str = '\n'.join([f"{news['time'][:16]}: {news['title'][:100]} - {news['content'][:max_news_length]}" for news in daily_news])

    prompt = [
        {
            "role": "system",
            "content": (
                "作为一名股票交易新闻分析师，您是一名乐于助人的精准助手。"
                "你的任务是请从以下新闻中提取可能影响{}股价的前3个因素，并按照以下格式分点回答：\n"
                "1. 第一个因素\n"
                "2. 第二个因素\n"
                "3. 第三个因素\n"
            ).format(stock_name),
        },
        {
            "role": "user",
            "content": (
                f"\n以下是今天的新闻：\n{daily_news_str}"
            ),
        },
    ]
    return prompt


def parse_response(response_text, date, top_n=3):
    """Parse the model response and extract the top top_n factors that influence stock prices"""
    # Use regular expressions to extract the factors labeled as 1., 2., 3.
    factors = []
    pattern = r'\d+\.\s*(.*?)(?=\n\d+\.|$)'
    matches = re.findall(pattern, response_text, re.DOTALL)

    while len(matches) < top_n:
        matches.append("--")

    for i in range(top_n):
        if matches[i].strip() == "":
            matches[i] = "--"

    # Return the date and the top top_n factors.
    return {
        '日期': date,
        'top1': matches[0],
        'top2': matches[1],
        'top3': matches[2]
    }


def append_to_csv(results, output_file='daily_analysis.csv'):

    fieldnames = ['日期', 'top1', 'top2', 'top3']


    file_exists = False
    try:
        with open(output_file, 'r', newline='', encoding='utf-8-sig') as file:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(output_file, mode='a', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in results:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='your model path', help='model path')
    parser.add_argument('--tokenizer_path', type=str, default='your model path', help='tokenizer path')
    parser.add_argument('--gpu_num', type=int, default=4, help='gpu num')
    parser.add_argument('--dataset', type=str, default='CSMD50', help='dataset name')
    parser.add_argument('--original_news_path', type=str, default='', help='original news path')
    parser.add_argument('--output_news_path', type=str, default='', help='output file path')
    args = parser.parse_args()

    llm = LLM(model=args.model_path, dtype='float16', trust_remote_code=True, tensor_parallel_size=args.gpu_num)
    tokenizer_qwen = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)


    stock_names = load_stock_names(f'./{args.datset}.csv')


    news_folder = args.original_news_path
    output_folder = args.output_news_path

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(news_folder):
        if filename.endswith('.json'):

            stock_code, _ = os.path.splitext(filename)
            print(stock_code)


            stock_name = stock_names.get(stock_code, "未知股票")
            print(stock_name)


            output_file = os.path.join(output_folder, f'{stock_code}_factors.csv')


            json_file_path = os.path.join(news_folder, filename)
            news_data = load_news_from_json(json_file_path)
            grouped_news = group_news_by_date(news_data)

            sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=1000)

            all_results = []


            for date, daily_news in grouped_news.items():

                prompt = construct_daily_prompt(daily_news, stock_name)
                

                texts = tokenizer_qwen.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )
                outputs = llm.generate(texts, sampling_params)

                generated_text = outputs[0].outputs[0].text

                # Parse the model response and extract factors that influence stock prices.
                result = parse_response(generated_text, date)

                all_results.append(result)

                # print(f"Date: {date}\nGenerated Text:\n{generated_text}\n{'-' * 80}")

            append_to_csv(all_results, output_file=output_file)

            print(f"result saved to {output_file}")