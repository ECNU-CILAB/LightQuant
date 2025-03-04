# author:Liu Yu
# time:2025/2/11 18:33
import argparse
import torch

parser = argparse.ArgumentParser()


# TODO 常改动参数
parser.add_argument('--dataset', default="csi300", type=str) #csi50 csi300
parser.add_argument('--experiment', default='price_news_experiment', type=str)
# 解析命令行参数
args = parser.parse_args()
parser.add_argument('--train_folder', default=f'/home/users/liuyu/Framework/dataset/{args.dataset}/train/price', type=str)
parser.add_argument('--val_folder', default=f'/home/users/liuyu/Framework/dataset/{args.dataset}/val/price', type=str)
parser.add_argument('--test_folder', default=f'/home/users/liuyu/Framework/dataset/{args.dataset}/test/price', type=str)
parser.add_argument('--epochs', default=9, type=int) # 训练轮数
parser.add_argument('--model', default="lstm", type=str) # 模型名称 lstm alstm adv_lstm bi_lstm
parser.add_argument('--epsilon', default=0.1, type=float)
parser.add_argument('--perturbation_size', default=0.1, type=float)
parser.add_argument('--layers', default=2, type=int) # 层数
parser.add_argument('--input_size', default=5, type=int) #输入特征的维度
parser.add_argument('--hidden_size', default=128, type=int) #隐藏层的维度
parser.add_argument('--attention_size', default=128, type=int) #注意力层的维度
parser.add_argument('--lr', default=2e-6, type=float) #learning rate 学习率
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--look_back', default=5, type=int) # lookback时间长度
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--useGPU', default=True, type=bool) #是否使用GPU
parser.add_argument('--GPU_ID', default=1, type=int)
parser.add_argument('--batch_first', default=True, type=bool) #是否将batch_size放在第一维
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--model_save_folder', default=f'/home/users/liuyu/Framework/{args.experiment}/result/{args.dataset}/model_saved/') # 模型保存位置
parser.add_argument('--figure_save_folder', default=f'/home/users/liuyu/Framework/{args.experiment}//result/{args.dataset}/figure/')
parser.add_argument('--test_result_save_folder', default=f'/home/users/liuyu/Framework/{args.experiment}//result/{args.dataset}/test_result/')


args = parser.parse_args()
