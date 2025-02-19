# author:Liu Yu
# time:2025/2/11 18:33
import argparse
import torch

parser = argparse.ArgumentParser()


# TODO 常改动参数
parser.add_argument('--train_folder', default='/home/users/liuyu/Framework/dataset/csi300/train/price', type=str)#/home/users/liuyu/experiment/ACL18_D/fill0/train/  /home/users/liuyu/Framework/dataset/train/price
parser.add_argument('--val_folder', default='/home/users/liuyu/Framework/dataset/csi300/val/price', type=str)
parser.add_argument('--test_folder', default='/home/users/liuyu/Framework/dataset/csi300/test/price', type=str)#/home/users/liuyu/experiment/ACL18_D/fill0/test/
parser.add_argument('--epochs', default=100, type=int) # 训练轮数
parser.add_argument('--model', default="lstm", type=str) # 模型名称 lstm mlp
parser.add_argument('--layers', default=2, type=int) # 层数
parser.add_argument('--input_size', default=5, type=int) #输入特征的维度
parser.add_argument('--hidden_size', default=32, type=int) #隐藏层的维度
parser.add_argument('--lr', default=0.0001, type=float) #learning rate 学习率
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--look_back', default=5, type=int) # lookback时间长度
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--useGPU', default=True, type=bool) #是否使用GPU
parser.add_argument('--GPU_ID', default=1, type=int)
parser.add_argument('--batch_first', default=True, type=bool) #是否将batch_size放在第一维
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--model_save_folder', default='./model_saved/') # 模型保存位置
parser.add_argument('--figure_save_folder', default='./figure/')
parser.add_argument('--test_result_save_folder', default='./test_result/')


args = parser.parse_args()
