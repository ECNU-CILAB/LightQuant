# author:Liu Yu
# time:2025/2/11 18:31
import os
import sys
# 获取当前文件的父目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将父目录添加到 sys.path
sys.path.append(parent_dir)

import torch.nn as nn
import torch
from model.LSTM import lstm
from model.ALSTM import ALSTM
from model.BiLSTM import BiLSTM
from my_parser import args
from utils.price_dataloader import *
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm
import sklearn
from utils.plot import *
import random
import numpy as np


def train():

    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}" )
    else:
        device = torch.device("cpu")

    train_dataset, val_dataset, _ = create_dataset(train_folder=args.train_folder, val_folder=args.val_folder, test_folder=None, look_back=args.look_back)

    train_dataloader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dataloader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)


    if args.model == "lstm":
        model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2, dropout=args.dropout, batch_first=args.batch_first )

    if args.model == "alstm":
        model = ALSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2, dropout=args.dropout, batch_first=args.batch_first, attention_size=args.attention_size )

    if args.model == "bi_lstm":
        model = BiLSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2, dropout=args.dropout, batch_first=args.batch_first )

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)  # 余弦退火学习率

    # 早停
    early_stop_patience = 10
    early_stop_counter = 0
    best_val_loss = float('inf')

    avg_train_loss_list = []
    avg_val_loss_list = []
    avg_acc_list = []
    avg_mcc_list = []
    plot_epoch = 0

    for epoch in tqdm(range(args.epochs)):
        total_loss = 0

        for idx, (input_data, label) in enumerate(train_dataloader):

            input_data = input_data.to(device)
            label = label.squeeze()
            label = torch.where(label == -1, torch.tensor(0), label).to(torch.int).long()
            label = label.to(device)
            prediction = model(input_data)
            prediction = prediction.squeeze()
            prediction = prediction.to(device)


            loss = criterion(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # 梯度裁剪
            optimizer.step()

            total_loss += loss.item()

        # 计算平均损失
        avg_loss = total_loss / len(train_dataloader)
        avg_train_loss_list.append(avg_loss)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss}")

        scheduler.step()  # 更新学习率

        # 5个 epoch 后验证模型
        if (epoch + 1) % 5 == 0:
            # 验证
            val_loss, val_acc, val_mcc, val_f1_score = validate(model, val_dataloader, device, criterion)
            # 恢复训练模式
            model.train()

            avg_val_loss_list.append(val_loss)
            avg_acc_list.append(val_acc)
            avg_mcc_list.append(val_mcc)
            print(f"Epoch {epoch + 1}/{args.epochs}, Val Loss: {val_loss}, Val Acc: {val_acc}, Val Mcc: {val_mcc}, Val F1_score: {val_f1_score}")

            # 早停判断
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0  # 重置计数器
                torch.save(model.state_dict(), f"{args.model_save_folder}{args.model}.pth")  # 保存最佳模型
            else:
                early_stop_counter += 1

            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                plot_epoch = epoch + 1
                break
        plot_epoch = epoch + 1
    print(f'训练完成')
    plot_figure(avg_train_loss_list, avg_val_loss_list, avg_acc_list, avg_mcc_list, plot_epoch, args.figure_save_folder, args.model)
    print("绘图完成")

def validate(model, val_dataloader, device, criterion):
    """ 验证集评估 """
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for input_data, label in val_dataloader:
            input_data, label = input_data.to(device), label.to(device).long().squeeze()
            prediction = model(input_data)  # 获取 logits
            loss = criterion(prediction, label)
            total_loss += loss.item()

            _, predicted_classes = torch.max(prediction, 1)
            correct += (predicted_classes == label).sum().item()
            total_samples += label.size(0)

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted_classes.cpu().numpy())

    val_loss = total_loss / len(val_dataloader)
    val_acc = correct / total_samples
    val_mcc = matthews_corrcoef(all_labels, all_predictions)
    val_f1_score = f1_score(all_labels, all_predictions)
    return val_loss, val_acc, val_mcc, val_f1_score
