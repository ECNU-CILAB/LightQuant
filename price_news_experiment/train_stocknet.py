# author:Liu Yu
# time:2025/2/11 18:31
import os
import sys
import torch
# 获取当前文件的父目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将父目录添加到 sys.path
sys.path.append(parent_dir)

import torch.nn as nn
from model.LSTM import lstm
from model.ALSTM import ALSTM
from model.BiLSTM import BiLSTM
from model.BiGRU import BiGRU
from model.StockNet import StockNet
from utils.price_news_dataloader import StockDatasetFromPickle
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm
from utils.plot import *
from my_parser import args
from utils.price_news_dataloader import *

def train():

    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}" )
    else:
        device = torch.device("cpu")

    # train_dataset = StockDataset(args.train_price_folder, args.train_news_folder, look_back_window=args.look_back_window)
    # val_dataset = StockDataset(args.val_price_folder, args.val_news_folder, look_back_window=args.look_back_window)

    train_dataset = StockDatasetFromPickle("/home/users/liuyu/Framework/dataset/csi300/train/dataset.pkl")
    val_dataset = StockDatasetFromPickle("/home/users/liuyu/Framework/dataset/csi300/val/dataset.pkl")

    train_dataloader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.model == "lstm":
        model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2, dropout=args.dropout, batch_first=args.batch_first)

    if args.model == "alstm":
        model = ALSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2, dropout=args.dropout, batch_first=args.batch_first, attention_size=args.attention_size )

    if args.model == "bi_lstm":
        model = BiLSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2, dropout=args.dropout, batch_first=args.batch_first)

    if args.model == "stocknet":
        model = StockNet(5, 20, 128, 64, 10)

    model.to(device)


    criterion = nn.CrossEntropyLoss().to(device)  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)  # 余弦退火学习率

    # 早停
    early_stop_patience = 15
    early_stop_counter = 0
    best_val_loss = float('inf')

    avg_train_loss_list = []
    avg_val_loss_list = []
    avg_acc_list = []
    avg_mcc_list = []
    plot_epoch = 0

    # 初始化历史预测
    history_outputs = torch.zeros((args.batch_size, args.look_back_window, 128)).to(device)
    for epoch in tqdm(range(args.epochs)):
        total_loss = 0

        for idx, batch in enumerate(train_dataloader):
            market, news, label = batch["market"].to(device), batch["news"].to(device), batch["label"].to(device)
            label = label.long()
            if label.shape[0] != args.batch_size:
                # 将label用0填充，扩充到batch_size
                label = torch.cat((label, torch.zeros(args.batch_size - label.shape[0], dtype=torch.long).to(device)), dim=0)

            prediction, mu, logvar, prediction_mapped = model(market, news, history_outputs)
            # print(prediction.shape)
            # 更新历史预测
            history_outputs = torch.cat((history_outputs[:, 1:, :], prediction_mapped.detach().unsqueeze(1)), dim=1)
            loss_ce = criterion(prediction, label)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss_ce + 0.01 * kl_div  # 变分自编码器损失
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
            val_loss, val_acc, val_mcc = validate(model, val_dataloader, device, criterion, history_outputs)
            # 恢复训练模式
            model.train()

            avg_val_loss_list.append(val_loss)
            avg_acc_list.append(val_acc)
            avg_mcc_list.append(val_mcc)
            print(f"Epoch {epoch + 1}/{args.epochs}, Val Loss: {val_loss}, Val Acc: {val_acc}, Val Mcc: {val_mcc}")

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
    return history_outputs

def validate(model, val_dataloader, device, criterion, history_outputs):
    """ 验证集评估 """
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for batch in val_dataloader:
            market, news, label = batch["market"].to(device), batch["news"].to(device), batch["label"].to(device)
            label = label.long()
            if label.shape[0] != args.batch_size:
                # 将label用0填充，扩充到batch_size
                label = torch.cat((label, torch.zeros(args.batch_size - label.shape[0], dtype=torch.long).to(device)),
                                  dim=0)

            prediction, mu, logvar, prediction_mapped = model(market, news, history_outputs)

            # 更新历史预测（注意 `.detach()` 避免计算图回溯）
            history_outputs = torch.cat((history_outputs[:, 1:, :], prediction_mapped.detach().unsqueeze(1)), dim=1)

            loss_ce = criterion(prediction, label)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss_ce + 0.01 * kl_div
            total_loss += loss.item()

            _, predicted_classes = torch.max(prediction, 1)
            correct += (predicted_classes == label).sum().item()
            total_samples += label.size(0)

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted_classes.cpu().numpy())

    val_loss = total_loss / len(val_dataloader)
    val_acc = correct / total_samples
    val_mcc = matthews_corrcoef(all_labels, all_predictions)
    val_f1 = f1_score(all_labels, all_predictions)
    return val_loss, val_acc, val_mcc, val_f1
