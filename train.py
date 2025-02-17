# author:Liu Yu
# time:2025/2/11 18:31

import torch.nn as nn
import torch
from model.LSTM import lstm
from my_parser import args
from utils.dataloader import *
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
import sklearn
from utils.plot import *
from model.mlp import SimpleMLPModel


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

    if args.model == "mlp":
        model = SimpleMLPModel(input_size=args.input_size)

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降

    best_val_acc = 0.0  # 记录最佳 ACC

    avg_train_loss_list = []
    avg_val_loss_list = []
    avg_acc_list = []
    avg_mcc_list = []

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
            optimizer.step()

            total_loss += loss.item()

        # 计算平均损失
        avg_loss = total_loss / len(train_dataloader)
        avg_train_loss_list.append(avg_loss)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss}")


        # 10个 epoch 后验证模型
        if (epoch + 1) % 10 == 0:
            # 验证
            val_loss, val_acc, val_mcc = validate(model, val_dataloader, device, criterion)
            # 恢复训练模式
            model.train()

            avg_val_loss_list.append(val_loss)
            avg_acc_list.append(val_acc)
            avg_mcc_list.append(val_mcc)
            print(f"Epoch {epoch + 1}/{args.epochs}, Val Loss: {val_loss}, Val Acc: {val_acc}, Val Mcc: {val_mcc}")

            # 保存最佳 MCC 模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save( model.state_dict(), f"{args.model_save_folder}{args.model}.pth")
    print(f'训练完成')
    plot_figure(avg_train_loss_list, avg_val_loss_list, avg_acc_list, avg_mcc_list, args.epochs)
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
    return val_loss, val_acc, val_mcc
