import os
import sys
import torch.nn as nn
import torch
from model.LSTM import LSTM
from model.ALSTM import ALSTM
from model.BiLSTM import BiLSTM
from model.Adv_LSTM import AdvLSTM
from model.DTML import DTML
from model.SCINet import SCINet
from utils.price_dataloader import *
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm
import sklearn
from utils.plot import *
import random
import numpy as np
import swanlab

def train(args):
    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}" )
    else:
        device = torch.device("cpu")

    if args.model == "dtml":
        train_dataset = DTML_Dataset(args.train_price_folder, args.look_back_window, args.n_stocks)
        val_dataset = DTML_Dataset(args.val_price_folder, args.look_back_window, args.n_stocks)

    elif args.model == "scinet":
        train_dataset = SCINet_Dataset(args.train_price_folder, args.seq_len, args.pred_len)
        val_dataset = SCINet_Dataset(args.val_price_folder, args.seq_len, args.pred_len)

    else:
        train_dataset, val_dataset, _ = create_dataset(train_folder=args.train_price_folder, val_folder=args.val_price_folder, test_folder=None, look_back_window=args.look_back_window)

    train_dataloader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dataloader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)


    if args.model == "lstm":
        model = LSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first )

    elif args.model == "alstm":
        model = ALSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first, attention_size=args.attention_size )

    elif args.model == "bi_lstm":
        model = BiLSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first )

    elif args.model == "adv_lstm":
        model = AdvLSTM(input_size=args.input_size, hidden_size=args.hidden_size, output_size=1, attention_size=args.attention_size, perturbation_size=args.perturbation_size, epsilon=args.epsilon)

    elif args.model == "dtml":
        model = DTML(input_size = 7, hidden_size = 64, num_layers = 2, n_heads = 4)

    elif args.model == "scinet":
        model = SCINet(input_len=args.seq_len, pred_len=args.pred_len, input_dim=8, hidden_dim=args.hidden_size,
                       SCINet_Layers=args.SCINet_Layers).to(device)

    else:
        raise ValueError(f"Invalid model name: {args.model}")

    model.to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)


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
            if args.model == "adv_lstm":
                input_data.requires_grad_()

            label = label.squeeze()
            label = torch.where(label == -1, torch.tensor(0), label).to(torch.int).float()
            label = label.to(device)

            prediction = model(input_data)
            prediction = prediction.squeeze()
            prediction = prediction.to(device)


            loss = criterion(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss.item()


        avg_loss = total_loss / len(train_dataloader)
        avg_train_loss_list.append(avg_loss)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss}")

        swanlab.log({"avg_loss": avg_loss})

        scheduler.step()


        if (epoch + 1) % 5 == 0:

            val_loss, val_acc, val_mcc, val_f1_score = validate(model, val_dataloader, device, criterion)

            model.train()

            avg_val_loss_list.append(val_loss)
            avg_acc_list.append(val_acc)
            avg_mcc_list.append(val_mcc)
            print(f"Epoch {epoch + 1}/{args.epochs}, Val Loss: {val_loss}, Val Acc: {val_acc}, Val Mcc: {val_mcc}, Val F1_score: {val_f1_score}")

            swanlab.log({"Val_loss": val_loss, "Val_acc": val_acc, "Val_mcc": val_mcc, "Val_f1": val_f1_score})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), f"{args.model_save_folder}{args.model}.pth")
            else:
                early_stop_counter += 1

            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                plot_epoch = epoch + 1
                break
        plot_epoch = epoch + 1
    print(f'finished training')
    plot_figure(avg_train_loss_list, avg_val_loss_list, avg_acc_list, avg_mcc_list, plot_epoch, args.figure_save_folder, args.model)
    print("finished plotting")

def validate(model, val_dataloader, device, criterion):

    model.eval()
    total_loss, correct, total_samples = 0, 0, 0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for input_data, label in val_dataloader:

            input_data, label = input_data.to(device), label.to(device).float().squeeze()
            prediction = model(input_data)
            prediction = prediction.squeeze()

            loss = criterion(prediction, label)
            total_loss += loss.item()

            predicted_classes = (prediction > 0.5).float()
            correct += (predicted_classes == label).sum().item()

            total_samples += label.size(0) * label.size(1)  if label.dim() > 1 else label.size(0)

            all_labels.extend(label.cpu().numpy().flatten())
            all_predictions.extend(predicted_classes.cpu().numpy().flatten())

    val_loss = total_loss / len(val_dataloader)
    val_acc = correct / total_samples
    val_mcc = matthews_corrcoef(all_labels, all_predictions)
    val_f1_score = f1_score(all_labels, all_predictions)
    return val_loss, val_acc, val_mcc, val_f1_score

