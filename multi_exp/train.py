import os
import sys
import torch
import torch.nn as nn
from model.StockNet import StockNet
from model.HAN import HAN
from model.PEN import PEN
from utils.price_news_dataloader import *
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm
from utils.plot import *
from utils.price_news_dataloader import *
import swanlab


def train_StockNet(args):

    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}" )
    else:
        device = torch.device("cpu")

    if args.model == "StockNet":
        train_dataset = StockNet_Dataset(args.train_price_folder, args.train_news_folder, look_back_window=args.look_back_window)
        val_dataset = StockNet_Dataset(args.val_price_folder, args.val_news_folder, look_back_window=args.look_back_window)

        # if .pkl file exists, use it
        # train_dataset = StockDatasetFromPickle(f"../dataset/{args.dataset}/train/dataset.pkl")
        # val_dataset = StockDatasetFromPickle(f"../dataset/{args.dataset}/val/dataset.pkl")

        train_dataloader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_dataloader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        model = StockNet(5, 20, 128, 64, args.look_back_window)

    else:
        raise ValueError("Invalid model name")

    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    early_stop_patience = 15
    early_stop_counter = 0
    best_val_loss = float('inf')

    avg_train_loss_list = []
    avg_val_loss_list = []
    avg_acc_list = []
    avg_mcc_list = []
    plot_epoch = 0

    # initialized
    history_outputs = torch.zeros((args.batch_size, args.look_back_window, 128)).to(device)
    for epoch in tqdm(range(args.epochs)):
        total_loss = 0

        for idx, batch in enumerate(train_dataloader):
            market, news, label = batch["market"].to(device), batch["news"].to(device), batch["label"].to(device)
            label = label.long()
            if label.shape[0] != args.batch_size:

                label = torch.cat((label, torch.zeros(args.batch_size - label.shape[0], dtype=torch.long).to(device)), dim=0)

            prediction, mu, logvar, prediction_mapped = model(market, news, history_outputs)
            # print(prediction.shape)

            history_outputs = torch.cat((history_outputs[:, 1:, :], prediction_mapped.detach().unsqueeze(1)), dim=1)

            torch.save(history_outputs, f"{args.history_output}history_outputs.pth")
            loss_ce = criterion(prediction, label)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss_ce + 0.01 * kl_div  # Variational autoencoder loss.
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss.item()

        # average loss
        avg_loss = total_loss / len(train_dataloader)
        avg_train_loss_list.append(avg_loss)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss}")

        swanlab.log({"avg_loss":  avg_loss})

        scheduler.step()

        if (epoch + 1) % 5 == 0:

            val_loss, val_acc, val_mcc, val_f1 = validate_StockNet(model, val_dataloader, device, criterion, history_outputs, args)

            model.train()

            avg_val_loss_list.append(val_loss)
            avg_acc_list.append(val_acc)
            avg_mcc_list.append(val_mcc)
            print(f"Epoch {epoch + 1}/{args.epochs}, Val Loss: {val_loss}, Val Acc: {val_acc}, Val Mcc: {val_mcc}, Val F1: {val_f1}")

            swanlab.log({"Val_loss": val_loss, "Val_acc": val_acc, "Val_mcc": val_mcc, "Val_f1": val_f1})

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


def validate_StockNet(model, val_dataloader, device, criterion, history_outputs, args):

    model.eval()
    total_loss, correct, total_samples = 0, 0, 0
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for batch in val_dataloader:
            market, news, label = batch["market"].to(device), batch["news"].to(device), batch["label"].to(device)
            label = label.long()
            if label.shape[0] != args.batch_size:

                label = torch.cat((label, torch.zeros(args.batch_size - label.shape[0], dtype=torch.long).to(device)),
                                  dim=0)

            prediction, mu, logvar, prediction_mapped = model(market, news, history_outputs)


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


def train_HAN(args):
    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}")
    else:
        device = torch.device("cpu")

    if args.model == "HAN":

        HAN_Dataset_Prepare(args)

        train_dataset = HAN_Dataset(args.train_x_path, args.train_y_path, args.days, args.max_num_tweets_len, args.max_num_tokens_len)
        val_dataset = HAN_Dataset(args.val_x_path, args.val_y_path, args.days, args.max_num_tweets_len, args.max_num_tokens_len)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

        model = HAN(hidden_size=args.hidden_size, bert_dim=args.bert_dim, pretrained_model=args.pretrained_model, days=args.days, max_num_tweets_len=args.max_num_tweets_len, dropout=args.dropout)

    else:
        raise ValueError("Invalid model name")

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

            val_loss, val_acc, val_mcc, val_f1_score = validate_HAN(model, val_dataloader, device, criterion)

            model.train()

            avg_val_loss_list.append(val_loss)
            avg_acc_list.append(val_acc)
            avg_mcc_list.append(val_mcc)
            print(
                f"Epoch {epoch + 1}/{args.epochs}, Val Loss: {val_loss}, Val Acc: {val_acc}, Val Mcc: {val_mcc}, Val F1_score: {val_f1_score}")

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
    plot_figure(avg_train_loss_list, avg_val_loss_list, avg_acc_list, avg_mcc_list, plot_epoch, args.figure_save_folder,
                args.model)
    print("finished plotting")


def validate_HAN(model, val_dataloader, device, criterion):
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

            total_samples += label.size(0) * label.size(1) if label.dim() > 1 else label.size(0)

            all_labels.extend(label.cpu().numpy().flatten())
            all_predictions.extend(predicted_classes.cpu().numpy().flatten())

    val_loss = total_loss / len(val_dataloader)
    val_acc = correct / total_samples
    val_mcc = matthews_corrcoef(all_labels, all_predictions)
    val_f1_score = f1_score(all_labels, all_predictions)
    return val_loss, val_acc, val_mcc, val_f1_score

def train_PEN(args):
    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}")
    else:
        device = torch.device("cpu")

    if args.model == "PEN":

        train_dataset = PEN_Dataset(args.train_price_folder, args.train_news_folder, args.days, args.max_num_tweets, args.max_num_tokens, args.pretrained_model)
        val_dataset = PEN_Dataset(args.val_price_folder, args.val_news_folder, args.days, args.max_num_tweets, args.max_num_tokens, args.pretrained_model)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

        model = PEN(args.pretrained_model, args.max_num_tweets, args.max_num_tokens, args.hidden_size, args.dropout)

    else:
        raise ValueError("Invalid model name")

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

            val_loss, val_acc, val_mcc, val_f1_score = validate_HAN(model, val_dataloader, device, criterion)

            model.train()

            avg_val_loss_list.append(val_loss)
            avg_acc_list.append(val_acc)
            avg_mcc_list.append(val_mcc)
            print(
                f"Epoch {epoch + 1}/{args.epochs}, Val Loss: {val_loss}, Val Acc: {val_acc}, Val Mcc: {val_mcc}, Val F1_score: {val_f1_score}")

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
    plot_figure(avg_train_loss_list, avg_val_loss_list, avg_acc_list, avg_mcc_list, plot_epoch, args.figure_save_folder,
                args.model)
    print("finished plotting")


def validate_PEN(model, val_dataloader, device, criterion):
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

            total_samples += label.size(0) * label.size(1) if label.dim() > 1 else label.size(0)

            all_labels.extend(label.cpu().numpy().flatten())
            all_predictions.extend(predicted_classes.cpu().numpy().flatten())

    val_loss = total_loss / len(val_dataloader)
    val_acc = correct / total_samples
    val_mcc = matthews_corrcoef(all_labels, all_predictions)
    val_f1_score = f1_score(all_labels, all_predictions)
    return val_loss, val_acc, val_mcc, val_f1_score
