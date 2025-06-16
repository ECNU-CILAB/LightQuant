import torch.nn as nn
from model.StockNet import StockNet
from model.HAN import HAN
from model.PEN import PEN
from utils.price_dataloader import *
from sklearn.metrics import matthews_corrcoef, f1_score
from utils.plot import *
from utils.price_news_dataloader import *

def test_StockNet(args):

    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}" )
    else:
        device = torch.device("cpu")

    if args.model == "StockNet":
        model = StockNet(5, 20, 128, 64, args.look_back_window)

    else:
        raise ValueError("Invalid model name.")

    model.load_state_dict(torch.load(f"{args.model_save_folder}{args.model}.pth", weights_only=True))

    model.to(device)

    test_dataset = StockNet_Dataset(args.test_price_folder, args.test_news_folder, look_back_window=args.look_back_window)

    # if .pkl file exists, use it
    # test_dataset = StockDatasetFromPickle("/home/users/liuyu/Framework/dataset/csi300/test/dataset.pkl")

    test_dataloader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    y_true = []
    y_pred = []

    if os.path.exists(f"{args.history_output}history_outputs.pth"):
        history_outputs = torch.load(f"{args.history_output}history_outputs.pth").to(device)
    else:
        raise ValueError("history_outputs.pth not found")
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            market, news, label = batch["market"].to(device), batch["news"].to(device), batch["label"].to(device)
            label = label.long()
            if label.shape[0] != args.batch_size:

                label = torch.cat((label, torch.zeros(args.batch_size - label.shape[0], dtype=torch.long).to(device)),
                                  dim=0)

            prediction, mu, logvar, prediction_mapped = model(market, news, history_outputs)

            # Update Historical Predictions
            history_outputs = torch.cat((history_outputs[:, 1:, :], prediction_mapped.detach().unsqueeze(1)), dim=1)

            loss_ce = criterion(prediction, label)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss_ce + 0.01 * kl_div
            total_loss += loss.item()


            _, predicted_classes = torch.max(prediction, 1)


            correct_predictions += (predicted_classes == label).sum().item()
            total_samples += label.size(0)

            y_true.extend(label.cpu().numpy())
            y_pred.extend(predicted_classes.cpu().numpy())

    avg_loss = total_loss / len(test_dataloader)
    accuracy = correct_predictions / total_samples
    mcc = matthews_corrcoef(y_true, y_pred)
    F1_score = f1_score(y_true, y_pred)

    print(f"Test Loss: {avg_loss}, Test Accuracy: {accuracy}, Test MCC: {mcc}, Test F1_score: {F1_score}")
    with open(f'{args.test_result_save_folder}{args.model}.txt', 'w') as file:
        file.write(f"Test Loss: {avg_loss}\n")
        file.write(f"Test Accuracy: {accuracy}\n")
        file.write(f"Test MCC: {mcc}\n")
        file.write(f"Test F1 Score: {F1_score}\n")


def test_HAN(args):
    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}")
    else:
        device = torch.device("cpu")

    if args.model == "HAN":

        HAN_Dataset_Prepare(args)

        test_dataset = HAN_Dataset(args.train_x_path, args.train_y_path, args.days, args.max_num_tweets_len, args.max_num_tokens_len)

        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

        model = HAN(hidden_size=args.hidden_size, bert_dim=args.bert_dim, pretrained_model=args.pretrained_model, days=args.days, max_num_tweets_len=args.max_num_tweets_len, dropout=args.dropout)

    else:
        raise ValueError("Invalid model name")

    model.load_state_dict(torch.load(f"{args.model_save_folder}{args.model}.pth", weights_only=True))

    model.to(device)

    model.eval()

    criterion = nn.BCELoss().to(device)
    total_loss, correct, total_samples = 0, 0, 0
    all_labels, all_predictions = [], []
    total_samples = 0

    with torch.no_grad():
        for idx, (input_data, label) in enumerate(test_dataloader):
            input_data = input_data.to(device)
            label = label.squeeze()
            label = torch.where(label == -1, torch.tensor(0), label).to(torch.int).float()
            label = label.to(device)

            prediction = model(input_data)
            prediction = prediction.squeeze()
            prediction = prediction.to(device)

            loss = criterion(prediction, label)
            total_loss += loss.item()

            predicted_classes = (prediction > 0.5).float()
            correct += (predicted_classes == label).sum().item()
            total_samples += label.size(0) * label.size(1)  if label.dim() > 1 else label.size(0)

            all_labels.extend(label.cpu().numpy().flatten())
            all_predictions.extend(predicted_classes.cpu().numpy().flatten())

    avg_loss = total_loss / len(test_dataloader)
    test_acc = correct / total_samples
    test_mcc = matthews_corrcoef(all_labels, all_predictions)
    test_f1_score = f1_score(all_labels, all_predictions)

    print(f"Test Loss: {avg_loss}, Test Accuracy: {test_acc}, Test MCC: {test_mcc}, Test f1_score: {test_f1_score}")

    os.makedirs(args.test_result_save_folder, exist_ok=True)
    with open(f'{args.test_result_save_folder}{args.model}.txt', 'w') as file:
        file.write(f"Test Loss: {avg_loss}\n")
        file.write(f"Test Accuracy: {test_acc}\n")
        file.write(f"Test MCC: {test_mcc}\n")
        file.write(f"Test F1 Score: {test_f1_score}\n")


def test_PEN(args):
    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}")
    else:
        device = torch.device("cpu")

    if args.model == "PEN":

        test_dataset = PEN_Dataset(args.test_price_folder, args.test_news_folder, args.days, args.max_num_tweets, args.max_num_tokens, args.pretrained_model)

        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

        model = PEN(args.pretrained_model, args.max_num_tweets, args.max_num_tokens, args.hidden_size, args.dropout)

    else:
        raise ValueError("Invalid model name")

    model.load_state_dict(torch.load(f"{args.model_save_folder}{args.model}.pth", weights_only=True))

    model.to(device)

    model.eval()

    criterion = nn.BCELoss().to(device)
    total_loss, correct, total_samples = 0, 0, 0
    all_labels, all_predictions = [], []
    total_samples = 0

    with torch.no_grad():
        for idx, (input_data, label) in enumerate(test_dataloader):
            input_data = input_data.to(device)
            label = label.squeeze()
            label = torch.where(label == -1, torch.tensor(0), label).to(torch.int).float()
            label = label.to(device)

            prediction = model(input_data)
            prediction = prediction.squeeze()
            prediction = prediction.to(device)

            loss = criterion(prediction, label)
            total_loss += loss.item()

            predicted_classes = (prediction > 0.5).float()
            correct += (predicted_classes == label).sum().item()
            total_samples += label.size(0) * label.size(1)  if label.dim() > 1 else label.size(0)

            all_labels.extend(label.cpu().numpy().flatten())
            all_predictions.extend(predicted_classes.cpu().numpy().flatten())

    avg_loss = total_loss / len(test_dataloader)
    test_acc = correct / total_samples
    test_mcc = matthews_corrcoef(all_labels, all_predictions)
    test_f1_score = f1_score(all_labels, all_predictions)

    print(f"Test Loss: {avg_loss}, Test Accuracy: {test_acc}, Test MCC: {test_mcc}, Test f1_score: {test_f1_score}")

    os.makedirs(args.test_result_save_folder, exist_ok=True)
    with open(f'{args.test_result_save_folder}{args.model}.txt', 'w') as file:
        file.write(f"Test Loss: {avg_loss}\n")
        file.write(f"Test Accuracy: {test_acc}\n")
        file.write(f"Test MCC: {test_mcc}\n")
        file.write(f"Test F1 Score: {test_f1_score}\n")