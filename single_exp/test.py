import torch.nn as nn
import torch
from model.LSTM import LSTM
from model.ALSTM import ALSTM
from model.Adv_LSTM import AdvLSTM
from model.BiLSTM import BiLSTM
from model.DTML import DTML
from model.SCINet import SCINet
from utils.price_dataloader import *
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm
from utils.plot import *

def test(args):
    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}")
    else:
        device = torch.device("cpu")

    if args.model == "dtml":
        test_dataset = DTML_Dataset(args.test_price_folder, args.look_back_window)

    elif args.model == "scinet":
        test_dataset = SCINet_Dataset(args.test_price_folder, args.seq_len, args.pred_len)

    else:
        _, _, test_dataset = create_dataset(train_folder=None, val_folder=None, test_folder=args.test_price_folder,
                                        look_back_window=args.look_back_window)

    test_dataloader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    if args.model == "lstm":
        model = LSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers, output_size=1,
                     dropout=args.dropout, batch_first=args.batch_first)

    elif args.model == "alstm":
        model = ALSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers, output_size=1,
                      dropout=args.dropout, batch_first=args.batch_first, attention_size=args.attention_size)

    elif args.model == "adv_lstm":
        model = AdvLSTM(input_size=args.input_size, hidden_size=args.hidden_size, output_size=1,
                        attention_size=args.attention_size, perturbation_size=args.perturbation_size,
                        epsilon=args.epsilon)

    elif args.model == "bi_lstm":
        model = BiLSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers, output_size=1,
                       dropout=args.dropout, batch_first=args.batch_first)

    elif args.model == "dtml":
        model = DTML(input_size = 7, hidden_size = 64, num_layers = 2, n_heads = 4)

    elif args.model == "scinet":
        model = SCINet(input_len=args.seq_len, pred_len=args.pred_len, input_dim=8, hidden_dim=args.hidden_size,
                       SCINet_Layers=args.SCINet_Layers)

    else:
        raise ValueError(f"Invalid model name: {args.model}")

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