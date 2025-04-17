# author:Liu Yu
# time:2025/2/13 15:19
import torch.nn as nn
from model.LSTM import lstm
from model.ALSTM import ALSTM
from model.Adv_LSTM import AdvLSTM
from model.BiLSTM import BiLSTM
from model.BiGRU import BiGRU
from model.StockNet import StockNet
from utils.price_dataloader import *
from sklearn.metrics import matthews_corrcoef, f1_score
from my_parser import args
from utils.plot import *
from utils.price_news_dataloader import *
def test(history_outputs):

    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}" )
    else:
        device = torch.device("cpu")

    if args.model == "lstm":
        model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2, dropout=args.dropout, batch_first=args.batch_first )

    if args.model == "alstm":
        model = ALSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2, dropout=args.dropout, batch_first=args.batch_first, attention_size=args.attention_size )

    if args.model == "adv_lstm":
        model = AdvLSTM(input_size=args.input_size, hidden_size=args.hidden_size, output_size=2, attention_size=args.attention_size, perturbation_size=args.perturbation_size, epsilon=args.epsilon)

    if args.model == "bi_lstm":
        model = BiLSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2, dropout=args.dropout, batch_first=args.batch_first )

    if args.model == "stocknet":
        model = StockNet(5, 20, 128, 64, 10)

    model.load_state_dict(torch.load(f"{args.model_save_folder}{args.model}.pth", weights_only=True))

    model.to(device)

    # test_dataset = StockDataset(args.test_price_folder, args.test_news_folder, look_back_window=args.look_back_window)
    test_dataset = StockDatasetFromPickle("/home/users/liuyu/Framework/dataset/csi300/test/dataset.pkl")

    test_dataloader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model.eval()  # 将模型设置为评估模式

    criterion = nn.CrossEntropyLoss().to(device)  # 定义损失函数
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    y_true = []
    y_pred = []

    with torch.no_grad():  # 在测试时，不需要计算梯度
        for idx, batch in enumerate(test_dataloader):
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

            # 获取预测类别
            _, predicted_classes = torch.max(prediction, 1)

            # 计算准确度
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