# author:Liu Yu
# time:2025/2/13 15:19
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
def test():
    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}" )
    else:
        device = torch.device("cpu")

    if args.model == "lstm":
        model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=2, dropout=args.dropout, batch_first=args.batch_first )

    if args.model == "mlp":
        model = SimpleMLPModel()

    model.load_state_dict(torch.load(f"{args.model_save_folder}{args.model}.pth", weights_only=True))
    model.to(device)

    _, _, test_dataset = create_dataset(train_folder=None, val_folder=None, test_folder=args.test_folder, look_back=args.look_back)

    test_dataloader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model.eval()  # 将模型设置为评估模式

    criterion = nn.CrossEntropyLoss().to(device)  # 定义损失函数
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    y_true = []
    y_pred = []

    with torch.no_grad():  # 在测试时，不需要计算梯度
        for idx, (input_data, label) in enumerate(test_dataloader):

            input_data = input_data.to(device)
            label = label.squeeze()
            label = torch.where(label == -1, torch.tensor(0), label).to(torch.int).long()
            label = label.to(device)
            prediction = model(input_data)
            prediction = prediction.squeeze()
            prediction = prediction.to(device)

            loss = criterion(prediction, label)
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

    print(f"Test Loss: {avg_loss}, Test Accuracy: {accuracy}, Test MCC: {mcc}")
    with open(f'{args.test_result_save_folder}{args.model}.txt', 'w') as file:
        file.write(f"Test Loss: {avg_loss}\n")
        file.write(f"Test Accuracy: {accuracy}\n")
        file.write(f"Test MCC: {mcc}\n")