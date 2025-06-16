import os, sys
from tqdm import tqdm
from utils.price_dataloader import *
from metrics import *
from model.LSTM import LSTM
from model.ALSTM import ALSTM
from model.Adv_LSTM import AdvLSTM
from model.BiLSTM import BiLSTM
from model.DTML import DTML
from model.SCINet import SCINet


def calculate_mean_without_nan(data_list):

    clean_list = [x for x in data_list if not np.isnan(x)]

    return sum(clean_list) / len(clean_list) if clean_list else float('nan')


def backtest_normal(dataloader, model, device):
    money, stock = 10000, 0
    cost = 0.000
    asset_list = []
    predictions_list = []
    actuals_list = []
    init_price = None

    for input_data_normalized, input_data_raw, label in dataloader:
        input_data = input_data_normalized.to(device)
        label = label.squeeze().to(device).float()

        prediction = model(input_data)
        prediction = prediction.squeeze()
        pred_class = (prediction > 0.5).float()
        pred_value = pred_class.item()


        price = input_data_raw[:, -1, 3].item()

        predictions_list.append(pred_value)
        actuals_list.append(label.item())


        if pred_value == 1 and stock < 1:
            stock += 1
            money -= price
            money -= price * cost

        # short selling is allowed here.
        elif pred_value == 0 and stock > -1:
            stock -= 1
            money += price
            money -= price * cost

        # if you don't want to short sell, you can change the condition here.
        # elif pred_value == 0 and stock > 0:
        #     stock -= 1
        #     money += price
        #     money -= price * cost

        if init_price is None:
            init_price = price

        asset_value = (money + stock * price) / init_price
        asset_list.append(float(asset_value))

    ACC = calculate_ACC(actuals_list, predictions_list)
    ARR = calculate_ARR(asset_list)
    SR = calculate_SR(asset_list)
    MDD = calculate_MDD(asset_list)
    CR = calculate_Calmar_Ratio(ARR, MDD)
    IR = calculate_IR(asset_list)


    return ACC, ARR, SR, MDD, CR, IR

def backtest_dtml(dataloader, model, device, n_stocks):
    ACC_List, ARR_List, SR_List, MDD_List, CR_List, IR_List = [], [], [], [], [], []

    for i in range(n_stocks):
        money, stock = 10000, 0
        cost = 0.000
        asset_list = []
        predictions_list = []
        actuals_list = []
        init_price = None
        for input_data, label in dataloader:
            input_data = input_data.to(device)  # (1, n_stocks, seq_len, input_size)
            # print(f"input_data:{input_data.shape}")
            label = label.squeeze().to(device).float()  # (n_stocks, )

            prediction = model(input_data)  # (1, n_stocks)
            predictions = (prediction > 0.5).float().squeeze(0).cpu().numpy()  # (n_stocks,)
            actuals = label.cpu().numpy()

            pred_value = predictions[i]
            actual_value = actuals[i]

            predictions_list.append(pred_value)
            actuals_list.append(actual_value)

            price = input_data[0, -1, i, 3].item()  # (seq_len, input_size)

            if pred_value == 1 and stock < 1:
                stock += 1
                money -= price
                money -= price * cost

            # short selling is allowed here.
            elif pred_value == 0 and stock > -1:
                stock -= 1
                money += price
                money -= price * cost

            # if you don't want to short sell, you can change the condition here.
            # elif pred_value == 0 and stock > 0:
            #     stock -= 1
            #     money += price
            #     money -= price * cost

            if init_price is None:
                init_price = price

            asset_value = (money + stock * price) / init_price
            asset_list.append(float(asset_value))


        ACC = calculate_ACC(actuals_list, predictions_list)
        ARR = calculate_ARR(asset_list)
        SR = calculate_SR(asset_list)
        MDD = calculate_MDD(asset_list)
        CR = calculate_Calmar_Ratio(ARR, MDD)
        IR = calculate_IR(asset_list)

        ACC_List.append(ACC)
        ARR_List.append(ARR)
        SR_List.append(SR)
        MDD_List.append(MDD)
        CR_List.append(CR)
        IR_List.append(IR)

    return ACC_List, ARR_List, SR_List, MDD_List, CR_List, IR_List

def backtest_scinet(dataloader, model, device):

    money, stock = 10000, 0
    cost = 0.000
    asset_list = []
    predictions_list = []
    actuals_list = []
    init_price = None

    for input_data, raw_close_price, label in dataloader:
        input_data = input_data.to(device)
        label = label.squeeze().to(device).float()

        prediction = model(input_data)
        pred_value = (prediction[:, 0] > 0.5).float().item()
        actual_value = label.item()

        predictions_list.append(pred_value)
        actuals_list.append(actual_value)

        price = raw_close_price.item()

        if pred_value == 1 and stock < 1:
                stock += 1
                money -= price
                money -= price * cost

            # short selling is allowed here.
        elif pred_value == 0 and stock > -1:
            stock -= 1
            money += price
            money -= price * cost

        # if you don't want to short sell, you can change the condition here.
        # elif pred_value == 0 and stock > 0:
        #     stock -= 1
        #     money += price
        #     money -= price * cost

        if init_price is None:
            init_price = price

        asset_value = (money + stock * price) / init_price
        asset_list.append(float(asset_value))

    ACC = calculate_ACC(actuals_list, predictions_list)
    ARR = calculate_ARR(asset_list)
    SR = calculate_SR(asset_list)
    MDD = calculate_MDD(asset_list)
    CR = calculate_Calmar_Ratio(ARR, MDD)
    IR = calculate_IR(asset_list)

    return ACC, ARR, SR, MDD, CR, IR


def backtest_single(args):
    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}" )
    else:
        device = torch.device("cpu")

    if args.model == "lstm":
        model = LSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first )

    elif args.model == "alstm":
        model = ALSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first, attention_size=args.attention_size )

    elif args.model == "adv_lstm":
        model = AdvLSTM(input_size=args.input_size, hidden_size=args.hidden_size, output_size=1, attention_size=args.attention_size, perturbation_size=args.perturbation_size, epsilon=args.epsilon)

    elif args.model == "bi_lstm":
        model = BiLSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first )

    elif args.model == "dtml":
        model = DTML(input_size=7, hidden_size=64, num_layers=2, n_heads=4)

    elif args.model == "scinet":
        model = SCINet(input_len=args.seq_len, pred_len=args.pred_len, input_dim=8, hidden_dim=args.hidden_size,
                       SCINet_Layers=args.SCINet_Layers)

    else:
        raise ValueError(f"Invalid model name: {args.model}")

    model.load_state_dict(torch.load(f"{args.model_save_folder}/{args.model}.pth", weights_only=True))

    model.to(device)

    model.eval()

    ACC_List, ARR_List, SR_List, MDD_List, CR_List, IR_List = [], [], [], [], [], []

    if args.model == "lstm" or args.model == "bi_lstm" or args.model == "alstm" or args.model == "adv_lstm":
        for file in tqdm(os.listdir(args.test_price_folder)):
            if not file.endswith(".csv"):
                continue
            file_path = os.path.join(args.test_price_folder, file)
            backtest_dataset = create_dataset(backtest_file=file_path, look_back_window=args.look_back_window)

            backtest_dataloader = create_dataloader(backtest_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

            ACC, ARR, SR, MDD, CR, IR = backtest_normal(backtest_dataloader, model, device)

            ACC_List.append(ACC)
            ARR_List.append(ARR)
            SR_List.append(SR)
            MDD_List.append(MDD)
            CR_List.append(CR)
            IR_List.append(IR)

    elif args.model == "dtml":
        backtest_dataset = DTML_Dataset(
            data_folder=args.test_price_folder,
            look_back_window=args.look_back_window,
            n_stocks=args.n_stocks
        )
        backtest_dataloader = create_dataloader(backtest_dataset, batch_size=1, shuffle=False, drop_last=False)

        ACC_List, ARR_List, SR_List, MDD_List, CR_List, IR_List = backtest_dtml(backtest_dataloader, model, device, n_stocks=args.n_stocks)

    elif args.model == "scinet":
        for file in tqdm(os.listdir(args.test_price_folder)):
            if not file.endswith(".csv"):
                continue
            file_path = os.path.join(args.test_price_folder, file)
            backtest_dataset = Backtest_SCINet_Dataset(file_path, args.seq_len, args.pred_len)
            backtest_dataloader = create_dataloader(backtest_dataset, batch_size=1, shuffle=False, drop_last=False)

            ACC, ARR, SR, MDD, CR, IR = backtest_scinet(backtest_dataloader, model, device)

            ACC_List.append(ACC)
            ARR_List.append(ARR)
            SR_List.append(SR)
            MDD_List.append(MDD)
            CR_List.append(CR)
            IR_List.append(IR)

    print(f"mean ACC: {calculate_mean_without_nan(ACC_List)}")
    print(f"mean ARR: {calculate_mean_without_nan(ARR_List)}")
    print(f"mean SR: {calculate_mean_without_nan(SR_List)}")
    print(f"mean MDD: {calculate_mean_without_nan(MDD_List)}")
    print(f"mean CR: {calculate_mean_without_nan(CR_List)}")
    print(f"mean IR: {calculate_mean_without_nan(IR_List)}")

    os.makedirs(args.backtest_result_save_folder, exist_ok=True)
    with open(f'{args.backtest_result_save_folder}/{args.model}.txt', 'w') as file:
        file.write(f"mean ACC: {calculate_mean_without_nan(ACC_List)}\n")
        file.write(f"mean ARR: {calculate_mean_without_nan(ARR_List)}\n")
        file.write(f"mean SR: {calculate_mean_without_nan(SR_List)}\n")
        file.write(f"mean MDD: {calculate_mean_without_nan(MDD_List)}\n")
        file.write(f"mean CR: {calculate_mean_without_nan(CR_List)}\n")
        file.write(f"mean IR: {calculate_mean_without_nan(IR_List)}")







