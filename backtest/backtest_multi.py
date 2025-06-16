from model.StockNet import StockNet
from model.HAN import HAN
from model.PEN import PEN
from tqdm import tqdm
from utils.price_news_dataloader import *
from metrics import *


def calculate_mean_without_nan(data_list):

    clean_list = [x for x in data_list if not np.isnan(x)]

    return sum(clean_list) / len(clean_list) if clean_list else float('nan')


def backtest_normal(dataloader, model, device, args):
    money, stock = 10000, 0
    cost = 0.000
    asset_list = []
    predictions_list = []
    actuals_list = []
    init_price = None

    if os.path.exists(f"{args.history_output}/history_outputs.pth"):
        history_outputs = torch.load(f"{args.history_output}/history_outputs.pth").to(device)
    else:
        raise ValueError(f"{args.history_output}/history_outputs.pth not found")

    for price_normalized, price_raw, news, label in dataloader:
        price = price_normalized.to(device)
        news = news.to(device)
        label = label.long().to(device)

        if label.shape[0] != 1:
            label = torch.cat((label, torch.zeros(1 - label.shape[0], dtype=torch.long).to(device)),dim=0)

        prediction, mu, logvar, prediction_mapped = model(price, news, history_outputs)

        pred_classes = torch.argmax(prediction, dim=1).cpu().numpy()
        label = label.cpu().numpy()


        if label.ndim == 0:
            label = [label.item()]
            pred_classes = [pred_classes.item()]
        else:
            label = label.tolist()
            pred_classes = pred_classes.tolist()

        for pred_i, label_i in zip(pred_classes, label):
            predictions_list.append(pred_i)
            actuals_list.append(label_i)

            price = price_raw[-1, -1, 3].item()

            if pred_i == 1 and stock < 1:
                stock += 1
                money -= price
                money -= price * cost
            elif pred_i == 0 and stock > -1:
                stock -= 1
                money += price
                money -= price * cost

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



def backtest_multi(args):
    if args.useGPU:
        device = torch.device(f"cuda:{args.GPU_ID}" )
    else:
        device = torch.device("cpu")

    if args.model == "StockNet":
        model = StockNet(5, 20, 128, 64, args.look_back_window)

    else:
        raise ValueError(f"Invalid model name: {args.model}")

    model.load_state_dict(torch.load(f"{args.model_save_folder}/{args.model}.pth", weights_only=True))

    model.to(device)

    model.eval()

    ACC_List, ARR_List, SR_List, MDD_List, CR_List, IR_List = [], [], [], [], [], []

    price_files = [f for f in os.listdir(args.test_price_folder) if f.endswith('.csv')]

    for price_file in tqdm(price_files):

        stock_id = price_file.replace('.csv', '')


        price_file_path = os.path.join(args.test_price_folder, price_file)
        news_folder_path = os.path.join(args.test_news_folder, stock_id)

        if not os.path.exists(news_folder_path):
            print(f"warning：{stock_id} lack of news")
            continue

        backtest_dataset = create_dataset(price_file_path, news_folder_path, args.look_back_window)

        test_dataloader = create_dataloader(backtest_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

        ACC, ARR, SR, MDD, CR, IR = backtest_normal(test_dataloader, model, device, args)

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