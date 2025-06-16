import argparse
import torch
import random
import numpy as np
from eval_single import eval_single
from eval_multi import eval_multi
import swanlab

def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(37)

    parser = argparse.ArgumentParser()


    parser.add_argument('--dataset', default="CSMD50", type=str)
    parser.add_argument('--train_price_folder', default='./dataset/{}/train/price'.format(parser.get_default('dataset')),
                        type=str)
    parser.add_argument('--val_price_folder', default='./dataset/{}/val/price'.format(parser.get_default('dataset')),
                        type=str)
    parser.add_argument('--test_price_folder', default='./dataset/{}/test/price'.format(parser.get_default('dataset')),
                        type=str)
    parser.add_argument('--train_news_folder', default='./dataset/{}/train/news_embedding'.format(parser.get_default('dataset')),
                        type=str)
    parser.add_argument('--val_news_folder', default='./dataset/{}/val/news_embedding'.format(parser.get_default('dataset')),
                        type=str)
    parser.add_argument('--test_news_folder', default='./dataset/{}/test/news_embedding'.format(parser.get_default('dataset')),
                        type=str)
    parser.add_argument('--use_news', default=True, type=bool, help='use news or not')
    parser.add_argument('--model', default="HAN", type=str) # lstm, alstm, bi_lstm, adv_lstm, dtml, scinet, StockNet, HAN, PEN
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--input_size', default=5, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--attention_size', default=128, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--look_back_window', default=7, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--useGPU', default=True, type=bool)
    parser.add_argument('--GPU_ID', default=1, type=int)
    parser.add_argument('--batch_first', default=True, type=bool)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--model_save_folder', default='./result/{}/model_saved/'.format(parser.get_default('dataset')))
    parser.add_argument('--figure_save_folder', default='./result/{}/figure/'.format(parser.get_default('dataset')))
    parser.add_argument('--test_result_save_folder',default='./result/{}/test_result/'.format(parser.get_default('dataset')))
    parser.add_argument('--swanlab_api', default='qyetvQOMh2976y7gjjdNZ')
    parser.add_argument( '--swanlab_project', default='framework')
    parser.add_argument( '--swanlab_workspace', default='allen333')
    parser.add_argument('--epsilon', default=0.1, type=float, help='epsilon for adversarial lstm')
    parser.add_argument('--perturbation_size', default=0.1, type=float, help='perturbation size for adversarial lstm')
    parser.add_argument('--n_stocks', default=5, type=int, help='stocks number for DTML')
    parser.add_argument('--SCINet_Layers', default=3, type=int, help='SCINet layers number')
    parser.add_argument('--seq_len', default=32, type=int, help='input sequence length for SCINet')
    parser.add_argument('--pred_len', default=5, type=int, help='prediction length for SCINet')
    parser.add_argument('--history_output', default='./result/{}/history_output/'.format(parser.get_default('dataset')), help='storage history outputs for StockNet')
    parser.add_argument('--pretrained_model', type=str, default='yiyanghkust/finbert-pretrain', help='pretrained financial model for HAN')
    parser.add_argument('--train_start_date', type=str, default='2021-01-01', help='train start date for HAN dataset')
    parser.add_argument('--train_end_date', type=str, default='2024-03-14', help='train end date for HAN dataset')
    parser.add_argument('--dev_start_date', type=str, default='2024-03-15', help='dev start date for HAN dataset')
    parser.add_argument('--dev_end_date', type=str, default='2024-08-07', help='dev end date for HAN dataset')
    parser.add_argument('--test_start_date', type=str, default='2024-08-08', help='test start date for HAN dataset')
    parser.add_argument('--test_end_date', type=str, default='2024-12-31', help='test end date for HAN dataset')
    parser.add_argument('--max_num_tweets_len', type=int, default=20, help='max number of tweets for HAN')
    parser.add_argument('--max_num_tokens_len', type=int, default=30, help='max number of tokens for HAN')
    parser.add_argument('--days', type=int, default=5, help='days for HAN')
    parser.add_argument('--bert_dim', type=int, default= 768, help='bert dim for HAN')



    args = parser.parse_args()

    swanlab.login(api_key=args.swanlab_api, save=True)

    # Initialize a new SwanLab run to track this script
    swanlab.init(
        # configurate your own infor
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        experiment_name=args.model,
        config={
            "learning_rate": args.lr,
            "model": args.model,
            "dataset": args.dataset,
            "epochs": args.epochs
        },
    )

    if args.use_news == False:
        eval_single(args)
    else:
        eval_multi(args)