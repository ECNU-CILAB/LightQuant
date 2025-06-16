import os, sys
# 添加项目根目录到 PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import argparse
from backtest_single import backtest_single
from backtest_multi import backtest_multi


if __name__ == "__main__":

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="CSMD50", type=str)
    parser.add_argument('--test_price_folder', default=os.path.join(project_root, 'dataset', parser.get_default('dataset'), 'test', 'price'), type=str)
    parser.add_argument('--test_news_folder', default=os.path.join(project_root, 'dataset', parser.get_default('dataset'), 'test', 'news_embedding'), type=str)
    parser.add_argument('--use_news', default=True, type=bool, help='use news or not')
    parser.add_argument('--model', default="StockNet" ,type=str)  # lstm, alstm, bi_lstm, adv_lstm, dtml, scinet, StockNet, HAN
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--input_size', default=5, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--attention_size', default=128, type=int)
    parser.add_argument('--look_back_window', default=7, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--useGPU', default=True, type=bool)
    parser.add_argument('--GPU_ID', default=1, type=int)
    parser.add_argument('--batch_first', default=True, type=bool)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--model_save_folder', default=os.path.join(project_root, 'result', parser.get_default('dataset'), 'model_saved'))
    parser.add_argument('--backtest_result_save_folder',
                        default=os.path.join(project_root, 'backtest', 'result', parser.get_default('dataset')))
    parser.add_argument('--epsilon', default=0.1, type=float, help='epsilon for adversarial lstm')
    parser.add_argument('--perturbation_size', default=0.1, type=float, help='perturbation size for adversarial lstm')
    parser.add_argument('--n_stocks', default=5, type=int, help='stocks number for DTML')
    parser.add_argument('--SCINet_Layers', default=3, type=int, help='SCINet layers number')
    parser.add_argument('--seq_len', default=32, type=int, help='input sequence length for SCINet')
    parser.add_argument('--pred_len', default=5, type=int, help='prediction length for SCINet')
    parser.add_argument('--history_output', default=os.path.join(project_root, 'result', parser.get_default('dataset'), 'history_output'), help='history outputs for StockNet')

    args = parser.parse_args()

    if args.use_news == False:
        backtest_single(args)
    else:
        backtest_multi(args)
