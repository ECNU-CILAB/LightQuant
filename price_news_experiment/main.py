# author:Liu Yu
# time:2025/2/13 15:18
from price_experiment.my_parser import args
from train import train
from test import test
from train_adv_lstm import train_adv_lstm


if __name__ == '__main__':
    if args.model != 'adv_lstm':
        train()
    if args.model == 'adv_lstm':
        train_adv_lstm()

    test()

