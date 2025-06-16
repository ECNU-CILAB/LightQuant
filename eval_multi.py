from multi_exp.train import *
from multi_exp.test import *

def eval_multi(args):
    if args.model == 'StockNet':
        train_StockNet(args)
    if  args.model != 'StockNet':
        train_HAN(args)

    if args.model == 'StockNet':
        test_StockNet(args)
    if args.model != 'StockNet':
        test_HAN(args)


