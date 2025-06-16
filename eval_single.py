from single_exp.train import train
from single_exp.test import test


def eval_single(args):

    train(args)

    test(args)

