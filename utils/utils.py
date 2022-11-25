import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import yaml
import time
import pandas as pd

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def read_yaml(path):
    with open(path, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            return parsed_yaml
        except yaml.YAMLError as exc:
            print(exc)

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def comp_precision(preds):
    precision = {}
    recall = {}
    temp = preds.groupby('datetime', group_keys=False).apply(lambda x: x.sort_values(by='pred', ascending=False))
    # if len(temp.index[0]) > 2:
    #     temp = temp.reset_index(level=0).drop('datetime', axis=1)

    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = temp.groupby('datetime', group_keys=False).apply(lambda x: (x.label.iloc[:k] > 0).sum() / k).mean()
        recall[k] = temp.groupby('datetime', group_keys=False).apply(lambda x: (x.label.iloc[:k] > 0).sum() / (x.label > 0).sum()).mean()

    return precision, recall

def metric_fn(preds):
    preds = preds[~np.isnan(preds['label'])]

    precision, recall = comp_precision(preds)

    ic = preds.groupby('datetime', group_keys=False).apply(lambda x: x.label.corr(x.pred)).mean()
    rank_ic = preds.groupby('datetime', group_keys=False).apply(lambda x: x.label.corr(x.pred, method='spearman')).mean()

    return precision, recall, ic, rank_ic