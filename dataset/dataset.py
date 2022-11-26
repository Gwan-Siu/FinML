import numpy as np

import torch.utils.data as data
from torch.utils.data import DataLoader
from qlib.data.dataset.handler import DataHandlerLP


def create_dataloder_H(dataset, loader_configs):

    df_train, df_valid, df_test = dataset.prepare(["train", "valid", "test"], col_set=["feature", "label"],
                                                  data_key=DataHandlerLP.DK_L)
    train_set = Dataset(df_train["feature"], df_train["label"])
    train_loader = DataLoader(train_set, batch_size=loader_configs["train"]["batch_size"],
                              shuffle=loader_configs["train"]["shuffle"])

    val_set = Dataset(df_valid["feature"], df_valid["label"])
    val_loader = DataLoader(val_set, batch_size=loader_configs["val"]["batch_size"])

    test_set = Dataset(df_test["feature"], df_test["label"])
    test_loader = DataLoader(test_set, batch_size=loader_configs["test"]["batch_size"])
    return train_loader, val_loader, test_loader

def create_dataloder_TS(dataset, loader_configs):

    df_train, df_valid, df_test = dataset.prepare(["train", "valid", "test"], col_set=["feature", "label"],
                                                  data_key=DataHandlerLP.DK_L)


    df_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
    df_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
    df_test.config(fillna_type="ffill+bfill")

    train_loader = DataLoader(df_train, batch_size=loader_configs["train"]["batch_size"],
                              shuffle=loader_configs["train"]["shuffle"], drop_last=True)
    val_loader = DataLoader(df_valid, batch_size=loader_configs["val"]["batch_size"], drop_last=True)
    test_loader = DataLoader(df_test, batch_size=loader_configs["test"]["batch_size"], drop_last=True)

    return train_loader, val_loader, test_loader

class Dataset(data.Dataset):
    def __init__(self, df_feats, df_label):

        assert len(df_feats) == len(df_label)

        self.feats = df_feats.values # size:  NxF
        self.label = df_label.values # size: Nx1
        self.index = df_label.index


    def __getitem__(self, item):

        x = self.feats[item, :]
        y = self.label[item, :] # to be noted that the type of label is float64
        index = self.index[item]

        index_time = index[0].strftime('%Y-%m-%d')
        stock_index = index[1]
        return x, y.squeeze(), index_time, stock_index


    def __len__(self):
        return self.label.shape[0]
