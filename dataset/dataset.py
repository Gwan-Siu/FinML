import numpy as np

import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, df_feats, df_label):

        assert len(df_feats) == len(df_label)

        self.feats = df_feats.values # size:  NxF
        self.label = df_label.values # size: Nx1
        self.index = df_label.index


    def __getitem__(self, item):

        x = self.feats[item, :].astype(np.float32)
        y = self.label[item, :].astype(np.float32)  # to be noted that the type of label is float64
        index = self.index[item]

        index_time = index[0].strftime('%Y-%m-%d')
        stock_index = index[1]
        return x, y.squeeze(), index_time, stock_index


    def __len__(self):
        return self.label.shape[0]
