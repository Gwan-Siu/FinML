from qlib.data.dataset import DatasetH, TSDatasetH
from dataset.dataset import create_dataloder_H, create_dataloder_TS

def create_loaders(opt):

    if opt['class'] == 'DatasetH':
        dataset = DatasetH(**opt['kwargs'])
        train_loader, val_loader, test_loader = create_dataloder_H(dataset, opt['dataloader'])
    elif opt['class'] == 'TSDatasetH':
        dataset = TSDatasetH(**opt['kwargs'])
        train_loader, val_loader, test_loader = create_dataloder_TS(dataset, opt['dataloader'])
    else:
        raise RuntimeError("No other data classes.")

    return train_loader, val_loader, test_loader
