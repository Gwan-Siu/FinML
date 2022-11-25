from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from torch.utils.data import DataLoader
from dataset.dataset import Dataset

def create_loaders(handler, segments, opt):

    dataset = DatasetH(handler, segments)

    df_train, df_valid, df_test = dataset.prepare(["train", "valid", "test"], col_set=["feature", "label"],
                                                  data_key=DataHandlerLP.DK_L, )

    train_set = Dataset(df_train["feature"], df_train["label"])
    train_loader = DataLoader(train_set, batch_size=opt["train"]["batch_size"], shuffle=opt["train"]["shuffle"])

    val_set = Dataset(df_valid["feature"], df_valid["label"])
    val_loader = DataLoader(val_set, batch_size=opt["val"]["batch_size"])

    test_set = Dataset(df_test["feature"], df_test["label"])
    test_loader = DataLoader(test_set, batch_size=opt["test"]["batch_size"])

    return train_loader, val_loader, test_loader
