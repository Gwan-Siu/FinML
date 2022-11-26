import os, importlib
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import AverageMeter, get_logger, read_yaml, get_time_str, metric_fn
import pandas as pd

# from utils.lr_schedule import CosineAnnealingRestartLR

from dataset.create_dataloader import create_loaders

import wandb

import qlib
from qlib.config import REG_US, REG_CN

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--option', default='./options/test/LSTM/test_LSTM_alpha360.yml', type=str, help='model name')

## parse the args
args = parser.parse_args()
opt = read_yaml(args.option)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt['id_gpu'])

provider_uri = opt['provider_uri'] # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

def valid(valid_loader, model):

    model.eval()
    losses = []
    preds = []

    for val_pairs in valid_loader:
        input_var, target_var, time_index, stock_index = val_pairs[0], val_pairs[1], val_pairs[2], val_pairs[3]
        time_index = pd.to_datetime(time_index, format='%Y-%m-%d')
        input_var = input_var.cuda()
        target_var = target_var.cuda()
        with torch.no_grad():
            pred = model(input_var)
            loss = F.mse_loss(pred, target_var)
            losses.append(loss.cpu().numpy())
            preds.append(pd.DataFrame({ 'datetime': time_index, 'stock': stock_index, 'pred': pred.cpu().numpy(), 'label': target_var.cpu().numpy()}))

    #evaluate
    mean_loss = sum(losses) / len(losses)
    preds = pd.concat(preds, axis=0)
    precision, recall, ic, rank_ic = metric_fn(preds)

    res ={
        "loss": mean_loss,
        "ic": ic,
        "rank_ic": rank_ic,
        "precision": precision
    }

    return res


def main():
    seed = opt['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ### create dataset
    handler = opt['handler']
    segments = opt['segments']

    name = opt["name"]
    ## check the save_folder
    save_dir = os.path.join('./results/', name)

    if os.path.exists(save_dir):
        new_name = save_dir + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}')
        os.rename(save_dir, new_name)
    os.makedirs(save_dir)

    ## set the log
    log_path = save_dir + '/test_record.log'
    logger = get_logger(log_path)
    logger.info('Start the Inference.')

    train_loader, val_loader, test_loader = create_loaders(handler, segments, opt["dataset"])

    ## load the model
    model_file = '.arch' + '.' + opt['model']['name'] + '_arch'
    model = importlib.import_module(model_file, package='model').Model()

    # device_ids = opt['device_ids']
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.to('cpu')

    ## inference
    pretrain_path = opt['path']['pretrain_net']
    model_info = torch.load(pretrain_path)
    print('==> loading the best model:\n')
    model.load_state_dict(model_info['state_dict'])

    test_metric_loss = valid(test_loader, model)
    logger.info("the testing loss is {:.4f}".format(test_metric_loss["loss"]))
    logger.info("the IC is {:.6f}, the rank IC is {:.6f}, the precision@3 is {:.6f}".format(test_metric_loss["ic"], test_metric_loss["rank_ic"], test_metric_loss["precision"][3]))

if __name__ == '__main__':
    main()