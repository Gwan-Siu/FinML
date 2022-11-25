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
parser.add_argument('--option', default='./options/train/LSTM/train_LSTM.yml', type=str, help='model name')

## parse the args
args = parser.parse_args()
opt = read_yaml(args.option)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt['id_gpu'])

provider_uri = opt['provider_uri'] # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

def train(train_loader, model, optimizer):
    batch_loss = AverageMeter()

    model.train()
    if opt['train']['loss'] == 'mse':
        loss_criterion = torch.nn.MSELoss()

    loss_criterion = loss_criterion.cuda()

    for train_pairs in train_loader:
        input_var, target_var = train_pairs[0], train_pairs[1]

        input_var = input_var.cuda()
        target_var = target_var.cuda()
        output = model(input_var)

        loss = loss_criterion(output, target_var)
        batch_loss.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return batch_loss.avg


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

    name = opt['name']
    if opt["wandb"]["use"]:
        wandb.init(project=opt['wandb']['project'], entity=opt['wandb']['entity'], name=name)

    ### create dataset
    handler = opt['handler']
    segments = opt['segments']

    train_loader, val_loader, test_loader = create_loaders(handler, segments, opt["dataset"])

    ## load the model
    model_file = '.arch' + '.' + opt['model']['name'] + '_arch'
    model = importlib.import_module(model_file, package='model').Model()

    # device_ids = opt['device_ids']
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.to('cpu')

    model = nn.DataParallel(model)

    if opt['path']['resume']:
        pretrain_path = opt['path']['pretrain_net']
        model_info = torch.load(pretrain_path)
        print('==> loading existing model:')
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt['train']['total_epoch'], eta_min=opt['train']['scheduler']['eta_min'])
        optimizer.load_state_dict(model_info['optimizer'])
        scheduler.load_state_dict(model_info['scheduler'])
        cur_epoch = model_info['epoch']
        best_loss = model_info['loss']
    else:
        if opt['train']['optim_g']['type'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt['train']['optim_g']['lr'], weight_decay=opt['train']['optim_g']['weight_decay'], betas=opt['train']['optim_g']['betas'])
        if opt['train']['optim_g']['type'] == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=opt['train']['optim_g']['lr'], weight_decay=opt['train']['optim_g']['weight_decay'], betas=opt['train']['optim_g']['betas'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt['train']['total_epoch'], eta_min=opt['train']['scheduler']['eta_min'])
        cur_epoch = 0
        best_loss = float('inf')

    name = opt["name"]
    ## check the save_folder
    save_dir = os.path.join('./experiments/', name)

    if os.path.exists(save_dir):
        new_name = save_dir + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}')
        os.rename(save_dir, new_name)
    os.makedirs(save_dir)

    ## set the log
    log_path = save_dir + '/record.log'
    logger = get_logger(log_path)
    logger.info('Star the training')
    logger.info('the learning rate is {}'.format(opt['train']['optim_g']['lr']))
    logger.info('the total number of epoches is {}'.format(opt['train']['total_epoch']))

    if opt["wandb"]["use"]:
        wandb.config = {
            "learning_rate": opt['train']['optim_g']['lr'],
            "epochs": opt['train']['total_epoch'],
            "batch_size": opt['dataset']['train']['batch_size']
        }

    ## train the model
    for epoch in range(cur_epoch, opt['train']['total_epoch'] + 1):
        train_loss = train(train_loader, model, optimizer)
        scheduler.step()

        # validation
        if (epoch + 1) % opt["val"]["val_freq"] == 0:
            val_metric_loss = valid(val_loader, model)
            val_loss = val_metric_loss["loss"]
            # val_ic = val_metric_loss['ic']

            if opt['wandb']['use']:
                wandb.log(
                    {
                        'val_loss': val_metric_loss["loss"],
                        'IC': val_metric_loss["ic"],
                        'Rank_IC': val_metric_loss["rank_ic"],
                        'Precision@1': val_metric_loss["precision"]["1"],
                        'Precision@3': val_metric_loss["precision"]["3"],
                        'Precision@5': val_metric_loss["precision"]["5"],
                    }
                )

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'loss': best_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    os.path.join(save_dir, 'best_model.pth.tar'))


            logger.info("the current epoch is {}, the training loss is {:.4f}, the validation loss is {:.4f}".format((epoch + 1), train_loss, val_metric_loss["loss"]))
            logger.info("the IC is {:.6f}, the rank IC is {:.6f}, the precision@3 is {:.6f}".format(val_metric_loss["ic"], val_metric_loss["rank_ic"], val_metric_loss["precision"][3]))

    ## inference
    pretrain_path = os.path.join(save_dir, 'best_model.pth.tar')
    model_info = torch.load(pretrain_path)
    print('==> loading the best model:\n')
    model.load_state_dict(model_info['state_dict'])

    test_metric_loss = valid(test_loader, model)
    logger.info("the testing loss is {:.4f}".format(test_metric_loss["loss"]))
    logger.info("the IC is {}, the rank IC is {}, the precision@3 is {}".format(test_metric_loss["ic"], test_metric_loss["rank_ic"], test_metric_loss["precision"]['3']))

if __name__ == '__main__':
    main()