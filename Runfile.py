import json
import random
import argparse
import numpy as np
import torch
import warnings
from torch_geometric.loader import DataLoader
from training import Training
from model import BA
import os
from dataset import GNNDataset


os.environ["CUDA_VISIBLE_DEVICE"] = '0'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True,
                        help=f'Args file < JSON >\n')
    arg = parser.parse_args()
    return arg


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def main():
    setup_seed(64)
    warnings.filterwarnings('ignore')
    # args
    args_file = args_parse()
    with open(args_file.file, 'r') as f:
        args = json.load(f)

    args_batch_size = args['dataset']['batch_size']
    args_layers = args['model']['num_layers']
    args_epochs = args['train']['epochs']
    args_device = args['train']['device']
    args_lr = args['train']['lr']
    args_prefix = args['output']['prefix']
    args_model_dir = args['output']['model_dir']
    args_logs = args['output']['tensorboard_logs']

    PP_train_dataset = GNNDataset('features/pkls/edge_thr_8',
                                  phase='train')
    PP_val_dataset = GNNDataset('features/pkls/edge_thr_8',
                                phase='val')

    train_loader = DataLoader(PP_train_dataset,
                              batch_size=args_batch_size,
                              num_workers=6,
                              shuffle=True,
                              follow_batch=['edge_features_s', 'edge_features_t'])
    val_loader = DataLoader(PP_val_dataset,
                            batch_size=args_batch_size,
                            num_workers=6,
                            shuffle=True,
                            follow_batch=['edge_features_s', 'edge_features_t'])

    # set running device
    if args_device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # construct the model
    model = BA(device=device,
               layers=args_layers)

    Trainer = Training(model,
                       train_dataloader=train_loader,
                       val_dataloader=val_loader,
                       prefix=args_prefix,
                       device=device,
                       lr=args_lr,
                       model_path=args_model_dir,
                       log_dir=args_logs)
    Trainer.train(epochs=args_epochs)


if __name__ == '__main__':
    main()


# usage:
# python Runfile.py -f running_args.json
