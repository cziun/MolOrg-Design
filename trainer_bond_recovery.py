import numpy as np
import argparse
import random
import datetime
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

from config.config import get_config
from vocab_generation import Tokenizer
from bond.bpe_dataset import get_dataloader
from bond.BondRecovery import BondRecovery


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    seed_everything(seed=seed)


def train(model, train_loader, valid_loader, test_loader, config, args):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        dirpath=f"checkpoints/BondRecovery/{config.data.data}/{args.condition}/{current_time}",
        filename='best_model',
        mode='min', 
        save_top_k=1,
        verbose=True
    )
    logger = TensorBoardLogger("logs_bondRecovery/", name=f"{config.data.data}")
    trainer_config = {
        'logger': logger,
        'gpus': args.gpus,
        'max_epochs': args.epochs,
        'default_root_dir': f"checkpoints/BondRecovery/{config.data.data}/{args.condition}/{current_time}",
        'callbacks': checkpoint_callback,
        'gradient_clip_val': args.grad_clip
    }
    num_available_gpus = torch.cuda.device_count()
    if args.gpus == -1:
        if num_available_gpus > 1:
            trainer_config['accelerator'] = 'dp'
        else:
            trainer_config['gpus'] = 1
    trainer = pl.Trainer(**trainer_config)
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, dataloaders=test_loader)  


def parse():
    """parse command"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, required=True,
                        help="e.g. sample_hce")
    parser.add_argument('--vocab_path', type=str, default='preprocess/hce/vocab.txt')
    parser.add_argument('--train_set', type=str, default='data/hce/train_props.csv')
    parser.add_argument('--valid_set', type=str, default='data/hce/valid_props.csv')
    parser.add_argument('--test_set', type=str, default='data/hce/test_props.csv')
    parser.add_argument('--batch_size', type=int, default=32, help='size of mini-batch')
    parser.add_argument('--condition', type=str, default='')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=4, help='number of cpus to load data')
    parser.add_argument('--monitor', type=str, default='val_loss', help='Value to monitor in early stopping')
    parser.add_argument('--epochs', type=int, default=10, help='max epochs')
    parser.add_argument('--gpus', default=-1, help='gpus to use') 
    parser.add_argument('--grad_clip', type=float, default=10.0,
                        help='clip large gradient to prevent gradient boom')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    setup_seed(args.seed)
    config = get_config(args.config, args.seed)
    tokenizer = Tokenizer(args.vocab_path)
    # Prepare data
    train_loader = get_dataloader(args.train_set, tokenizer, batch_size=args.batch_size,
                                              shuffle=args.shuffle, num_workers=args.num_workers)
    valid_loader = get_dataloader(args.valid_set, tokenizer, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)
    test_loader = get_dataloader(args.test_set, tokenizer, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)
    # Using pytorch_lightning
    model = BondRecovery(config, tokenizer, args.condition)
    # Train
    train(model, train_loader, valid_loader, test_loader, config, args)
    