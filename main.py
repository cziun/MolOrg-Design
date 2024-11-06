import argparse
import time

from config.config import get_config
from trainer import Trainer
from sampler import Sampler_mol


def main(args):
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    config = get_config(args.config, args.seed)

    # -------- Train --------
    if args.type == 'train':
        trainer = Trainer(config, args)
        ckpt = trainer.train(ts)

    # -------- Generation --------
    elif args.type == 'sample':
        sampler = Sampler_mol(config)
        sampler.sample(args)

    else:
        raise ValueError(f'Wrong type : {args.type}')


def parse():
    """parse command"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True,
                        help="train or sample")
    parser.add_argument('--gpus', type=int, default=0,
                        help="cpu: -1")
    parser.add_argument('--config', type=str, required=True,
                        help="HCE: hce or sample_hce")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt_train_path', type=str, default='',
                        help=" The full path: ./checkpoints/{config.data.data}/condition/{--ckpt_train_path}.pth ")
    parser.add_argument('--ckpt_bond_path', type=str, default='',
                        help=" The full path: ./checkpoints/BondRecovery/{config.data.data}/{--ckpt_bond_path}/best_model.ckpt ")
    parser.add_argument('--vocab', type=str, default="preprocess/hce/vocab.txt")
    parser.add_argument('--output', type=str, default="output_hce",
                        help=" The full path: output/{--output}.txt ")
    parser.add_argument('--condition', type=str, default="pce_pcbm_sas")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    main(args)
