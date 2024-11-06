import os
from time import time
import numpy as np
import networkx as nx
import torch
from torch.utils.data import DataLoader, Dataset
import json


def dataloader(config, condition, get_graph_list=False):
    start_time = time()

    mols = []
    arr_x = np.load(os.path.join('preprocess', config.data.data.lower(), 'arr_x.npy'))
    arr_adj = np.load(os.path.join('preprocess', config.data.data.lower(), 'arr_adj.npy'))
    if condition == 'pce_pcbm_sas':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'pce_pcbm_sas.npy'))
        mean = -0.3
        std = 2.38
    elif condition == 'pce_pcdtbt_sas':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'pce_pcdtbt_sas.npy'))
        mean = -2.98
        std = 4.18
    elif condition == 'pce12_sas':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'pce12_sas.npy'))
        mean = 0.55
        std = 4.77
    elif condition == 'sas':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'sas.npy'))
        mean = 3.84
        std = 0.58
    elif condition == 'pce_1':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'pce_1.npy'))
        mean = 3.54
        std = 2.33
    elif condition == 'pce_2':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'pce_2.npy'))
        mean = 0.86
        std = 4.15
    elif condition == 'singlet-triplet value':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'singlet-triplet value.npy'))
        mean = 1
        std = 0.4
    elif condition == 'oscillator strength':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'oscillator strength.npy'))
        mean = 0.09
        std = 0.15
    elif condition == 'multi-objective value':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'multi-objective value.npy'))
        mean = -1.6
        std = 0.65
    elif condition == 'Ea':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'Ea.npy'))
        mean = 84.1
        std = 3.08
    elif condition == 'Er':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'Er.npy'))
        mean = -0.74
        std = 4.51
    elif condition == 'sum_Ea_Er':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'sum_Ea_Er.npy'))
        mean = 83.36
        std = 7.04
    elif condition == 'diff_Ea_Er':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'diff_Ea_Er.npy'))
        mean = -84.85
        std = 3.16
    elif condition == 'sas_react':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), 'sas_react.npy'))
        mean = 6.56
        std = 0.39
    elif condition == '4lde score':
        arr_con = np.load(os.path.join('preprocess', config.data.data.lower(), '4lde score.npy'))
        mean = -7.61
        std = 1.51
    else:
        print("Wrong condition input")

    # Normalization
    arr_con = (arr_con - mean) / std

    for i in range(0, len(arr_x)):
        tmp = (arr_x[i], arr_adj[i], arr_con[i])
        mols.append(tmp)

    with open(os.path.join(config.data.dir, config.data.data.lower(), f'valid_idx_{config.data.data.lower()}.json')) as f:
        test_idx = json.load(f)

    train_idx = [i for i in range(len(mols)) if i not in test_idx]
    print(f'Number of training mols: {len(train_idx)} | Number of test mols: {len(test_idx)}')

    train_mols = [mols[i] for i in train_idx]
    test_mols = [mols[i] for i in test_idx]

    train_dataset = MolDataset(train_mols, get_transform_fn(config.data.data, config.data.max_node_num, config.data.max_feat_num))
    test_dataset = MolDataset(test_mols, get_transform_fn(config.data.data, config.data.max_node_num, config.data.max_feat_num))

    if get_graph_list:
        train_mols_nx = [nx.from_numpy_matrix(np.array(adj)) for x, adj, con in train_dataset]
        test_mols_nx = [nx.from_numpy_matrix(np.array(adj)) for x, adj, con in test_dataset]
        return train_mols_nx, test_mols_nx
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=True)

    print(f'{time() - start_time:.2f} sec elapsed for data loading')
    return train_dataloader, test_dataloader


def get_transform_fn(dataset, max_node_num, max_feat_num):
    def transform(data):
            x, adj, con = data
            x_ = np.zeros((max_node_num, max_feat_num + 1))
            indices = np.where(x >= 1, x - 1, max_feat_num)
            x_[np.arange(max_node_num), indices] = 1
            x = torch.tensor(x_).to(torch.float32)
            x = x[:, :-1]
            adj = torch.tensor(adj).to(torch.float32)
            con = torch.tensor(con).to(torch.float32)
            return x, adj, con
    return transform


class MolDataset(Dataset):
    def __init__(self, mols, transform):
        self.mols = mols
        self.transform = transform

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        return self.transform(self.mols[idx])
