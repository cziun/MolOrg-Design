import json
import pickle
import argparse
import numpy as np
import networkx as nx

from vocab_generation import Tokenizer, MolInPiece
from utils.mol_utils import smiles2mol


def Check_max_motif_per_mol(fname, tokenizer):
    print(f'Loading mols from {fname} ...')
    smis = [smi.strip("\n") for smi in open(fname)]
    max = 0
    for s in smis:
        mol = smiles2mol(s, sanitize = True)
        mol = MolInPiece(mol)
        while True:
            nei_smis = mol.get_nei_smis()
            max_freq, merge_smi = -1, ''
            for smi in nei_smis:
                if smi not in tokenizer.vocab_dict:
                    continue
                freq = tokenizer.vocab_dict[smi][1]
                if freq > max_freq:
                    max_freq, merge_smi = freq, smi
            if max_freq == -1:
                break
            mol.merge(merge_smi)
        res = mol.get_smis_pieces()
        piece_idxs = [tokenizer.piece_to_idx(x[0]) + 1 for x in res]
        if len(piece_idxs) > max:
            max = len(piece_idxs)
    return max    


def preprocess(fname, vocab_path, max, tokenizer, arr_x_path, arr_adj_path, test_nx_path):
    print(f'Loading mols from {fname} ...')
    smis = [smi.strip("\n") for smi in open(fname)]
    x = np.zeros((len(smis), max)) 
    adj = np.zeros((len(smis), max, max))

    mapping = {}
    index = 1
    nx_graphs = []
    
    with open(vocab_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            s = tokens[0]
            mapping[index] = s
            index += 1

    for s in smis:
        mol = smiles2mol(s, sanitize = True)
        rdkit_mol = mol
        mol = MolInPiece(mol)
        G = nx.Graph()
        while True:
            nei_smis = mol.get_nei_smis()
            max_freq, merge_smi = -1, ''
            for smi in nei_smis:
                if smi not in tokenizer.vocab_dict:
                    continue
                freq = tokenizer.vocab_dict[smi][1]
                if freq > max_freq:
                    max_freq, merge_smi = freq, smi
            if max_freq == -1:
                break
            mol.merge(merge_smi)
        res = mol.get_smis_pieces()

        piece_idxs = [tokenizer.piece_to_idx(x[0]) + 1 for x in res] 
        for index, motif_id in enumerate(piece_idxs):
            x[i][index] = motif_id
            G.add_node(index, label = mapping[motif_id], order = motif_id) 

        aid2pid = {}
        for pid, piece in enumerate(res):
            _, aids = piece
            for aid in aids:
                aid2pid[aid] = pid
        for aid in range(rdkit_mol.GetNumAtoms()):
            atom = rdkit_mol.GetAtomWithIdx(aid)
            for nei in atom.GetNeighbors():
                nei_id = nei.GetIdx()
                m, n = aid2pid[aid], aid2pid[nei_id]
                if m != n:
                    adj[i][m][n] = adj[i][n][m] = 1
                    G.add_edge(m, n, label = 1)  
        nx_graphs.append(G)
        i += 1

    np.save(arr_x_path, x.astype(int))
    np.save(arr_adj_path, adj)

    with open(args.valid_idx_path) as f:
        test_idx = json.load(f)
    test_idx = [int(i) for i in test_idx]
    test_mols = [nx_graphs[i] for i in test_idx]
    with open(test_nx_path, 'wb') as f:
        pickle.dump(test_mols, f)



def parse():
    """parse command"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/hce/hce.smiles')
    parser.add_argument('--vocab_path', type=str, default='preprocess/hce/vocab.txt')
    parser.add_argument('--arr_x_path', type=str, default='preprocess/hce/arr_x.npy')
    parser.add_argument('--arr_adj_path', type=str, default='preprocess/hce/arr_adj.npy')
    parser.add_argument('--valid_idx_path', type=str, default='data/hce/valid_idx_hce.json')
    parser.add_argument('--test_nx_path', type=str, default='preprocess/hce/hce_test_nx.pkl')

    return parser.parse_args()



if __name__ == '__main__':
    args = parse()
    tokenizer = Tokenizer(args.vocab_path)
    max_motif_per_mol = Check_max_motif_per_mol(args.data, tokenizer) 
    print("Max num of motifs per mol: ", max_motif_per_mol)
    preprocess(args.data, args.vocab_path, max_motif_per_mol, tokenizer, args.arr_x_path, args.arr_adj_path, args.test_nx_path)
