import json
import pandas as pd
import networkx as nx
from copy import copy
from queue import Queue
import rdkit.Chem as Chem
from rdkit import Chem
from rdkit.Chem.rdchem import BondType


# hce   
MAX_VALENCE = {'C':4, 'N':3, 'O':2, 'S':6, 'Se': 2, 'Si':4}
# gdb13 
# MAX_VALENCE = {'C':4, 'N':3, 'O':2, 'S':6}
# reactivity
# MAX_VALENCE = {'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':3, 'O':2, 'S':6}
# dtp
# MAX_VALENCE = {'Ga': 3, 'Sb': 5, 'F': 1, 'Bi': 5, 'In': 3, 'Se': 2, 'P': 5, 'B': 3, 'Ge': 4, 'Br': 1, 'N': 3, 'S': 6, 'Cl': 1, 'Tl': 3, 'O': 2, 'Te': 6, 'Hg': 2, 'As': 5, 'C': 4, 'Pb': 4}

Bond_List = [None, BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]


def smiles2mol(smiles: str, sanitize: bool=False) -> Chem.rdchem.Mol:
    mol = Chem.MolFromSmiles(smiles, sanitize)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    Chem.Kekulize(mol)
    return mol


def mol2smiles(mol):
    return Chem.MolToSmiles(mol)


def get_submol(mol, idx2atom):
    sub_mol = Chem.RWMol()
    oid2nid = {}  # old id to new id
    for nid, oid in enumerate(idx2atom.keys()):
        atom = mol.GetAtomWithIdx(oid)
        new_atom = Chem.Atom(atom.GetSymbol())
        sub_mol.AddAtom(new_atom)
        oid2nid[oid] = nid
    for aid in idx2atom:
        atom = mol.GetAtomWithIdx(aid)
        for bond in atom.GetBonds():
            nei_id = bond.GetBeginAtomIdx()
            if nei_id == aid:
                nei_id = bond.GetEndAtomIdx()
            if nei_id in idx2atom and nei_id < aid:
                sub_mol.AddBond(oid2nid[aid], oid2nid[nei_id], bond.GetBondType())

    sub_mol = sub_mol.GetMol()
    return sub_mol


def cnt_atom(smi):
    cnt = 0
    for c in smi:
        if c in MAX_VALENCE:
            cnt += 1
    return cnt


class GeneralVocab:
    def __init__(self, atom_special=None, bond_special=None):
        # atom
        self.idx2atom = list(MAX_VALENCE.keys())
        if atom_special is None:
            atom_special = []
        self.idx2atom += atom_special
        self.atom2idx = {atom: i for i, atom in enumerate(self.idx2atom)}
        # bond
        self.idx2bond = copy(Bond_List)
        if bond_special is None:
            bond_special = []
        self.idx2bond += bond_special
        self.bond2idx = {bond: i for i, bond in enumerate(self.idx2bond)}

        self.atom_special = atom_special
        self.bond_special = bond_special

    def idx_to_atom(self, idx):
        return self.idx2atom[idx]

    def atom_to_idx(self, atom):
        return self.atom2idx[atom]

    def idx_to_bond(self, idx):
        return self.idx2bond[idx]

    def bond_to_idx(self, bond):
        return self.bond2idx[bond]

    def bond_idx_to_valence(self, idx):
        bond_enum = self.idx2bond[idx]
        if bond_enum == BondType.SINGLE:
            return 1
        elif bond_enum == BondType.DOUBLE:
            return 2
        elif bond_enum == BondType.TRIPLE:
            return 3
        else:  # invalid bond
            return -1

    def num_atom_type(self):
        return len(self.idx2atom)

    def num_bond_type(self):
        return len(self.idx2bond)


def load_smiles(dataset='HCE'):
    if dataset == 'HCE':
        col = 'smiles'
    elif dataset == 'GDB13':
        col = 'smiles'
    elif dataset == 'REACTIVITY':
        col = 'smiles'
    elif dataset == 'DTP':
        col = 'smiles'
    else:
        raise ValueError('wrong dataset name in load_smiles')

    df = pd.read_csv(f'data/{dataset.lower()}/{dataset.lower()}.csv')

    with open(f'data/{dataset.lower()}/valid_idx_{dataset.lower()}.json') as f:
        test_idx = json.load(f)

    train_idx = [i for i in range(len(df)) if i not in test_idx]

    return list(df[col].loc[train_idx]), list(df[col].loc[test_idx])


def canonicalize_smiles(smiles):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]


def valence_check(aid1, aid2, edges1, edges2, new_edge, vocab, c1=0, c2=0):
    new_valence = vocab.bond_idx_to_valence(new_edge)
    if new_valence == -1:
        return False
    atom1 = vocab.idx_to_atom(aid1)
    atom2 = vocab.idx_to_atom(aid2)
    a1_val = sum(list(map(vocab.bond_idx_to_valence, edges1)))
    a2_val = sum(list(map(vocab.bond_idx_to_valence, edges2)))
    # special for S as S is likely to have either 2 or 6 valence
    if (atom1 == 'S' and a1_val == 2) or (atom2 == 'S' and a2_val == 2):
        return False
    return a1_val + new_valence + abs(c1) <= MAX_VALENCE[atom1] and \
           a2_val + new_valence + abs(c2) <= MAX_VALENCE[atom2]


def cycle_check(i, j, mol):
    cycle_len = shortest_path_len(i, j, mol)
    return cycle_len is None or (cycle_len > 4 and cycle_len < 7)


def shortest_path_len(i, j, mol):
    queue = Queue()
    queue.put((mol.GetAtomWithIdx(i), 1))
    visited = {}
    visited[i] = True
    while not queue.empty():
        atom, dist = queue.get()
        neis = []
        for nei in atom.GetNeighbors():
            idx = nei.GetIdx()
            if idx == j:
                return dist + 1
            if idx not in visited:
                visited[idx] = True
                neis.append(idx)
                queue.put((mol.GetAtomWithIdx(idx), dist + 1))
    return None


def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=atom.GetSymbol())
                    #    atomic_num=atom.GetAtomicNum(),
                    #    formal_charge=atom.GetFormalCharge(),
                    #    chiral_tag=atom.GetChiralTag(),
                    #    hybridization=atom.GetHybridization(),
                    #    num_explicit_hs=atom.GetNumExplicitHs(),
                    #    is_aromatic=atom.GetIsAromatic())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))
                    #    bond_type=bond.GetBondType())
        nx_graphs.append(G)
    return nx_graphs
