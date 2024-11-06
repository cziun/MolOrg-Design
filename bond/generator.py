import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from bond.encoder import Encoder, MLP
from utils.mol_utils import smiles2mol, valence_check, cycle_check


class Generator(nn.Module):
    def __init__(self, tokenizer, atom_embedding_dim, piece_embedding_dim, max_pos,
                 pos_embedding_dim, num_edge_type, node_hidden_dim, property_embedding_dim, t=4):
        super(Generator, self).__init__()
        self.tokenizer = tokenizer
        self.atom_embedding = nn.Embedding(tokenizer.num_atom_type(), atom_embedding_dim)  
        self.piece_embedding = nn.Embedding(tokenizer.num_piece_type(), piece_embedding_dim)  
        self.pos_embedding = nn.Embedding(max_pos, pos_embedding_dim) 

        self.property_embedding = nn.Linear(1, property_embedding_dim)
        self.addition = nn.Linear(node_hidden_dim+property_embedding_dim, node_hidden_dim)

        node_dim = atom_embedding_dim + piece_embedding_dim + pos_embedding_dim
        # graph embedding
        self.graph_embedding = Encoder(node_dim, num_edge_type, node_hidden_dim, dim_out=1, t=t)
        mlp_in = node_hidden_dim * 2
        # edge link predictor
        self.edge_predictor = nn.Sequential(
            MLP(
                dim_in=mlp_in,
                dim_hidden=mlp_in // 2,
                dim_out=mlp_in,
                act_func=nn.ReLU,
                num_layers=3
            ),
            nn.Linear(mlp_in, num_edge_type)
        )
        # loss
        self.edge_loss = nn.CrossEntropyLoss()
    
    def forward(self, x, edge_index, edge_attr, pieces, edge_select, golden_edge, props, return_accu=False):

        # graph embedding
        batch_size, node_num, node_dim = x.shape
        node_x = x.view(-1, node_dim)

        prop = self.property_embedding(props)
        prop1 = prop.unsqueeze(1) 
        propp = prop1.repeat(1, node_num, 1) 

        node_embedding, _ = self.graph_embedding.embed_node(node_x, edge_index, edge_attr)  # [batch_size * N, node_dim] 
        node_embedding = node_embedding.view(batch_size, node_num, -1)  # [batch_size, N, node_dim] 
        # adding condition
        node_embedding1 = torch.cat((node_embedding, propp), dim=2) 
        node_embeddingg = self.addition(node_embedding1) 

        # edge prediction
        src_embedding = torch.repeat_interleave(node_embeddingg, node_num, dim=1).view(batch_size, node_num, node_num, -1)
        src_embedding = src_embedding[edge_select]  # to [E, node_dim]
        dst_embedding = torch.repeat_interleave(node_embeddingg, node_num, dim=0).view(batch_size, node_num, node_num, -1)
        dst_embedding = dst_embedding[edge_select]
        edge_pred_in = torch.cat([src_embedding, dst_embedding], dim=-1)
        pred_edge = self.edge_predictor(edge_pred_in)  # [E, num_edge_type]   

        # loss
        edge_loss = self.edge_loss(pred_edge, golden_edge)

        # accu
        if return_accu:
            edge_accu = (torch.argmax(pred_edge, dim=-1) == golden_edge).sum().item() / len(golden_edge)
            return edge_loss, edge_accu

        return edge_loss

    def inference(self, motif_ids, adj_inter_motif, add_edge_th, con, device):

        cond = torch.full((1, 1), con, dtype=torch.float32).to(device)
        cond1 = self.property_embedding(cond).unsqueeze(1)  # torch.Size([1, 1, 100])

        piece_ids = list(motif_ids)
        print(piece_ids)
        x, edge_index, edge_attr, groups = [], [], [], []
        aid2gid = {}  # map atom idx to group idx
        aid2bid = {}  # map atom idx to connected block (bid)
        block_atom_cnt = []
        gen_mol = Chem.RWMol() # generated mol
        edge_sets = []  # record each atom is connected to which kinds of bonds
        x_pieces, x_pos = [], []
        if all(value == -1 for value in piece_ids):
            return gen_mol.GetMol()
        for pos, pid in enumerate(piece_ids):
            if pid == -1 and gen_mol.GetNumAtoms() != 0:
                break
            if pid == -1 and gen_mol.GetNumAtoms() == 0:
                continue
            smi = self.tokenizer.idx_to_piece(pid)
            mol = smiles2mol(smi, sanitize = True)
            if mol == None:
                return Chem.RWMol().GetMol()
            offset = len(x) 
            group, atom_num = [], mol.GetNumAtoms()
            for aid in range(atom_num): 
                atom = mol.GetAtomWithIdx(aid)
                group.append(len(x))
                aid2gid[len(x)], aid2bid[len(x)] = len(groups), len(groups)
                x.append(self.tokenizer.chem_vocab.atom_to_idx(atom.GetSymbol()))
                edge_sets.append([])
                x_pieces.append(pid)
                x_pos.append(pos + 1) 
                gen_mol.AddAtom(Chem.Atom(atom.GetSymbol()))  # add atom to generated mol
            groups.append(group)
            block_atom_cnt.append(atom_num)
            for bond in mol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                attr = self.tokenizer.chem_vocab.bond_to_idx(bond.GetBondType())
                begin, end = begin + offset, end + offset
                edge_index.append([begin, end])
                edge_index.append([end, begin])
                edge_attr.append(attr)
                edge_attr.append(attr)
                edge_sets[begin].append(attr)
                edge_sets[end].append(attr)
                gen_mol.AddBond(begin, end, bond.GetBondType())  # add bond to generated mol

        atoms, edges, edge_types = x, edge_index, edge_attr
        node_x = self.embed_atom(torch.tensor(x, dtype=torch.long, device=device),
                                 torch.tensor(x_pieces, dtype=torch.long, device=device),
                                 torch.tensor(x_pos, dtype=torch.long, device=device))
        if len(edge_index) == 0:
            edge_index = torch.randn(2, 0, device=device).long()
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous() 
        edge_attr = F.one_hot(torch.tensor(edge_attr, dtype=torch.long, device=device),
                              num_classes=self.tokenizer.chem_vocab.num_bond_type()) 
        node_embedding, _ = self.graph_embedding.embed_node(node_x, edge_index,
                                                            edge_attr)  # [n, node_embeding_dim] 
        # construct edge select mat, only care about up triangle
        node_num = len(x)
        edge_select = torch.triu(torch.ones(node_num, node_num, dtype=torch.long, device=device)) 
        for group in groups:
            group_len = len(group)
            for i in range(group_len):
                for j in range(i, group_len):
                    m, n = group[i], group[j]
                    edge_select[m][n] = edge_select[n][m] = 0 

        for i in range(len(edge_select)):
            for j in range(len(edge_select[i])):
                if edge_select[i][j] > 0:
                    i_motif, j_motif = x_pos[i], x_pos[j]
                    if adj_inter_motif[i_motif-1][j_motif-1] == 0:
                        edge_select[i][j] = 0
        
        edge_select = edge_select.unsqueeze(0).bool()  # [1, node_num, node_num]
        # predict edge
        node_embedding = node_embedding.unsqueeze(0)  # [1, n, embedding_dim] 

        condd = cond1.repeat(1, node_embedding.size(1), 1) 
        # adding condition
        node_embedding1 = torch.cat((node_embedding, condd), dim=2) 
        node_embeddingg = self.addition(node_embedding1) 

        src_embedding = torch.repeat_interleave(node_embeddingg, node_num, dim=1).view(1, node_num, node_num, -1)
        src_embedding = src_embedding[edge_select]  
        dst_embedding = torch.repeat_interleave(node_embeddingg, node_num, dim=0).view(1, node_num, node_num, -1)
        dst_embedding = dst_embedding[edge_select]  
        edge_pred_in = torch.cat([src_embedding, dst_embedding], dim=-1)
        if edge_pred_in.shape[0]:  # maybe only one piece is generated -> no edges need to be predicted

            pred_edge = self.edge_predictor(edge_pred_in)

            pred_edge = torch.softmax(pred_edge, dim=-1)
            # add edge to mol by confidence level
            pred_edge_index = torch.nonzero(edge_select.squeeze())  # [E, 2] 
            none_bond = self.tokenizer.chem_vocab.bond_to_idx(None)  # 0
            confidence, edge_type = torch.max(pred_edge, dim=-1)  # [E], [E]
            possible_edge_idx = [i for i in range(len(pred_edge))
                                 if confidence[i] >= add_edge_th and edge_type[i] != none_bond]
            sorted_idx = sorted(possible_edge_idx, key=lambda i: confidence[i], reverse=True)
            for i in sorted_idx:
                begin, end = pred_edge_index[i]
                begin, end = begin.item(), end.item()
                bond_type = edge_type[i]
                # the cycle check is very important (only generate cycles with 5 or 6 atoms)
                if valence_check(atoms[begin], atoms[end], edge_sets[begin],
                                 edge_sets[end], bond_type, self.tokenizer.chem_vocab) and \
                        cycle_check(begin, end, gen_mol):
                    gen_mol.AddBond(begin, end, self.tokenizer.chem_vocab.idx_to_bond(bond_type))
                    edge_sets[begin].append(bond_type)
                    edge_sets[end].append(bond_type)
                    # update connected block
                    bid1, bid2 = aid2bid[begin], aid2bid[end]
                    if bid1 != bid2:
                        gid = aid2gid[begin]
                        for aid in aid2bid:  # redirect all atom in block1 to block2
                            if aid2bid[aid] == bid1:
                                aid2bid[aid] = bid2
                        block_atom_cnt[bid2] += block_atom_cnt[bid1]
        # delete isolated parts
        # find connect block with max atom num
        bid = max(range(len(block_atom_cnt)), key=lambda i: block_atom_cnt[i])
        atoms_to_remove = sorted([aid for aid in aid2bid.keys() if aid2bid[aid] != bid], reverse=True)
        for aid in atoms_to_remove:
            gen_mol.RemoveAtom(aid)
        gen_mol = gen_mol.GetMol()
        Chem.SanitizeMol(gen_mol)
        Chem.Kekulize(gen_mol)
        return gen_mol

    def embed_atom(self, atom_ids, piece_ids, pos_ids):
        atom_embed = self.atom_embedding(atom_ids)
        piece_embed = self.piece_embedding(piece_ids)
        pos_embed = self.pos_embedding(pos_ids)
        return torch.cat([atom_embed, piece_embed, pos_embed], dim=-1)
    