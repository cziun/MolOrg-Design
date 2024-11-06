import time
import torch
import pytorch_lightning as pl

from bond.generator import Generator


class BondRecovery(pl.LightningModule):
    def __init__(self, config, tokenizer, condition):
        super(BondRecovery, self).__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer
        if condition == 'singlet-triplet value':
            self.cond_index = 0
            self.mean = 1
            self.std = 0.4
        elif condition == 'oscillator strength':
            self.cond_index = 1
            self.mean = 0.09
            self.std = 0.15
        elif condition == 'multi-objective value':
            self.cond_index = 2
            self.mean = -1.6
            self.std = 0.65
        elif condition == 'pce_pcbm_sas':
            self.cond_index = 0
            self.mean = -0.3
            self.std = 2.38
        elif condition == 'pce_pcdtbt_sas':
            self.cond_index = 1
            self.mean = -2.98
            self.std = 4.18
        elif condition == 'pce12_sas':
            self.cond_index = 5
            self.mean = 0.55
            self.std = 4.77
        elif condition == 'sas':
            self.cond_index = 4
            self.mean = 3.84
            self.std = 0.58
        elif condition == 'pce_1':
            self.cond_index = 2
            self.mean = 3.54
            self.std = 2.33
        elif condition == 'pce_2':
            self.cond_index = 3
            self.mean = 0.86
            self.std = 4.15
        elif condition == 'Ea':
            self.cond_index = 1
            self.mean = 84.1
            self.std = 3.08
        elif condition == 'Er':
            self.cond_index = 2
            self.mean = -0.74
            self.std = 4.51
        elif condition == 'sum_Ea_Er':
            self.cond_index = 3
            self.mean = 83.36
            self.std = 7.04
        elif condition == 'diff_Ea_Er':
            self.cond_index = 4
            self.mean = -84.85
            self.std = 3.16
        elif condition == 'sas_react':
            self.cond_index = 0
            self.mean = 6.56
            self.std = 0.39
        elif condition == '4lde score':
            self.cond_index = 1
            self.mean = -7.61
            self.std = 1.51
        else:
            print('Wrong condition input')
        self.atom_dim = config.generate.atom_embedding_dim + \
                        config.generate.piece_embedding_dim + \
                        config.generate.pos_embedding_dim
        self.decoder = Generator(tokenizer, config.generate.atom_embedding_dim, config.generate.piece_embedding_dim,
                                 config.generate.max_pos, config.generate.pos_embedding_dim,
                                 config.generate.num_edge_type, config.generate.node_hidden_dim, config.generate.property_embedding_dim)
        self.total_time = 0

    def forward(self, batch, return_accu=False):
        x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr'] 
        x_pieces, x_pos = batch['x_pieces'], batch['x_pos'] 
        x = self.decoder.embed_atom(x, x_pieces, x_pos)
        batch_size, node_num, node_dim = x.shape
        in_piece_edge_idx = batch['in_piece_edge_idx']
        # adding conditin
        props = batch['props'][:,self.cond_index]
        props = (props - self.mean) / self.std
        prop = props.unsqueeze(1)
        res = self.decoder(x=x, edge_index=edge_index[:, in_piece_edge_idx],  # do not include the edges to be predicted
                              edge_attr=edge_attr[in_piece_edge_idx],  # do not include the edges to be predicted
                              pieces=batch['pieces'], 
                              edge_select=batch['edge_select'],
                              golden_edge=batch['golden_edge'],  # only include edges to be predicted
                              props=prop,
                              return_accu=return_accu)
        return res

    def training_step(self, batch, batch_idx):
        st = time.time()
        loss = self.forward(batch)
        self.log('train_loss', loss)
        self.total_time += time.time() - st
        self.log('total time', self.total_time)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accu = self.forward(batch, return_accu=True)
        self.log('val_loss', loss)
        self.log('val_accu', accu)

    def test_step(self, batch, batch_idx):
        loss, accu = self.forward(batch, return_accu=True)
        self.log('test_loss', loss)
        self.log('test_accu', accu)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.generate.lr)
        return optimizer

    # interface
    def inference_single_z(self, motif_ids, adj_inter_motif, add_edge_th, con, device):
        return self.decoder.inference(motif_ids, adj_inter_motif, add_edge_th, con, device)
