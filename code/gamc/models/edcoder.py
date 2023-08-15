from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gat import GAT
from .gin import GIN, VGIN
from .loss_func import sce_loss
from gamc.utils import create_norm
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, remove_self_loops
from random import sample


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            out_dim=int(out_dim),
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == 'vgin':
        mod = VGIN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            out_dim=int(out_dim),
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise_without_new(self, x, news_node, mask_rate=0.3):
        num_nodes = x.shape[0]
        i = 0
        news_id_list = []
        news_node = news_node.reshape(-1)
        for id in news_node:
            if id.item() == 1:
                news_id_list.append(i)
            i = i + 1

        # perm = torch.randperm(num_nodes, device=x.device) # 返回一个随机打散的数组
        perm = torch.randperm(num_nodes).numpy().tolist()  # 返回一个随机打散的数组
        perm = [item for item in perm if item not in news_id_list]


        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = torch.tensor(perm[: num_mask_nodes], device=x.device)
        keep_nodes = torch.tensor(perm[num_mask_nodes: ] + news_id_list, device=x.device)


        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def encoding_mask_noise(self, x, news_node, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def forward(self, x, edge_index, news_node):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction_with_contrastive(x, edge_index, news_node)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    def mask_attr_prediction(self, x, edge_index, news_node):
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, news_node, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_edge_index, masked_edges = dropout_edge(edge_index, self._drop_edge_rate)
            # use_edge_index = add_self_loops(use_edge_index)[0]
        else:
            use_edge_index = edge_index

        enc_rep, all_hidden = self.encoder(use_x, use_edge_index, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(rep, use_edge_index)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        kl_divergence = 0.5 / x_rec.size(0) * (1 + 2 * all_hidden[-2][mask_nodes] - all_hidden[-3][mask_nodes] ** 2 - torch.exp(all_hidden[-2][mask_nodes])).sum(
            1).mean()
        loss = self.criterion(x_rec, x_init) - kl_divergence
        return loss

    def mask_attr_prediction_with_contrastive(self, x, edge_index, news_node):
        use_x_1, (mask_nodes_1, keep_nodes_1) = self.encoding_mask_noise(x, news_node, self._mask_rate)
        use_x_2, (mask_nodes_2, keep_nodes_2) = self.encoding_mask_noise(x, news_node, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_edge_index, masked_edges = dropout_edge(edge_index, self._drop_edge_rate)
            # use_edge_index = add_self_loops(use_edge_index)[0]
        else:
            use_edge_index = edge_index

        enc_rep_1, all_hidden_1 = self.encoder(use_x_1, use_edge_index, return_hidden=True)
        enc_rep_2, all_hidden_2 = self.encoder(use_x_2, use_edge_index, return_hidden=True)

        if self._concat_hidden:
            enc_rep_1 = torch.cat(all_hidden_1, dim=1)
            enc_rep_2 = torch.cat(all_hidden_1, dim=1)

        # ---- attribute reconstruction ----
        rep_1 = self.encoder_to_decoder(enc_rep_1)
        rep_2 = self.encoder_to_decoder(enc_rep_2)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep_1[mask_nodes_1] = 0
            rep_2[mask_nodes_1] = 0

        if self._decoder_type in ("mlp", "linear"):
            recon_1 = self.decoder(rep_1)
            recon_2 = self.decoder(rep_2)
        else:
            recon_1 = self.decoder(rep_1, use_edge_index)
            recon_2 = self.decoder(rep_2, use_edge_index)

        x_init_1 = x[mask_nodes_1]
        x_init_2 = x[mask_nodes_2]
        x_rec_1 = recon_1[mask_nodes_1]
        x_rec_2 = recon_2[mask_nodes_2]

        num_graph = torch.sum(news_node).int()

        # kl_divergence = 0.5 / x_rec_1.size(0) * (
        #             1 + 2 * all_hidden_1[-2][mask_nodes_1] - all_hidden_1[-3][mask_nodes_1] ** 2 - torch.exp(
        #         all_hidden_1[-2][mask_nodes_1])).sum(
        #     1).mean()
        rec_1 = global_mean_pool(recon_1, None)[0]
        rec_2 = global_mean_pool(recon_2, None)[0]
        loss_c = torch.cosine_similarity(rec_1, rec_2, dim=0)
        # loss_c = torch.mean(loss_c)
        loss = self.criterion(x_rec_1, x_init_1) + self.criterion(x_rec_2, x_init_2)
        loss = - loss_c
        return loss

    def embed(self, x, edge_index):
        rep = self.encoder(x, edge_index)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
