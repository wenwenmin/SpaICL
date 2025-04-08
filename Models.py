import copy
import numpy as np
import random
import torch.nn.functional as F
import torch
from torch import nn
import scipy
import ot
from torch_geometric.nn import (
    TransformerConv,
    LayerNorm,
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)

from utils.attention import CrossAttention, MultiHeadAttention

import faiss

def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def full_block(in_features, out_features, p_drop, act=nn.ELU()):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        act,  # nn.ELU(),
        nn.Dropout(p=p_drop),
    )

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, act=F.relu, bn=True, graphtype="gcn"):
        super(GraphConv, self).__init__()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn(out_features)
        self.act = act
        self.dropout = dropout
        if graphtype == "gcn":
            self.conv = GCNConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "gat": # Default heads=1
            self.conv = GATConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "gin": # Default heads=1
            self.conv = TransformerConv(in_channels=self.in_features, out_channels=self.out_features)
        else:
            raise NotImplementedError(f"{graphtype} is not implemented.")

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = config['feat_hidden1']
        self.feat_hidden2 = config['feat_hidden2']
        self.gcn_hidden = config['gcn_hidden']
        self.latent_dim = config['latent_dim']

        self.p_drop = config['p_drop']
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))
        # GCN layers
        self.gc1 = GraphConv(self.feat_hidden2, self.gcn_hidden, dropout=self.p_drop, act=F.relu)
        self.gc2 = GraphConv(self.gcn_hidden, self.latent_dim, dropout=self.p_drop, act=lambda x: x)

    def forward(self, x, edge_index):
        x = self.encoder(x)
        x = self.gc1(x, edge_index)
        x = self.gc2(x, edge_index)
        return x

class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['latent_dim']
        self.gcn_hidden = config['project_dim']
        self.p_drop = config['p_drop']
        self.layer1 = GraphConv(self.input_dim, self.gcn_hidden, dropout=self.p_drop, act=F.relu)
        self.layer2 = nn.Linear(self.gcn_hidden, self.input_dim, bias=False)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim, config):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = config['latent_dim']
        self.p_drop = config['p_drop']
        self.layer1 = GraphConv(self.input_dim, self.output_dim, dropout=self.p_drop, act=nn.Identity())

    def forward(self, x, edge_index):
        return self.layer1(x, edge_index)


class Neighbor(nn.Module):
    def __init__(self, device, num_centroids, num_kmeans, clus_num_iters):
        super(Neighbor, self).__init__()
        self.device = device
        self.num_centroids = num_centroids
        self.num_kmeans = num_kmeans
        self.clus_num_iters = clus_num_iters

    def __get_close_nei_in_back(self, indices, each_k_idx, cluster_labels, back_nei_idxs, k):
        # get which neighbors are close in the background set
        batch_labels = cluster_labels[each_k_idx][indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = repeat_1d_tensor(batch_labels, k)

        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei

    def forward(self, adj, student, teacher, top_k):
        n_data, d = student.shape
        similarity = torch.matmul(student, torch.transpose(teacher, 1, 0).detach())
        similarity += torch.eye(n_data, device=self.device) * 10

        _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)
        tmp = torch.LongTensor(np.arange(n_data)).unsqueeze(-1).to(self.device)

        knn_neighbor = self.create_sparse(I_knn)
        locality = knn_neighbor * adj

        ncentroids = self.num_centroids
        niter = self.clus_num_iters

        pred_labels = []

        for seed in range(self.num_kmeans):
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, gpu=False, seed=seed + 1234)
            kmeans.train(teacher.cpu().numpy())
            _, I_kmeans = kmeans.index.search(teacher.cpu().numpy(), 1)

            clust_labels = I_kmeans[:, 0]

            pred_labels.append(clust_labels)

        pred_labels = np.stack(pred_labels, axis=0)
        cluster_labels = torch.from_numpy(pred_labels).long().to(self.device)

        all_close_nei_in_back = None
        with torch.no_grad():
            for each_k_idx in range(self.num_kmeans):
                curr_close_nei = self.__get_close_nei_in_back(tmp.squeeze(-1), each_k_idx, cluster_labels, I_knn, I_knn.shape[1])

                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

        all_close_nei_in_back = all_close_nei_in_back.to(self.device)

        globality = self.create_sparse_revised(I_knn, all_close_nei_in_back)

        pos_ = locality + globality
        ind = pos_.coalesce()._indices()
        anchor = ind[0]
        positive = ind[1]
        negative = torch.tensor(random.choices(list(range(n_data)), k=len(anchor))).to(self.device)
        return anchor, positive, negative

    def create_sparse(self, I):

        similar = I.reshape(-1).tolist()
        index = np.repeat(range(I.shape[0]), I.shape[1])

        assert len(similar) == len(index)
        indices = torch.tensor([index, similar]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])

        return result

    def create_sparse_revised(self, I, all_close_nei_in_back):
        n_data, k = I.shape[0], I.shape[1]

        index = []
        similar = []
        for j in range(I.shape[0]):
            for i in range(k):
                index.append(int(j))
                similar.append(I[j][i].item())

        index = torch.masked_select(torch.LongTensor(index).to(self.device), all_close_nei_in_back.reshape(-1))
        similar = torch.masked_select(torch.LongTensor(similar).to(self.device), all_close_nei_in_back.reshape(-1))

        assert len(similar) == len(index)
        indices = torch.tensor([index.cpu().numpy().tolist(), similar.cpu().numpy().tolist()]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones(len(index)).to(self.device), [n_data, n_data])

        return result


class AFRM(nn.Module):
    def __init__(self, num_clusters, input_dim, config, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.dec_in_dim = config['latent_dim']
        self.online_encoder = Encoder(input_dim, config)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self._init_target()
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.rep_mask = nn.Parameter(torch.zeros(1, self.dec_in_dim))
        self.decoder = Decoder(input_dim, config)

        # 可选
        self.encoder_to_decoder = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)

        self.projector = Projector(config)

        # for anchor
        self.neighbor = Neighbor(device, num_clusters, config['num_kmeans'], config['clus_num_iters'])
        self.topk = config['topk']
        self.anchor_pair = None
        # for mask
        self.mask_rate = config['mask_rate']
        self.replace_rate = config['replace_rate']
        self.mask_token_rate = 1 - self.replace_rate
        # for recon loss
        self.t = config['t']
        # for momentum
        self.momentum_rate = config['momentum_rate']

    def _init_target(self):
        for param_teacher in self.target_encoder.parameters():
            param_teacher.detach()
            param_teacher.requires_grad = False

    def encoding_mask_noise(self, x, edge_index, mask_rate=0.3):
        num_nodes = x.shape[0]
        self.num_nodes = num_nodes
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]

        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_edge_index = edge_index.clone()

        return out_x, use_edge_index, (mask_nodes, keep_nodes)

    def momentum_update(self, base_momentum=0.1):
        for param_encoder, param_teacher in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                param_teacher.data = param_teacher.data * base_momentum + param_encoder.data * (1. - base_momentum)



from torch_geometric.nn.inits import uniform
class MainModel(nn.Module):
    def __init__(self, num_clusters, input_dim, img_dim, config, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.attn_type = config['attn_type']

        self.mask_rate = config['mask_rate']
        self.replace_rate = config['replace_rate']
        self.mask_token_rate = 1 - self.replace_rate

        self.use_ot = config['use_ot']

        self.exp_model = AFRM(num_clusters, input_dim, config, device).to(device)
        self.img_model = Encoder(img_dim, config).to(device) # image encoder
        self.img_exp_cross = CrossAttention(config['latent_dim'], num_heads=2, use_bias=False, attn_drop=0., proj_drop=0.)


        if ((config['attn_type'] == 'q-exp_k-img_v-img') or (config['attn_type'] == 'q-img_k-exp_v-exp')
                or (config['attn_type'] == 'clfs_q-exp_k-img_v-img') or (config['attn_type'] == 'clfs_q-img_k-exp_v-exp')):
            self.att = CrossAttention(config['latent_dim'], num_heads=2, use_bias=False, attn_drop=0., proj_drop=0.)
        else:
            self.att = MultiHeadAttention(config['latent_dim'] * 2, num_heads=4, use_bias=False, attn_drop=0., proj_drop=0.)
        self.redu_dim = nn.Linear(config['latent_dim'] * 2, config['latent_dim'])

        self.decoder = Decoder(input_dim, config)

        self.weight = nn.Parameter(torch.empty(config['latent_dim'], config['latent_dim']))
        uniform(config['latent_dim'], self.weight)

        self.t = config['t']

    def encoding_mask_noise(self, x, edge_index, mask_rate=0.3):
        num_nodes = self.num_nodes
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]

        else:
            token_nodes = mask_nodes

            out_x = x.clone()
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.exp_model.enc_mask_token
        use_edge_index = edge_index.clone()

        return out_x, use_edge_index, (mask_nodes, keep_nodes)

    def generate_neg_nodes(self, mask_nodes):
        num_mask_nodes = mask_nodes.size(0)
        neg_nodes_x = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
        neg_nodes_y = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
        return neg_nodes_x, neg_nodes_y

    def avg_readout(self, rep_pos_x, adj):
        from torch_scatter import scatter_add

        masked_adj = (1 * (torch.where(adj == 1, torch.rand_like(adj.to(torch.float)), adj) > ( 1 - self.gamma)))
        adj_indices = masked_adj.to_sparse_coo()._indices()

        src, dst = adj_indices

        neighbor_sum = scatter_add(rep_pos_x[src], dst, dim=0, dim_size=rep_pos_x.size(0))
        neighbor_count = scatter_add(torch.ones_like(src, dtype=torch.float), dst, dim=0, dim_size=rep_pos_x.size(0))
        neighbor_count = neighbor_count.clamp(min=1)
        summary = neighbor_sum / neighbor_count.unsqueeze(-1)

        # return torch.sigmoid(summary)
        return summary

    def sce_loss(self, x, y, t=2):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
        loss = -torch.log(cos_m.pow_(t))
        return loss.mean()

    def match_loss(self, rep, rep_t, mask_nodes, t=2):
        pox_x_index, pox_y_index = mask_nodes, mask_nodes
        neg_x_index, neg_y_index = self.generate_neg_nodes(mask_nodes)
        std_emb = F.normalize(rep.clone(), p=2, dim=-1)
        tgt_emb = F.normalize(rep_t.clone(), p=2, dim=-1)

        pox_x = std_emb[pox_x_index]
        pox_y = tgt_emb[pox_y_index]
        neg_x = std_emb[neg_x_index]
        neg_y = tgt_emb[neg_y_index]

        # pos_loss = (1 - (pox_x * pox_y).sum(dim=-1)).pow_(t)
        # neg_loss = ((neg_x * neg_y).sum(dim=-1)).pow_(t)
        # loss = 0.5 * (pos_loss + neg_loss)
        # return loss

        pos_cos = (0.5 * (1 + (pox_x * pox_y).sum(dim=-1))).pow(t)
        pos_loss = -torch.log(pos_cos)
        neg_cos = (0.5 * (1 + (neg_x * neg_y).sum(dim=-1))).pow(t)
        neg_loss = -torch.log(1 - neg_cos)
        loss = 0.5 * (pos_loss.mean() + neg_loss.mean())
        return loss

    def mask_attr_prediction(self, x, img, edge_index, adj_coo):
        self.num_nodes = x.shape[0]
        use_x, use_adj, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, edge_index, self.mask_rate)

        enc_rep = self.exp_model.online_encoder(use_x, use_adj)

        # 冲量更新
        with torch.no_grad():
            # 表达更新
            x_t = x.clone()
            x_t[keep_nodes] = 0.0
            x_t[keep_nodes] += self.exp_model.enc_mask_token
            enc_rep_t = self.exp_model.target_encoder(x_t, use_adj)
            rep_t = enc_rep_t
            self.exp_model.momentum_update(self.exp_model.momentum_rate)

        # 表达
        rep = enc_rep
        rep = self.exp_model.encoder_to_decoder(rep)
        rep[mask_nodes] = 0.
        rep[mask_nodes] += self.exp_model.rep_mask
        rep = self.exp_model.projector(rep, use_adj)

        # 表达
        match_loss = self.match_loss(rep, rep_t, mask_nodes)

        img_rep = self.img_model(img, edge_index)
        rep_cond = self.avg_readout(rep, adj_coo.to_dense())
        img_rep = self.img_exp_cross(img_rep, rep_cond, rep_cond)

        # online = rep[mask_nodes]
        # target = rep_t[mask_nodes]

        if self.attn_type == 'q-exp_k-img_v-img' or self.attn_type == 'clfs_q-exp_k-img_v-img':
            latent = self.att(rep, img_rep, img_rep)
        elif self.attn_type == 'q-img_k-exp_v-exp' or self.attn_type == 'clfs_q-img_k-exp_v-exp':
            latent = self.att(img_rep, rep, rep)
        else:
            combine_latent = torch.concatenate([rep, img_rep], dim=-1)
            latent = self.att(combine_latent)
            latent = self.redu_dim(latent)

        # latent dgi loss
        # summary = self.avg_readout(latent, use_adj)
        # neg_node = torch.randint(0, len(keep_nodes), (len(mask_nodes),))
        # dis_loss = self.dgi_loss(latent[mask_nodes], latent[neg_node], summary[mask_nodes])


        # # remask(optional)
        # # 表达
        # rep[keep_nodes] = 0.0
        # # 图像
        # rep_t[keep_nodes] = 0.0

        # 表达
        recon = self.decoder(latent, use_adj)
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]


        # mean_loss = F.mse_loss(online, target)
        rec_loss = self.sce_loss(x_rec, x_init, t=self.t)

        return match_loss, rec_loss

    def forward(self, x, img, edge_index, adj_coo):
        return self.mask_attr_prediction(x, img, edge_index, adj_coo)

    @torch.no_grad()
    def evaluate(self, x, img, edge_index, adj_coo):
        enc_rep = self.exp_model.target_encoder(x, edge_index)

        rep = self.exp_model.encoder_to_decoder(enc_rep)

        rep = self.exp_model.projector(rep, edge_index)

        img_rep = self.img_model(img, edge_index)
        rep_cond = self.avg_readout(rep, adj_coo.to_dense())
        img_rep = self.img_exp_cross(img_rep, rep_cond, rep_cond)

        if self.attn_type == 'q-exp_k-img_v-img' or self.attn_type == 'clfs_q-exp_k-img_v-img':
            latent = self.att(rep, img_rep, img_rep)
        elif self.attn_type == 'q-img_k-exp_v-exp' or self.attn_type == 'clfs_q-img_k-exp_v-exp':
            latent = self.att(img_rep, rep, rep)
        else:
            combine_latent = torch.concatenate([rep, img_rep], dim=-1)
            latent = self.att(combine_latent)
            latent = self.redu_dim(latent)
        recon = self.decoder(latent, edge_index)

        return latent, recon

