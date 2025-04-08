import os
import torch
import torch.nn.modules.loss
from tqdm import tqdm
from torch.backends import cudnn

from Models import *

import pandas as pd

import pymysql
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def insert_parameters(data):
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='local')
    cursor = conn.cursor()
    cursor.execute(f"insert into parameter_test (dataset, dataset_slice, attn_type, ari, ep, var_num, "
                   f"mask_rate, w_recon, w_match, w_img_match, w_ot ) "
                   f"values {','.join(str(tuple(data[i])) for i in range(len(data)))}")
    conn.commit()
    cursor.close()
    conn.close()
    return 1

# def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='imputation', key_added_pred='impute_mclust',
#              random_seed=666):
#     """\
#     Clustering using the mclust algorithm.
#     The parameters are the same as those in the R package mclust.
#     """
#
#     np.random.seed(random_seed)
#     import rpy2.robjects as robjects
#     robjects.r.library("mclust")
#
#     import rpy2.robjects.numpy2ri
#     rpy2.robjects.numpy2ri.activate()
#     r_random_seed = robjects.r['set.seed']
#     r_random_seed(random_seed)
#     rmclust = robjects.r['Mclust']
#
#     res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
#     if type(res) == rpy2.rinterface_lib.sexp.NULLType:
#         res = rmclust(adata.obsm[used_obsm], num_cluster)
#     mclust_res = np.array(res[-2])
#
#     adata.obs[key_added_pred] = mclust_res
#     adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('int')
#     adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('category')
#     return adata
#
# import ot
# def refine_label(adata, radius=30, key='label'):
#     n_neigh = radius
#     new_type = []
#     old_type = adata.obs[key].values
#
#     # calculate distance
#     position = adata.obsm['spatial']
#     distance = ot.dist(position, position, metric='euclidean')
#
#     n_cell = distance.shape[0]
#
#     for i in range(n_cell):
#         vec = distance[i, :]
#         index = vec.argsort()
#         neigh_type = []
#         for j in range(1, n_neigh + 1):
#             neigh_type.append(old_type[index[j]])
#         max_type = max(neigh_type, key=neigh_type.count)
#         new_type.append(max_type)
#
#     new_type = [str(i) for i in list(new_type)]
#     # adata.obs['label_refined'] = np.array(new_type)
#
#     return new_type

class Mgac:
    def __init__(self, adata, graph_dict, num_clusters,  device, config, roundseed=0):
        seed = config['seed'] + roundseed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        self.device = device
        self.adata = adata
        self.img = adata.obsm['img_pca'].to(device)
        self.graph_dict = graph_dict
        self.data_info = config['data']
        self.mode = config['mode']
        self.train_config = config['train']
        self.model_config = config['model']
        self.num_clusters = num_clusters

        if config['train']['w_ot'] == 0:
            self.model_config['use_ot'] = False
        else:
            self.model_config['use_ot'] = True


    def _start_(self):
        if self.mode == 'clustering':
            self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        elif self.mode == 'imputation':
            self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        else:
            raise Exception
        self.graph = self.graph_dict["adj_label"].to(self.device)
        # self.adj_norm = self.graph_dict["adj_norm"].to_dense().to(self.device)
        # self.adj_label = self.graph_dict["adj_label"].to_dense().to(self.device)
        self.adj_norm = self.graph_dict["adj_label"].coalesce()._indices().to(self.device)
        self.adj_coo = self.graph_dict["adj_label"].to_sparse_coo().to(self.device) # dense: .to_dense()

        self.norm_value = self.graph_dict["norm_value"]

        self.input_dim = self.X.shape[-1]
        self.img_dim = self.img.shape[-1]
        self.model = MainModel(self.num_clusters, self.input_dim, self.img_dim, self.model_config, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.train_config['lr'],
            weight_decay=self.train_config['decay'],
        )

    def _fit_(self):
        pbar = tqdm(range(self.train_config['epochs']))
        para_list = []
        for epoch in pbar:
            self.model.train()
            self.optimizer.zero_grad()

            if epoch < (self.train_config['epochs'] - int(self.train_config['epochs'] * 0.1)):
                self.model.gamma = epoch / self.train_config['epochs']
            else:
                self.model.gamma = 1

            # mean_loss, rec_loss, tri_loss = self.model(self.X, self.adj_norm, flag, self.graph)
            # loss = self.train_config['w_recon'] * rec_loss + self.train_config['w_mean'] * mean_loss +  self.train_config['w_tri'] * tri_loss

            match_loss, rec_loss = self.model(self.X, self.img, self.adj_norm, self.adj_coo)
            loss = (self.train_config['w_recon'] * rec_loss + self.train_config['w_match'] * match_loss +
                    self.train_config['w_img_match'] * 0 + self.train_config['w_ot'] * 0)

            loss.backward()
            if self.train_config['gradient_clipping'] > 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.train_config['gradient_clipping'])
            self.optimizer.step()

            # if (epoch + 1) % 50 == 0:
            #     self.model.eval()
            #     enc_rep, recon = self.model.evaluate(self.X, self.img, self.adj_norm, self.adj_coo)
            #     self.adata.obsm['latent'] = enc_rep.detach().cpu().numpy()
            #     self.adata.obsm['recon'] = recon.detach().cpu().numpy()
            #
            #     temp_adata = mclust_R(self.adata, num_cluster=self.num_clusters, used_obsm='latent', key_added_pred='mclust')
            #     temp_adata.obs['domain'] = refine_label(temp_adata, 30, key='mclust')
            #     sub_adata = temp_adata[~pd.isnull(temp_adata.obs['layer_guess'])]
            #
            #     from sklearn import metrics
            #     ari_res = metrics.adjusted_rand_score(sub_adata.obs['layer_guess'], sub_adata.obs['domain'])
            #     print(ari_res)
            #     para_list.append(
            #         [self.data_info['dataset'], self.data_info['dataset_slice'], self.model_config['attn_type'],
            #          ari_res, (epoch + 1),
            #          self.data_info['top_genes'], self.model_config['mask_rate'], self.train_config['w_recon'],
            #          self.train_config['w_match'], self.train_config['w_img_match'], self.train_config['w_ot']])
            #     self.model.train()

            # pbar.set_description(
            #     "Epoch {0} total loss={1:.3f} recon loss={2:.3f} match loss={3:.3f} img match loss={4:.3f} ot loss={5:.3f}".format(
            #         epoch, loss, rec_loss, match_loss, img_match_loss, ot_loss), refresh=True)
            pbar.set_description(
                "Epoch {0} total loss={1:.3f} recon loss={2:.3f} match loss={3:.3f}".format(
                    epoch, loss, rec_loss, match_loss), refresh=True)
        # insert_parameters(para_list)
        torch.cuda.empty_cache()
        return para_list

    def train(self):
        self._start_()
        adata_max = self._fit_()
        return adata_max

    def process(self):
        self.model.eval()
        enc_rep, recon = self.model.evaluate(self.X, self.img, self.adj_norm, self.adj_coo)
        return enc_rep, recon