import os
import torch
import torch.nn.modules.loss
from tqdm import tqdm
from torch.backends import cudnn

from Models import *


class spaicl:
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

            match_loss, rec_loss = self.model(self.X, self.img, self.adj_norm, self.adj_coo)
            loss = (self.train_config['w_recon'] * rec_loss + self.train_config['w_match'] * match_loss +
                    self.train_config['w_img_match'] * 0 + self.train_config['w_ot'] * 0)

            loss.backward()
            if self.train_config['gradient_clipping'] > 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.train_config['gradient_clipping'])
            self.optimizer.step()

            pbar.set_description(
                "Epoch {0} total loss={1:.3f} recon loss={2:.3f} match loss={3:.3f}".format(
                    epoch, loss, rec_loss, match_loss), refresh=True)

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