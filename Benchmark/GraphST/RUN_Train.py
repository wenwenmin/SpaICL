import os
import gc
import torch
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score, normalized_mutual_info_score

from GraphST import GraphST


def train_her2st():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    save_path = '../../result/baseline/HER2ST/GraphST'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ari_list = []
    nmi_list = []
    for slice_name in ['A1', 'B1', 'C1', 'D1', 'E1', 'G1', 'H1']:
        print('*' * 50, slice_name, '*' * 50)
        adata = sc.read_h5ad(f'E:\Topic\st cluster\data\her2st-master\preprocess/{slice_name}/preprocessed.h5ad')
        n_clusters = adata.uns['num_cluster']

        model = GraphST.GraphST(adata, device=device)

        # train model
        adata = model.train()

        radius = 50

        tool = 'mclust'  # mclust, leiden, and louvain

        # clustering
        from GraphST.utils import clustering

        if tool == 'mclust':
            clustering(adata, n_clusters, radius=radius, method=tool,
                       refinement=True)  # For DLPFC dataset, we use optional refinement step.
        elif tool in ['leiden', 'louvain']:
            clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01,
                       refinement=False)

        adata = adata[~pd.isnull(adata.obs['label'])]

        obs_df = adata.obs.dropna()
        ARI = adjusted_rand_score(obs_df['domain'], obs_df['label'])
        NMI = normalized_mutual_info_score(obs_df['domain'], obs_df['label'])
        print(f"final_ari:{ARI:.4f},final_nmi:{NMI:.4f}")
        adata.uns['ari'] = ARI
        adata.uns['nmi'] = NMI
        ari_list.append(ARI)
        nmi_list.append(NMI)
        adata.write_h5ad(f'{save_path}/{slice_name}.h5ad')
        gc.collect()
    return ari_list, nmi_list


def train_bcdc():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    save_path = '../../result/baseline/BCDC'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    adata = sc.read('E:\Topic\st cluster\data\BCDC\preprocess\preprocessed.h5ad')
    adata.var_names_make_unique()
    n_clusters = adata.uns['num_cluster']

    model = GraphST.GraphST(adata, device=device)

    # train model
    adata = model.train()

    radius = 50

    tool = 'mclust'  # mclust, leiden, and louvain

    # clustering
    from GraphST.utils import clustering

    if tool == 'mclust':
        clustering(adata, n_clusters, radius=radius, method=tool,
                   refinement=True)  # For DLPFC dataset, we use optional refinement step.
    elif tool in ['leiden', 'louvain']:
        clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01,
                   refinement=False)

    adata = adata[~pd.isnull(adata.obs['groud_truth'])]

    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['domain'], obs_df['groud_truth'])
    NMI = normalized_mutual_info_score(obs_df['domain'], obs_df['groud_truth'])
    print(f"final_ari:{ARI:.4f},final_nmi:{NMI:.4f}")
    adata.uns['ari'] = ARI
    adata.uns['nmi'] = NMI
    adata.write_h5ad(f'{save_path}/GraphST_BCDC.h5ad')
    gc.collect()
    return ARI, NMI


if __name__ == '__main__':
    ari_list, nmi_list = train_her2st()
    print(ari_list, nmi_list)
    print(np.median(ari_list), np.median(nmi_list))
    # print(train_bcdc())