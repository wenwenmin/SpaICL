#%%
# from Diffusionst.DenoiseST import DenoiseST
from Diffusionst.DenoiseSTMain import DenoiseST
import torch
import pandas as pd
import numpy as np
import scanpy as sc
#%%
from Diffusionst.repair_model import main_repair
#%%
from Diffusionst.utils import clustering
import os
import gc
#%%
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import v_measure_score


def train_her2st():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    save_path = '../../result/baseline/HER2ST/DiffusionST'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ari_list = []
    nmi_list = []
    for slice_name in ['A1', 'B1', 'C1', 'D1', 'E1', 'G1', 'H1']:
        print('*' * 50, slice_name, '*' * 50)
        adata = sc.read_h5ad(f'E:\Topic\st cluster\data\her2st-master\preprocess/{slice_name}/preprocessed.h5ad')
        n_clusters = adata.uns['num_cluster']
        # %%
        model = DenoiseST(adata, device=device, n_top_genes=4096)
        adata = model.train()

        df = pd.DataFrame(adata.obsm['emb'])
        # %%
        main_repair(adata, df, device, save_name=slice_name)
        # %%
        csv_file = slice_name + "_example.csv"
        data_df = pd.read_csv(csv_file, header=None)
        data_df = data_df.values
        adata.obsm['emb'] = data_df

        radius = 50
        tool = 'mclust'  # mclust, leiden, and louvain
        if tool == 'mclust':
            clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)
        elif tool in ['leiden', 'louvain']:
            clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01,
                       refinement=False)

        df = adata.obs['domain']
        df.to_csv("label_" + slice_name + ".csv")

        adata = adata[~pd.isnull(adata.obs['label'])]

        sub_adata = adata[~pd.isnull(adata.obs['label'])]
        ARI = ari_score(sub_adata.obs['label'], sub_adata.obs['domain'])
        NMI = v_measure_score(sub_adata.obs['domain'], sub_adata.obs['label'])
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
    slice_name = 'temp/bcdc'

    save_path = '../../result/baseline/BCDC'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    adata = sc.read('E:\Topic\st cluster\data\BCDC\preprocess\preprocessed.h5ad')
    adata.var_names_make_unique()
    n_clusters = adata.uns['num_cluster']
    # %%
    model = DenoiseST(adata, device=device, n_top_genes=4096)
    adata = model.train()

    df = pd.DataFrame(adata.obsm['emb'])
    # %%
    main_repair(adata, df, device, save_name=slice_name)
    # %%
    csv_file = slice_name + "_example.csv"
    data_df = pd.read_csv(csv_file, header=None)
    data_df = data_df.values
    adata.obsm['emb'] = data_df

    radius = 50
    tool = 'mclust'  # mclust, leiden, and louvain
    if tool == 'mclust':
        clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)
    elif tool in ['leiden', 'louvain']:
        clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01,
                   refinement=False)

    df = adata.obs['domain']
    df.to_csv(slice_name + "_label.csv")

    adata = adata[~pd.isnull(adata.obs['groud_truth'])]

    sub_adata = adata[~pd.isnull(adata.obs['groud_truth'])]
    ARI = ari_score(sub_adata.obs['groud_truth'], sub_adata.obs['domain'])
    NMI = v_measure_score(sub_adata.obs['domain'], sub_adata.obs['groud_truth'])
    print(f"final_ari:{ARI:.4f},final_nmi:{NMI:.4f}")
    adata.uns['ari'] = ARI
    adata.uns['nmi'] = NMI
    adata.write_h5ad(f'{save_path}/DiffusionST_BCDC.h5ad')
    gc.collect()
    return ARI, NMI


if __name__ == '__main__':
    ari_list, nmi_list = train_her2st()
    print(ari_list, nmi_list)
    print(np.median(ari_list), np.median(nmi_list))
    # ARI, NMI = train_bcdc()
    # print(ARI, NMI)