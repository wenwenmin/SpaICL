import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os, gc
import sys
from pathlib import Path
from sklearn.metrics.cluster import adjusted_rand_score, v_measure_score, normalized_mutual_info_score
import STAGATE_pyG

# the location of R (used for the mclust clustering)

def train_one_slice(sample_name):
    # path
    data_root = Path("D:\\project\\datasets\\DLPFC\\")
    n_clusters = 5 if sample_name in ['151669', '151670', '151671', '151672'] else 7
    count_file = sample_name + "_filtered_feature_bc_matrix.h5"
    adata = sc.read_visium(data_root / sample_name, count_file=count_file)
    adata.var_names_make_unique()

    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    ## Constructing the spatial network
    STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=150)
    STAGATE_pyG.Stats_Spatial_Net(adata)

    ## Runing STAGATE
    adata = STAGATE_pyG.train_STAGATE(adata)
    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)
    adata = STAGATE_pyG.mclust_R(adata, used_obsm='STAGATE', num_cluster=n_clusters)
    # adata = STAGATE_pyG.Kmeans_cluster(adata, used_obsm='STAGATE', num_cluster=n_clusters, key_added_pred='mclust')

    truth_path = "D:\\project\\datasets\\DLPFC\\" + sample_name + '/' + sample_name + '_truth.txt'
    Ann_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

    adata = adata[~pd.isnull(adata.obs['Ground Truth'])]
    ARI = adjusted_rand_score(adata.obs['Ground Truth'], adata.obs['mclust'])
    NMI = 0

    print(f"sample_name:{sample_name}\tARI:{ARI}\tNMI:{NMI}")
    return ARI, NMI, adata


def train_dlpfc():
    ari_list = []
    nmi_list = []
    for sample_name in ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672",
                         "151673", "151674", "151675", "151676"]:
        ARI, NMI, adata = train_one_slice(sample_name)
        ari_list.append(ARI)
        nmi_list.append(NMI)
    mid_ari = np.median(ari_list)
    mid_nmi = np.median(nmi_list)
    print(f"mid_ARI:{mid_ari}\tmid_NMI:{mid_nmi}")


def train_her2st():
    save_path = '../../result/baseline/HER2ST/STAGATE'
    ari_list = []
    nmi_list = []
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for slice_name in ['A1', 'B1', 'C1', 'D1', 'E1', 'G1', 'H1']:
        print('*' * 50, slice_name, '*' * 50)
        adata = sc.read_h5ad(f'E:\Topic\st cluster\data\her2st-master\preprocess/{slice_name}/preprocessed.h5ad')
        adata.var_names_make_unique()
        n_clusters = adata.uns['num_cluster']

        # Normalization
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e3)
        sc.pp.log1p(adata)

        ## Constructing the spatial network
        STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=500)
        STAGATE_pyG.Stats_Spatial_Net(adata)

        ## Runing STAGATE
        adata = STAGATE_pyG.train_STAGATE(adata)
        sc.pp.neighbors(adata, use_rep='STAGATE')
        sc.tl.umap(adata)
        adata = STAGATE_pyG.mclust_R(adata, used_obsm='STAGATE', num_cluster=n_clusters)
        # adata = STAGATE_pyG.Kmeans_cluster(adata, used_obsm='STAGATE', num_cluster=n_clusters, key_added_pred='mclust')

        adata = adata[~pd.isnull(adata.obs['label'])]

        ARI = adjusted_rand_score(adata.obs['label'], adata.obs['mclust'])
        NMI = normalized_mutual_info_score(adata.obs['label'], adata.obs['mclust'])
        print(f"final_ari:{ARI:.4f},final_nmi:{NMI:.4f}")
        adata.uns['ari'] = ARI
        adata.uns['nmi'] = NMI
        ari_list.append(ARI)
        nmi_list.append(NMI)
        adata.write_h5ad(f'{save_path}/{slice_name}.h5ad')
        gc.collect()
    return ari_list, nmi_list


def train_bcdc():
    save_path = '../../result/baseline/BCDC'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    adata = sc.read('E:\Topic\st cluster\data\BCDC\preprocess\preprocessed.h5ad')
    adata.var_names_make_unique()
    n_clusters = adata.uns['num_cluster']

    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e3)
    sc.pp.log1p(adata)

    ## Constructing the spatial network
    STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=100) # 150
    STAGATE_pyG.Stats_Spatial_Net(adata)

    ## Runing STAGATE
    adata = STAGATE_pyG.train_STAGATE(adata)
    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)
    adata = STAGATE_pyG.mclust_R(adata, used_obsm='STAGATE', num_cluster=n_clusters)
    # adata = STAGATE_pyG.Kmeans_cluster(adata, used_obsm='STAGATE', num_cluster=n_clusters, key_added_pred='mclust')

    adata = adata[~pd.isnull(adata.obs['groud_truth'])]

    ARI = adjusted_rand_score(adata.obs['groud_truth'], adata.obs['mclust'])
    NMI = normalized_mutual_info_score(adata.obs['groud_truth'], adata.obs['mclust'])
    print(f"final_ari:{ARI:.4f},final_nmi:{NMI:.4f}")
    adata.uns['ari'] = ARI
    adata.uns['nmi'] = NMI
    adata.write_h5ad(f'{save_path}/STAGATE_BCDC.h5ad')
    gc.collect()
    return ARI, NMI


if __name__ == '__main__':
    ari_list, nmi_list = train_her2st()
    print(ari_list, nmi_list)
    print(np.median(ari_list), np.median(nmi_list))