import scanpy as sc
import pandas as pd
from sklearn import metrics
import torch

import matplotlib.pyplot as plt
import seaborn as sns

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import SEDR
import numpy as np
# the location of R (used for the mclust clustering)
import os
import gc
# the location of R (used for the mclust clustering)


def mclust_R(adata, n_clusters, use_rep='SEDR', key_added='SEDR', random_seed=2023):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    modelNames = 'EEE'

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[use_rep]), n_clusters, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata

def train_one_slice(sample_name):
    random_seed = 0
    SEDR.fix_seed(random_seed)

    # gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # path
    data_root = Path("D:\\project\\datasets\\DLPFC\\")

    n_clusters = 5 if sample_name in ['151669', '151670', '151671', '151672'] else 7
    count_file = sample_name + "_filtered_feature_bc_matrix.h5"
    adata = sc.read_visium(data_root / sample_name, count_file=count_file)
    adata.var_names_make_unique()

    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.
    adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X

    graph_dict = SEDR.graph_construction(adata, 10)

    sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
    using_dec = True
    if using_dec:
        sedr_net.train_with_dec(N=1)
    else:
        sedr_net.train_without_dec(N=1)
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm['SEDR_feat'] = sedr_feat

    sedr_recon = sedr_net.recon()
    adata.obsm['SEDR_recon'] = sedr_recon

    adata = mclust_R(adata, n_clusters, use_rep='SEDR_feat', key_added='mclust')

    truth_path = "D:\\project\\datasets\\DLPFC\\" + sample_name + '/' + sample_name + '_truth.txt'
    Ann_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

    adata = adata[~pd.isnull(adata.obs['Ground Truth'])]
    ARI = metrics.adjusted_rand_score(adata.obs['Ground Truth'], adata.obs['mclust'])
    NMI = metrics.normalized_mutual_info_score(adata.obs['Ground Truth'], adata.obs['mclust'])

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
    save_path = '../../result/baseline/HER2ST/SEDR'
    ari_list = []
    nmi_list = []
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for slice_name in ['A1', 'B1', 'C1', 'D1', 'E1', 'G1', 'H1']:
        print('*' * 50, slice_name, '*' * 50)
        adata = sc.read_h5ad(f'E:\Topic\st cluster\data\her2st-master\preprocess/{slice_name}/preprocessed.h5ad')
        adata.var_names_make_unique()

        # random_seed = 0
        # SEDR.fix_seed(random_seed)
        # gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        n_clusters = adata.uns['num_cluster']

        adata.layers['count'] = adata.X.toarray()
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable'] == True]
        sc.pp.scale(adata)

        from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.
        if adata.shape[0] < 200:
            # 样本数小于200时
            pca_dim_num = int(adata.shape[0] / 2)
        else:
            pca_dim_num = 200
        adata_X = PCA(n_components=pca_dim_num, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

        graph_dict = SEDR.graph_construction(adata, 10)

        sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
        using_dec = True
        if using_dec:
            sedr_net.train_with_dec(N=1)
        else:
            sedr_net.train_without_dec(N=1)
        sedr_feat, _, _, _ = sedr_net.process()
        adata.obsm['SEDR_feat'] = sedr_feat

        sedr_recon = sedr_net.recon()
        adata.obsm['SEDR_recon'] = sedr_recon

        adata = mclust_R(adata, n_clusters, use_rep='SEDR_feat', key_added='mclust')

        adata = adata[~pd.isnull(adata.obs['label'])]
        ARI = metrics.adjusted_rand_score(adata.obs['label'], adata.obs['mclust'])
        NMI = metrics.normalized_mutual_info_score(adata.obs['label'], adata.obs['mclust'])
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

    # random_seed = 0
    # SEDR.fix_seed(random_seed)
    # gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    n_clusters = adata.uns['num_cluster']

    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.
    if adata.shape[0] < 200:
        # 样本数小于200时
        pca_dim_num = int(adata.shape[0] / 2)
    else:
        pca_dim_num = 200
    adata_X = PCA(n_components=pca_dim_num, random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X

    graph_dict = SEDR.graph_construction(adata, 10)

    sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
    using_dec = True
    if using_dec:
        sedr_net.train_with_dec(N=1)
    else:
        sedr_net.train_without_dec(N=1)
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm['SEDR_feat'] = sedr_feat

    sedr_recon = sedr_net.recon()
    adata.obsm['SEDR_recon'] = sedr_recon

    adata = mclust_R(adata, n_clusters, use_rep='SEDR_feat', key_added='mclust')

    adata = adata[~pd.isnull(adata.obs['groud_truth'])]
    ARI = metrics.adjusted_rand_score(adata.obs['groud_truth'], adata.obs['mclust'])
    NMI = metrics.normalized_mutual_info_score(adata.obs['groud_truth'], adata.obs['mclust'])
    print(f"final_ari:{ARI:.4f},final_nmi:{NMI:.4f}")
    adata.uns['ari'] = ARI
    adata.uns['nmi'] = NMI
    adata.write_h5ad(f'{save_path}/SEDR_BCDC.h5ad')
    gc.collect()
    return ARI, NMI


if __name__ == '__main__':
    ari_list, nmi_list = train_her2st()
    print(ari_list, nmi_list)
    print(np.median(ari_list), np.median(nmi_list))
    # print(train_bcdc())
