import os,csv,re, gc
import pandas as pd
import numpy as np
import scanpy as sc
import math
# import SpaGCN as spg
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import SpaGCN as spg
#In order to read in image data, we need to install some package. Here we recommend package "opencv"
#inatll opencv in python
#!pip3 install opencv-python
import cv2
from scanpy import read_10x_h5

from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, completeness_score, \
    fowlkes_mallows_score, homogeneity_score
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score

# the location of R (used for the mclust clustering)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

def measureClusteringTrueLabel(labels_true, labels_pred):
    ari = adjusted_rand_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    cs = completeness_score(labels_true, labels_pred)
    fms = fowlkes_mallows_score(labels_true, labels_pred)
    vms = v_measure_score(labels_true, labels_pred)
    hs = homogeneity_score(labels_true, labels_pred)
    return ari, ami, nmi, cs, fms, vms, hs

def train_one_tissue(tissue_name):
    def load_data(tissue_name):
        adata_path = "D:\\project\\datasets\\DLPFC\\" + tissue_name + "/" + tissue_name + "_filtered_feature_bc_matrix.h5"
        adata = read_10x_h5(adata_path)
        spatial_path = "D:\\project\\datasets\\DLPFC\\" + tissue_name + "/spatial/tissue_positions_list.csv"
        spatial = pd.read_csv(spatial_path, sep=",", header=None,na_filter=False, index_col=0)
        adata.obs["x1"] = spatial[1]
        adata.obs["x2"] = spatial[2]
        adata.obs["x3"] = spatial[3]
        adata.obs["x4"] = spatial[4]
        adata.obs["x5"] = spatial[5]
        adata.obs["x_array"] = adata.obs["x2"]
        adata.obs["y_array"] = adata.obs["x3"]
        adata.obs["x_pixel"] = adata.obs["x4"]
        adata.obs["y_pixel"] = adata.obs["x5"]
        # Select captured samples
        adata = adata[adata.obs["x1"] == 1]
        adata.var_names = [i.upper() for i in list(adata.var_names)]
        adata.var["genename"] = adata.var.index.astype("str")
        # adata.write_h5ad("../tutorial/data/151673/sample_data.h5ad")
        # Read in gene expression and spatial location
        # adata = sc.read("../tutorial/data/151673/sample_data.h5ad")
        # Read in hitology image
        img_path = "D:\\project\\datasets\\DLPFC\\" + tissue_name + "/spatial/" + tissue_name + "_full_image.tif"
        img = cv2.imread(img_path)

        truth_path = "D:\\project\\datasets\\DLPFC\\" + tissue_name + '/' + tissue_name + '_truth.txt'
        Ann_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
        Ann_df.columns = ['Ground Truth']
        adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

        n_clusters = {'151507': 7, '151508': 7, '151509': 7, '151510': 7,
                      '151669': 5, '151670': 5, '151671': 5, '151672': 5,
                      '151673': 7, '151674': 7, '151675': 7, '151676': 7}

        cluster_n = n_clusters[tissue_name]
        return adata, img, cluster_n

    adata, img, cluster_n = load_data(tissue_name)

    x_array = adata.obs["x_array"].tolist()
    y_array = adata.obs["y_array"].tolist()
    x_pixel = adata.obs["x_pixel"].tolist()
    y_pixel = adata.obs["y_pixel"].tolist()

    # Run SpaGCN
    adata.obs["pred"] = spg.detect_spatial_domains_ez_mode(adata, img, x_array, y_array, x_pixel, y_pixel, n_clusters=cluster_n,
                                                           histology=True, s=1, b=49, p=0.5, r_seed=100, t_seed=100,
                                                           n_seed=100)

    adata.obs["pred"] = adata.obs["pred"].astype('category')
    # Refine domains (optional)
    # shape="hexagon" for Visium data, "square" for ST data.
    adata.obs["refined_pred"] = spg.spatial_domains_refinement_ez_mode(sample_id=adata.obs.index.tolist(),
                                                                       pred=adata.obs["pred"].tolist(), x_array=x_array,
                                                                       y_array=y_array, shape="hexagon")
    adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')

    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['pred'], obs_df['Ground Truth'])
    NMI = v_measure_score(obs_df['pred'], obs_df['Ground Truth'])
    print(f"ARI:{ARI}\tNMI:{NMI}")
    return adata, ARI, NMI

def train_dlpfc():
    ari_list = []
    nmi_list = []
    for tissue_name in ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672",
                        "151673", "151674", "151675", "151676"]:
        adata, ARI, NMI = train_one_tissue(tissue_name)
        # adata.write('./res/' + tissue_name + "_results.h5ad")
        ari_list.append(ARI)
        nmi_list.append(NMI)
    mid_ari = np.median(ari_list)
    mid_nmi = np.median(nmi_list)
    print(f"mid_ARI:{mid_ari}\tmid_NMI:{mid_nmi}")

def train_her2st():
    save_path = '../../result/baseline/HER2ST/SpaGCN'
    ari_list = []
    nmi_list = []
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    slice_list = ['A1', 'B1', 'C1', 'D1', 'E1', 'G1', 'H1']
    # slice_list = ['B3']
    for slice_name in slice_list:
        print('*' * 50, slice_name, '*' * 50)

        def load_data(slice_name):
            adata = sc.read_h5ad(f'E:\Topic\st cluster\data\her2st-master\preprocess/{slice_name}/preprocessed.h5ad')
            adata.var_names_make_unique()
            # adata.obs["x1"] = spatial[1]

            adata.obs["x_array"] = adata.obs["array_row"]
            adata.obs["y_array"] = adata.obs["array_col"]
            adata.obs["x_pixel"] = adata.obsm["spatial"][:, 0]
            adata.obs["y_pixel"] = adata.obsm["spatial"][:, 1]

            # Select captured samples
            # adata = adata[adata.obs["x1"] == 1]
            adata.var_names = [i.upper() for i in list(adata.var_names)]
            adata.var["genename"] = adata.var.index.astype("str")
            # adata.write_h5ad("../tutorial/data/151673/sample_data.h5ad")
            # Read in gene expression and spatial location
            # adata = sc.read("../tutorial/data/151673/sample_data.h5ad")
            # Read in hitology image
            img_path = f'E:\Topic\st cluster\data\her2st-master\data\ST-imgs/{slice_name[0]}/{slice_name}.jpg'
            img = cv2.imread(img_path)

            cluster_n = adata.uns['num_cluster']
            return adata, img, cluster_n

        adata, img, cluster_n = load_data(slice_name)

        x_array = adata.obs["x_array"].tolist()
        y_array = adata.obs["y_array"].tolist()
        x_pixel = adata.obs["x_pixel"].tolist()
        y_pixel = adata.obs["y_pixel"].tolist()

        # Run SpaGCN
        adata.obs["pred"] = spg.detect_spatial_domains_ez_mode(adata, img, x_array, y_array, x_pixel, y_pixel,
                                                               n_clusters=cluster_n,
                                                               histology=True, s=1, b=49, p=0.5, r_seed=100, t_seed=100,
                                                               n_seed=100)

        adata.obs["pred"] = adata.obs["pred"].astype('category')
        # Refine domains (optional)
        # shape="hexagon" for Visium data, "square" for ST data.
        adata.obs["refined_pred"] = spg.spatial_domains_refinement_ez_mode(sample_id=adata.obs.index.tolist(),
                                                                           pred=adata.obs["pred"].tolist(),
                                                                           x_array=x_array,
                                                                           y_array=y_array, shape="hexagon")
        adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')

        obs_df = adata.obs.dropna()

        ARI = adjusted_rand_score(obs_df['pred'], obs_df['label'])
        NMI = v_measure_score(obs_df['pred'], obs_df['label'])

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

    def load_data():
        adata = sc.read('E:\Topic\st cluster\data\BCDC\preprocess\preprocessed.h5ad')
        adata.var_names_make_unique()
        spatial_path = f'E:\Topic\st cluster\data\BCDC\spatial/tissue_positions_list.csv'
        spatial = pd.read_csv(spatial_path, sep=",", header=None, na_filter=False, index_col=0)
        adata.obs["x1"] = spatial[1]
        adata.obs["x2"] = spatial[2]
        adata.obs["x3"] = spatial[3]
        adata.obs["x4"] = spatial[4]
        adata.obs["x5"] = spatial[5]
        adata.obs["x_array"] = adata.obs["x2"]
        adata.obs["y_array"] = adata.obs["x3"]
        adata.obs["x_pixel"] = adata.obs["x4"]
        adata.obs["y_pixel"] = adata.obs["x5"]
        # Select captured samples
        adata = adata[adata.obs["x1"] == 1]
        adata.var_names = [i.upper() for i in list(adata.var_names)]
        adata.var["genename"] = adata.var.index.astype("str")
        # Read in hitology image
        img_path = f'E:\Topic\st cluster\data\BCDC/Visium_FFPE_Human_Breast_Cancer_image.tif'
        img = cv2.imread(img_path)

        cluster_n = adata.uns['num_cluster']
        return adata, img, cluster_n

    adata, img, cluster_n = load_data()

    x_array = adata.obs["x_array"].tolist()
    y_array = adata.obs["y_array"].tolist()
    x_pixel = adata.obs["x_pixel"].tolist()
    y_pixel = adata.obs["y_pixel"].tolist()

    # Run SpaGCN
    adata.obs["pred"] = spg.detect_spatial_domains_ez_mode(adata, img, x_array, y_array, x_pixel, y_pixel,
                                                           n_clusters=cluster_n,
                                                           histology=True, s=1, b=49, p=0.5, r_seed=100, t_seed=100,
                                                           n_seed=100)

    adata.obs["pred"] = adata.obs["pred"].astype('category')
    # Refine domains (optional)
    # shape="hexagon" for Visium data, "square" for ST data.
    adata.obs["refined_pred"] = spg.spatial_domains_refinement_ez_mode(sample_id=adata.obs.index.tolist(),
                                                                       pred=adata.obs["pred"].tolist(),
                                                                       x_array=x_array,
                                                                       y_array=y_array, shape="hexagon")
    adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')

    obs_df = adata.obs.dropna()

    ARI = adjusted_rand_score(obs_df['pred'], obs_df['groud_truth'])
    NMI = v_measure_score(obs_df['pred'], obs_df['groud_truth'])

    print(f"final_ari:{ARI:.4f},final_nmi:{NMI:.4f}")
    adata.uns['ari'] = ARI
    adata.uns['nmi'] = NMI
    adata.write_h5ad(f'{save_path}/SpaGCN_BCDC.h5ad')
    gc.collect()


if __name__ == '__main__':
    ari_list, nmi_list = train_her2st()
    print(ari_list, nmi_list)
    print(np.median(ari_list), np.median(nmi_list))
    # print(train_bcdc())
