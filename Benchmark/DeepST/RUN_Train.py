import torch.cuda

from deepst import DeepST
from pathlib import Path
from PIL import Image

# 提升像素限制
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import os
import gc
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score
from sklearn.cluster import KMeans
# the location of R (used for the mclust clustering)

def train_one_tissue(tissue_name, save_path='./'):
    n_domains = 5 if tissue_name in ['151669', '151670', '151671', '151672'] else 7
    dst = DeepST.run(save_path=save_path, task="Identify_Domain", pre_epochs=1000, epochs=500, use_gpu=True)
    adata = dst._get_adata(platform='Visium', data_path=Path("D:\\project\\datasets\\DLPFC_Simple\\"), data_name=tissue_name, count_file=tissue_name + "_filtered_feature_bc_matrix.h5")
    ###### Segment the Morphological Image
    adata = dst._get_image_crop(adata, data_name=tissue_name)
    ###### "use_morphological" defines whether to use morphological images.
    adata = dst._get_augment(adata, spatial_type="LinearRegress", use_morphological=True)
    ###### Build graphs. "distType" includes "KDTree", "BallTree", "kneighbors_graph", "Radius", etc., see adj.py
    graph_dict = dst._get_graph(adata.obsm["spatial"], distType="BallTree")
    ###### Enhanced data preprocessing
    data = dst._data_process(adata, pca_n_comps=200)

    ###### Training models
    deepst_embed = dst._fit(
        data=data,
        graph_dict=graph_dict, )
    ###### DeepST outputs
    adata.obsm["DeepST_embed"] = deepst_embed
    ###### Define the number of space domains, and the model can also be customized. If it is a model custom priori = False.
    adata = dst._get_cluster_data(adata, n_domains=n_domains, priori=True)

    truth_path = "D:\\project\\datasets\\DLPFC\\" + tissue_name + '/' + tissue_name + '_truth.txt'
    Ann_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['DeepST_refine_domain'], obs_df['Ground Truth'])
    NMI = v_measure_score(obs_df['DeepST_refine_domain'], obs_df['Ground Truth'])
    print(f"final_ari:{ARI:.4f},final_nmi:{NMI:.4f}")

    ###### Spatial localization map of the spatial domain
    sc.pl.spatial(adata, color='DeepST_refine_domain', frameon=False, spot_size=150)
    # plt.savefig(os.path.join(save_path, f'{tissue_name}_domains.pdf'), bbox_inches='tight', dpi=300)
    # adata.write(save_path + tissue_name + '_results.h5ad')

def train_dlpfc():
    save_path = './output/DLPFC/'
    for tissue_name in ["151672"]:
        train_one_tissue(tissue_name, save_path)

def train_her2st():
    save_path = '../../result/baseline/HER2ST/DeepST'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ari_list = []
    nmi_list = []
    for slice_name in ['A1', 'B1', 'C1', 'D1', 'E1', 'G1', 'H1']:
        print('*' * 50, slice_name, '*' * 50)
        dst = DeepST.run(save_path=save_path, task="Identify_Domain", pre_epochs=1000, epochs=500, use_gpu=True)
        adata = sc.read_h5ad(f'E:\Topic\st cluster\data\her2st-master\preprocess/{slice_name}/preprocessed.h5ad')
        n_domains = adata.uns['num_cluster']
        adata.obs['slices_path'] = adata.obsm['slice_path']

        library_id = slice_name
        image_coor = adata.obsm["spatial"]
        img = plt.imread(f'E:\Topic\st cluster\data\her2st-master\data\ST-imgs/{slice_name[0]}/{slice_name}.jpg', 0)
        if 'spatial' not in adata.uns.keys():
            adata.uns["spatial"] = {}
            adata.uns["spatial"][library_id] = {}
            adata.uns["spatial"][library_id]["images"] = {}
        adata.uns["spatial"][library_id]["images"]["fulres"] = img / 255

        adata.obs["imagecol"] = image_coor[:, 0]
        adata.obs["imagerow"] = image_coor[:, 1]
        adata.uns["spatial"][library_id]["use_quality"] = "fulres"

        ###### Segment the Morphological Image
        adata = dst._get_image_crop(adata, data_name=slice_name)
        ###### "use_morphological" defines whether to use morphological images.
        adata = dst._get_augment(adata, spatial_type="LinearRegress", use_morphological=True)
        ###### Build graphs. "distType" includes "KDTree", "BallTree", "kneighbors_graph", "Radius", etc., see adj.py
        graph_dict = dst._get_graph(adata.obsm["spatial"], distType="BallTree")
        ###### Enhanced data preprocessing
        if adata.shape[0] < 200:
            # 样本数小于200时
            pca_dim_num = int(adata.shape[0] / 2)
        else:
            pca_dim_num = 200
        data = dst._data_process(adata, pca_n_comps=pca_dim_num)

        ###### Training models
        deepst_embed = dst._fit(
            data=data,
            graph_dict=graph_dict, )
        ###### DeepST outputs
        adata.obsm["DeepST_embed"] = deepst_embed
        ###### Define the number of space domains, and the model can also be customized. If it is a model custom priori = False.
        adata = dst._get_cluster_data(adata, n_domains=n_domains, priori=True)

        obs_df = adata.obs.dropna()
        ARI = adjusted_rand_score(obs_df['DeepST_refine_domain'], obs_df['label'])
        NMI = v_measure_score(obs_df['DeepST_refine_domain'], obs_df['label'])
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

    dst = DeepST.run(save_path=save_path, task="Identify_Domain", pre_epochs=1000, epochs=500, use_gpu=True)
    adata = sc.read('E:\Topic\st cluster\data\BCDC\preprocess\preprocessed.h5ad')
    adata.var_names_make_unique()
    n_domains = adata.uns['num_cluster']

    adata.obs['slices_path'] = adata.obsm['slice_path']

    library_id = 'Visium_FFPE_Human_Breast_Cancer'
    image_coor = adata.obsm["spatial"]
    img = plt.imread(f'E:\Topic\st cluster\data\BCDC\Visium_FFPE_Human_Breast_Cancer_image.tif', 0)
    if 'spatial' not in adata.uns.keys():
        adata.uns["spatial"] = {}
        adata.uns["spatial"][library_id] = {}
        adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"]["fulres"] = img / 255

    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = "fulres"

    ###### Segment the Morphological Image
    adata = dst._get_image_crop(adata, data_name='BCDC')
    ###### "use_morphological" defines whether to use morphological images.
    adata = dst._get_augment(adata, spatial_type="LinearRegress", use_morphological=True)
    ###### Build graphs. "distType" includes "KDTree", "BallTree", "kneighbors_graph", "Radius", etc., see adj.py
    graph_dict = dst._get_graph(adata.obsm["spatial"], distType="BallTree")
    ###### Enhanced data preprocessing
    if adata.shape[0] < 200:
        # 样本数小于200时
        pca_dim_num = int(adata.shape[0] / 2)
    else:
        pca_dim_num = 200
    data = dst._data_process(adata, pca_n_comps=pca_dim_num)

    ###### Training models
    deepst_embed = dst._fit(
        data=data,
        graph_dict=graph_dict, )
    ###### DeepST outputs
    adata.obsm["DeepST_embed"] = deepst_embed
    ###### Define the number of space domains, and the model can also be customized. If it is a model custom priori = False.
    adata = dst._get_cluster_data(adata, n_domains=n_domains, priori=True)

    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['DeepST_refine_domain'], obs_df['groud_truth'])
    NMI = v_measure_score(obs_df['DeepST_refine_domain'], obs_df['groud_truth'])
    print(f"final_ari:{ARI:.4f},final_nmi:{NMI:.4f}")
    adata.uns['ari'] = ARI
    adata.uns['nmi'] = NMI
    adata.write_h5ad(f'{save_path}/DeepST_BCDC.h5ad')
    print(ARI, NMI)


if __name__ == '__main__':
    # ari_list, nmi_list = train_her2st()
    # print(ari_list, nmi_list)
    # print(np.median(ari_list), np.median(nmi_list))
    train_bcdc()