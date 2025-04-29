# SpaICL
![image](https://github.com/wenwenmin/SpaICL/blob/main/Fig1.png)

## Overview
Our proposed framework deeply integrates gene expression, spatial location, and tissue morphology information to construct a low-dimensional latent representation, enabling precise delineation of spatial domains. Initially, a complementary masking strategy is applied to the raw gene expression data, generating two complementary embedding representations via a shared graph neural network encoder. Meanwhile, histological images are processed through a pretrained deep convolutional neural network to extract local morphological features, which are then projected into the same space as the gene expression embeddings.
Next, the framework employs a dual-layer cross-attention module to align modalities, integrating them with the original gene expression latent representation. To ensure stable training, a curriculum learning strategy is adopted, gradually aligning local features before global alignment. Finally, the fused low-dimensional representation is used for downstream spatial domain identification, visualization, and other subsequent analyses.

## Datasets
Data are available at: https://zenodo.org/records/15289323.

## Installations
- NVIDIA GPU (a single Nvidia GeForce RTX 3060)
- `pip install -r requiremnt.txt`
  
## Running demo
We provide an example to test SpaICL on the DLPFC/BRCA/MBA dataset. You can test by running the Run_151675.ipynb/Run_BRCA.ipynb/Run_MBA.ipynb file.

## Contact details
If you have any questions, please contact zhao_jingcheng@aliyun.com and minwenwen@ynu.edu.cn.

