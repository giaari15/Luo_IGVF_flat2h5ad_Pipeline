import os,sys
import re
import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype']=42
import seaborn as sns
import warnings
from scipy.sparse import csr_matrix
warnings.filterwarnings("ignore")
import squidpy as sq
import anndata as ad

fileDir_slide3 = '/u/project/cluo/giaari15/IGVF_spatial_data/MB_20260211_IGVF_repro_slide3_segE6/'
fileDir_slide4 = '/u/project/cluo/giaari15/IGVF_spatial_data/MB_20260211_IGVF_repro_slide4_segE6/'
data_dir = '/u/project/cluo/giaari15/igvf_spatial/processed_data/20260429/'

exprMat3 = 'MB_20260211_IGVF_repro_slide3_exprMat_file.csv.gz'
exprMat4 = 'MB_20260211_IGVF_repro_slide4_exprMat_file.csv.gz'
meta3 = 'MB_20260211_IGVF_repro_slide3_metadata_file.csv.gz'
meta4 = 'MB_20260211_IGVF_repro_slide4_metadata_file.csv.gz'
name3 = 'MB_20260211_IGVF_repro_slide3_segE6'
name4 = 'MB_20260211_IGVF_repro_slide4_segE6'

def processSlides(filePath, exprMat, metadata, name):
    chunks = []
    chunksize = 100000
    for chunk in pd.read_csv(filePath + exprMat, chunksize=chunksize):
        chunks.append(chunk)

    df1 = pd.concat(chunks, ignore_index=True)
    df = df1
    
    df["new_cell_ID"] =  df.apply(lambda row: f"c_1_{row.fov}_{row.cell_ID}", axis = 1)
    df.set_index("new_cell_ID", inplace=True)
    df.drop(columns=["fov", "cell_ID"], inplace=True)
    dummy=re.compile(r'Negative|SystemControl', flags=re.IGNORECASE)
    chosen_probes = [col for col in list(df.columns) if not dummy.search(col)]
    chosen_cells = df.index
    raw = csr_matrix(df[chosen_probes].astype(pd.SparseDtype("float64",0)).sparse.to_coo())
    del df

    cell_meta=pd.read_csv(filePath + metadata)
    cell_meta.set_index("cell", inplace=True, drop = False)
    coords = cell_meta[["CenterX_global_px","CenterY_global_px"]]
    coords = coords.rename(columns={"CenterX_global_px":"x", 
                                "CenterY_global_px": "y"})
    coords = coords.reindex(index = chosen_cells)
    pixel_size = 0.12028
    coords = coords.mul(pixel_size)

    adata = ad.AnnData(
    X = raw, 
    obs = cell_meta, 
    # row names should be the same as gene names
    var = pd.DataFrame(
        list(chosen_probes), 
        columns = ["gene"],
        index = chosen_probes))
    adata.obsm['spatial'] = coords.to_numpy()
    adata.uns['name'] = name
    adata.strings_to_categoricals()

    return adata

adata3 = processSlides(fileDir_slide3, exprMat3, meta3, name3)
adata4 = processSlides(fileDir_slide4, exprMat4, meta4, name4)

adata3.write(os.path.join(data_dir, "MB_20260211_IGVF_repro_slide3_segE6_raw1.h5ad"), compression="gzip")
adata4.write(os.path.join(data_dir, "MB_20260211_IGVF_repro_slide4_segE6_raw1.h5ad"), compression="gzip")

slide3 = sc.read(os.path.join(data_dir, "MB_20260211_IGVF_repro_slide3_segE6_raw1.h5ad"))
slide4 = sc.read(os.path.join(data_dir, "MB_20260211_IGVF_repro_slide4_segE6_raw1.h5ad"))

def labelData(slide, x_cutoff, y_cutoff, days):
    fov_centroids = slide.obs.groupby('fov')[['CenterX_global_px', 'CenterY_global_px']].mean()
    quadrants = [
        (fov_centroids['CenterX_global_px'] < x_cutoff) & (fov_centroids['CenterY_global_px'] < y_cutoff), # Bottom-Left
        (fov_centroids['CenterX_global_px'] >= x_cutoff) & (fov_centroids['CenterY_global_px'] < y_cutoff), # Bottom-Right
        (fov_centroids['CenterX_global_px'] < x_cutoff) & (fov_centroids['CenterY_global_px'] >= y_cutoff), # Top-Left
        (fov_centroids['CenterX_global_px'] >= x_cutoff) & (fov_centroids['CenterY_global_px'] >= y_cutoff) # Top-Right
    ]

    donors = ['C38', 'C29', 'C38', 'C29']

    fov_centroids['day'] = np.select(quadrants, days, default='Unknown')
    fov_centroids['donor'] = np.select(quadrants, donors, default='Unknown')

    fov_map1 = fov_centroids['day'].to_dict()
    fov_map2 = fov_centroids['donor'].to_dict()

    slide.obs['day'] = slide.obs['fov'].map(fov_map1)
    slide.obs['donor'] = slide.obs['fov'].map(fov_map2)

    print(slide.obs['day'].value_counts(dropna=False))
    print(slide.obs['donor'].value_counts(dropna=False))

    return slide

x_cutoff = 65000
y_cutoff = 60000
days3 = ['day3','day3','day7','day7']
days4 = ['day9','day9','day13','day13']
slide3 = labelData(slide3, x_cutoff, y_cutoff, days3)
slide4 = labelData(slide4, x_cutoff, y_cutoff, days4)

def combineSlides(slide1, slide2, keys):
    slide1.obsm['spatial'] = slide1.obs[['CenterX_global_px', 'CenterY_global_px']].to_numpy()
    slide2.obsm['spatial'] = slide2.obs[['CenterX_global_px', 'CenterY_global_px']].to_numpy()

    adata = ad.concat(
        [slide1, slide2],
        label = 'batch',
        keys = keys,
        join = 'outer',
        merge = 'same'
    )

    adata.obs_names = (
        adata.obs['batch'].astype(str) 
        + '-' 
        + adata.obs_names.astype(str)
    )
    adata.obs.index.name = 'cell_id_unique'

    adata.obs['vis_x'] = adata.obsm['spatial'][:, 0]
    adata.obs['vis_y'] = adata.obsm['spatial'][:, 1]

    max_x = adata[adata.obs['batch'] == keys[0]].obs['vis_x'].max()

    # buffer
    shift_amount = max_x + 50000

    slide2_mask = adata.obs['batch'] == keys[1]
    adata.obs.loc[slide2_mask, 'vis_x'] += shift_amount

    adata.obsm['spatial_joined'] = adata.obs[['vis_x', 'vis_y']].to_numpy()

    return adata

keys = ['slide3', 'slide4']
adata = combineSlides(slide3, slide4, keys)

adata.write(os.path.join(data_dir, "MB_20260211_IGVF_slide3+4_segE6_combined.h5ad"), compression="gzip")