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

# data filepaths, read in chunks using processSlides function
fileDir_slide1 = '/u/project/cluo/giaari15/IGVF_spatial_data/MB_20250710_ibidi_test1_redo_segE6/'
fileDir_slide2 = '/u/project/cluo/giaari15/IGVF_spatial_data/MB_20250710_ibidi_test2_redo_segE6/'
data_dir = '/u/project/cluo/giaari15/igvf_spatial/processed_data/20260429/'

exprMat1 = 'MB_20250710_ibidi_test1_redo_exprMat_file.csv.gz'
exprMat2 = 'MB_20250710_ibidi_test2_redo_exprMat_file.csv.gz'
meta1 = 'MB_20250710_ibidi_test1_redo_metadata_file.csv.gz'
meta2 = 'MB_20250710_ibidi_test2_redo_metadata_file.csv.gz'
name1 = 'MB_20250710_IGVF_ibidi_test1_segE6'
name2 = 'MB_20250710_IGVF_ibidi_test2_segE6'

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

adata1 = processSlides(fileDir_slide1, exprMat1, meta1, name1)
adata2 = processSlides(fileDir_slide2, exprMat2, meta2, name2)

adata1.write(os.path.join(data_dir, "MB_20250710_IGVF_ibidi_test1_segE6_raw1.h5ad"), compression="gzip")
adata2.write(os.path.join(data_dir, "MB_20250710_IGVF_ibidi_test2_segE6_raw1.h5ad"), compression="gzip")

slide1 = sc.read(os.path.join(data_dir, "MB_20250710_IGVF_ibidi_test1_segE6_raw1.h5ad"))
slide2 = sc.read(os.path.join(data_dir, "MB_20250710_IGVF_ibidi_test2_segE6_raw1.h5ad"))



# for slide 1, data labelling
x_cutoff = 60000
y_cutoff = 80000

fov_centroids = slide1.obs.groupby('fov')[['CenterX_global_px', 'CenterY_global_px']].mean()
quadrants = [
    (fov_centroids['CenterX_global_px'] < x_cutoff) & (fov_centroids['CenterY_global_px'] < y_cutoff), # Bottom-Left
    (fov_centroids['CenterX_global_px'] >= x_cutoff) & (fov_centroids['CenterY_global_px'] < y_cutoff), # Bottom-Right
    (fov_centroids['CenterX_global_px'] < x_cutoff) & (fov_centroids['CenterY_global_px'] >= y_cutoff), # Top-Left
    (fov_centroids['CenterX_global_px'] >= x_cutoff) & (fov_centroids['CenterY_global_px'] >= y_cutoff) # Top-Right
]

types = ['ips','ips','fibro','fibro']
fov_centroids['type'] = np.select(quadrants, types, default='Unknown')
fov_map1 = fov_centroids['type'].to_dict()
slide1.obs['type'] = slide1.obs['fov'].map(fov_map1)


# for slide 2, data labelling
x_cutoff = 60000
y_cutoff1 = 55000
y_cutoff2 = 120000

fov_centroids = slide2.obs.groupby('fov')[['CenterX_global_px', 'CenterY_global_px']].mean()
quadrants = [
    (fov_centroids['CenterX_global_px'] < x_cutoff) & (fov_centroids['CenterY_global_px'] < y_cutoff1), # Bottom-Left
    (fov_centroids['CenterX_global_px'] >= x_cutoff) & (fov_centroids['CenterY_global_px'] < y_cutoff1), # Bottom-Right
    (fov_centroids['CenterX_global_px'] < x_cutoff) & (fov_centroids['CenterY_global_px'] >= y_cutoff1) & (fov_centroids['CenterY_global_px'] < y_cutoff2), # Middle-Left
    (fov_centroids['CenterX_global_px'] >= x_cutoff) & (fov_centroids['CenterY_global_px'] >= y_cutoff1)  & (fov_centroids['CenterY_global_px'] < y_cutoff2), # Middle-Right
    (fov_centroids['CenterX_global_px'] < x_cutoff) & (fov_centroids['CenterY_global_px'] >= y_cutoff2), # Top-Left
    (fov_centroids['CenterX_global_px'] >= x_cutoff) & (fov_centroids['CenterY_global_px'] >= y_cutoff2) # Top-Right
]

types = ['ips','ips','fibro','ips', 'artifacts', 'artifacts']
fov_centroids['type'] = np.select(quadrants, types, default='Unknown')
fov_map1 = fov_centroids['type'].to_dict()
slide2.obs['type'] = slide2.obs['fov'].map(fov_map1)



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

keys = ['slide1', 'slide2'] # for unique cell ids
adata = combineSlides(slide1, slide2, keys)

# plot to verify
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=adata.obs, 
    x='vis_x', 
    y='vis_y', 
    hue='batch',  # or 'leiden' clusters
    s=0.5,
    edgecolor=None
)
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left', 
    markerscale=10, 
    title="Batch"
)
plt.axis('equal')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=adata.obs, 
    x='vis_x', 
    y='vis_y', 
    hue='type',
    s=0.5,
    edgecolor=None
)
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left', 
    markerscale=10, 
    title="Type"
)
plt.axis('equal')
plt.show()


# write combined adata object to .h5ad
adata.write(os.path.join(data_dir, "MB_20250710_IGVF_slide1+2_segE6_combined.h5ad"), compression="gzip")