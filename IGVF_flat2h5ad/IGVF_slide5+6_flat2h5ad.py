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



fileDir_slide5 = '/u/project/cluo/giaari15/IGVF_spatial_data/MB_20260304_IGVF_repro_slide5_segE6/'
fileDir_slide6 = '/u/project/cluo/giaari15/IGVF_spatial_data/MB_20260304_IGVF_repro_slide6_segE6/'
data_dir = '/u/project/cluo/giaari15/igvf_spatial/processed_data/20260429/'

exprMat5 = 'MB_20260304_IGVF_repro_slide5_exprMat_file.csv.gz'
exprMat6 = 'MB_20260304_IGVF_repro_slide6_exprMat_file.csv.gz'
meta5 = 'MB_20260304_IGVF_repro_slide5_metadata_file.csv.gz'
meta6 = 'MB_20260304_IGVF_repro_slide6_metadata_file.csv.gz'
name5 = 'MB_20260304_IGVF_repro_slide5_segE6'
name6 = 'MB_20260304_IGVF_repro_slide6_segE6'

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

adata5 = processSlides(fileDir_slide5, exprMat5, meta5, name5)
adata6 = processSlides(fileDir_slide6, exprMat6, meta6, name6)



plt.rcParams['figure.figsize']=(10,14)
sns.scatterplot(data=adata5.obs,x='CenterX_global_px', y='CenterY_global_px', s=2, edgecolor=None)  # s parameter sets dot size, edgecolor=None removes outline
plt.show()

plt.rcParams['figure.figsize']=(10,14)
sns.scatterplot(data=adata6.obs,x='CenterX_global_px', y='CenterY_global_px', s=2, edgecolor=None)  # s parameter sets dot size, edgecolor=None removes outline
plt.show()



# write to h5ad file, gzip compression
adata5.write(os.path.join(data_dir, "MB_20260304_IGVF_repro_slide5_segE6_raw1.h5ad"), compression="gzip")
adata6.write(os.path.join(data_dir, "MB_20260304_IGVF_repro_slide6_segE6_raw1.h5ad"), compression="gzip")



slide5 = sc.read(os.path.join(data_dir, 'MB_20260304_IGVF_repro_slide5_segE6_raw1.h5ad'))
slide6 = sc.read(os.path.join(data_dir, 'MB_20260304_IGVF_repro_slide6_segE6_raw1.h5ad'))

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



x_cutoff = 60000
y_cutoff = 50000
days5 = ['day3','day3','day7','day7']
slide5 = labelData(slide5, x_cutoff, y_cutoff, days5)

# verify labels 
## by day
plt.rcParams['figure.figsize'] = (10, 14)
sns.scatterplot(
    data=slide5.obs,
    x='CenterX_global_px', 
    y='CenterY_global_px', 
    hue='day', 
    s=1, 
    edgecolor=None,
    palette='tab10'
)
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left', 
    markerscale=10, 
    title="Day"
)
# Draw the manual lines for reference
plt.axvline(x_cutoff, color='black', linestyle='--')
plt.axhline(y_cutoff, color='black', linestyle='--')
plt.title(f"Cells labeled by Day (Cutoff: {x_cutoff}, {y_cutoff})")
plt.show()

## by donor
plt.rcParams['figure.figsize'] = (10, 14)
sns.scatterplot(
    data=slide5.obs,
    x='CenterX_global_px', 
    y='CenterY_global_px', 
    hue='donor', 
    s=1, 
    edgecolor=None,
    palette='tab10'
)
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left', 
    markerscale=10, 
    title="Donor"
)
# Draw the manual lines for reference
plt.axvline(x_cutoff, color='black', linestyle='--')
plt.axhline(y_cutoff, color='black', linestyle='--')
plt.title(f"Cells labeled by Donor (Cutoff: {x_cutoff}, {y_cutoff})")
plt.show()



days6 = ['day9','day9','day13','day13']
slide6 = labelData(slide6, x_cutoff, y_cutoff, days6)

## by day
plt.rcParams['figure.figsize'] = (10, 14)
sns.scatterplot(
    data=slide6.obs,
    x='CenterX_global_px', 
    y='CenterY_global_px', 
    hue='day', 
    s=1, 
    edgecolor=None,
    palette='tab10'
)
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left', 
    markerscale=10, 
    title="Day"
)
# Draw the manual lines for reference
plt.axvline(x_cutoff, color='black', linestyle='--')
plt.axhline(y_cutoff, color='black', linestyle='--')
plt.title(f"Cells labeled by Day (Cutoff: {x_cutoff}, {y_cutoff})")
plt.show()

## by donor
plt.rcParams['figure.figsize'] = (10, 14)
sns.scatterplot(
    data=slide6.obs,
    x='CenterX_global_px', 
    y='CenterY_global_px', 
    hue='donor', 
    s=1, 
    edgecolor=None,
    palette='tab10'
)
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left', 
    markerscale=10, 
    title="Donor"
)
# Draw the manual lines for reference
plt.axvline(x_cutoff, color='black', linestyle='--')
plt.axhline(y_cutoff, color='black', linestyle='--')
plt.title(f"Cells labeled by Donor (Cutoff: {x_cutoff}, {y_cutoff})")
plt.show()



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


keys = ['slide5', 'slide6'] # for unique cell ids
adata = combineSlides(slide5, slide6, keys)



# to verify buffer and [slide5 slide6] format
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=adata.obs, 
    x='vis_x', 
    y='vis_y', 
    hue='batch',
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


# verify correct labels (Day)
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=adata.obs, 
    x='vis_x', 
    y='vis_y', 
    hue='day',
    s=0.5,
    edgecolor=None
)
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left', 
    markerscale=10, 
    title="Day"
)
plt.axis('equal')
plt.show()

# verify correct labels (donor)
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=adata.obs, 
    x='vis_x', 
    y='vis_y', 
    hue='donor',
    s=0.5,
    edgecolor=None
)
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left', 
    markerscale=10, 
    title="Donor"
)
plt.axis('equal')
plt.show()



# write to h5ad for combined slides
adata.write(os.path.join(data_dir, "MB_20260304_IGVF_repro_segE6_combined.h5ad"), compression="gzip")

