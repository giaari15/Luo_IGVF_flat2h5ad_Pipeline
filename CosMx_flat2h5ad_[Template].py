# env: Python 3.9

import os,sys
import re
import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData
import warnings
from scipy.sparse import csr_matrix
warnings.filterwarnings("ignore")
import anndata as ad

# file paths
INPUT_FILES = {
    'filePath': 'data/',
    'exprMat':  'exprMat_file.csv.gz',
    'metadata': 'metadata_file.csv.gz',
    'name':     '',
}

OUTPUT_DIR = ''

# Processing CSV files to adata object
# Reads in data using filepath names above in chunks
def processSlides(filePath, exprMat, metadata, name):
    chunks = []
    chunksize = 100000 # adjust to available RAM, worked for 32GB
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
    pixel_size = 0.12028 # CosMx pixel to um rate
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

adata = processSlides(**INPUT_FILES)
adata.write(os.path.join(OUTPUT_DIR, 'data.h5ad'), compression = "gzip") # write adata to .h5ad

# label data
def labelData(slide, x_cutoff, y_cutoff, days, donors):
    """Label each FOV by centroid quadrant; cells inherit their FOV's label.
    Cutoffs are in pixel space. days/donors are length-4 in quadrant
    order: [Bottom-Left, Bottom-Right, Top-Left, Top-Right]. Mutates slide.obs.
    """
    assert len(days) == len(donors) == 4, "days/donors must be 4: [BL, BR, TL, TR]"

    c = slide.obs.groupby('fov', observed=True)[['CenterX_global_px', 'CenterY_global_px']].mean()
    x, y = c['CenterX_global_px'], c['CenterY_global_px']
    quadrants = [
        (x <  x_cutoff) & (y <  y_cutoff),  # BL
        (x >= x_cutoff) & (y <  y_cutoff),  # BR
        (x <  x_cutoff) & (y >= y_cutoff),  # TL
        (x >= x_cutoff) & (y >= y_cutoff),  # TR
    ]

    for col, vals in [('day', days), ('donor', donors)]:
        c[col] = np.select(quadrants, vals, default='Unknown')
        slide.obs[col] = slide.obs['fov'].map(c[col].to_dict())
        print(slide.obs[col].value_counts(dropna=False))

    return slide


# Combine two slides/adata objs
# keys = ['slideX', 'slideY'] or some label to keep different slides distinct
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

