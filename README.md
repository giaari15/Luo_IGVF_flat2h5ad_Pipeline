# Luo_IGVF_flat2h5ad_Pipeline

**In this repo:** CosMX spatial transcriptomics exprMat and metadata flat files to .h5ad template script and the documentation for the flat2h5ad pipeline submitted to the Impact of Genomic Variation on Function Consortium data portal (data.igvf.org; PI: Chongyuan Luo).
- summary of spatial transcriptomics data available
- flat files to h5ad pipeline

**Study Overview:** Our group is investigating the efficiency of cellular reprogramming of fibroblasts into Induced Pluripotent Stem Cells (iPSCs) across 4 day points (day3, day7, day9, day13) between two different lines, a "good" and "bad" donor (C29, C38).


## Overview
"Raw" h5ad files for each slide and combined set of slides are available.

Slide5+6: 6K gene panel, slide5 contains days 3 and 7 and slide6 contains days 9 and 13.
Slide3+4: 1K gene panel, slide3 contains days 3 and 7 and slide4 contains days 9 and 13.
Slide1+2: Experimental run of two test slides with IPS and Fibroblast sections within the slide.


The pipeline reads in the exprMat (count matrix) and metadata (cell metadata) flat files in chunks and drops Negative/Control probes to create an AnnData object with the `processSlides` function. The raw h5ad files (gzip compression) are created after this processing. Then, the data is labelled based on the day and donor configuration of the quadrants using an `x_cutoff` and `y_cutoff` in the empty space. Lastly, the set of two slides are combined with batch as the keys with a 50,000 pixel buffer to ensure unique cell ids (in format: slideX-c_1_{FOV}_{CELL}).


Slides 1 and 2 used manual data labelling as these test slides were not set line-by-day, but by cell type regions. 