# Data folder

This folder contains the core datasets used in this project.

## Files

- `data.csv`

  - A matrix of RNA-Seq gene expression values.
  - Rows represent individual samples.
  - Columns represent gene expression levels (~20,000 genes).

- `labels.csv`
  - Contains the corresponding cancer type labels for each sample.
  - Two columns: `sample_id` and `class` (cancer type).

## Notes

- The dataset is sourced from the Kaggle dataset:  
  [Gene Expression Cancer RNA-Seq] (https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq?select=TCGA-PANCAN-HiSeq-801x20531)
- These files are merged and preprocessed in `src/data_preprocessing.py`.
- Ensure these files are placed before running the pipeline.
