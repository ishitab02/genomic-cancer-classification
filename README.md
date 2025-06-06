# Cancer type classification from gene expression data

This project builds a machine learning pipeline to classify types of cancer based on RNA-Seq gene expression profiles. The dataset is sourced from Kaggle: Gene Expression Cancer RNA-Seq

## Objective

The goal is to predict the **cancer type** using high-dimensional genomic data (RNA-Seq). The pipeline includes:

- Gene filtering based on variance
- Dimensionality reduction (PCA)
- Model training and evaluation

## Dataset

- **Source:** Kaggle – Gene Expression Cancer RNA-Seq (subset of the TCGA dataset)
- **Format:** Two CSV files:
  - `data.csv`: Gene expression matrix (containing samples and genes)
  - `labels.csv`: Class labels (sample ID with cancer types)
- **Classes:** BRCA, LUAD, COAD, KIRC, PRAD
- **Features:** ~20,000 gene expression values per sample

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/genomic-cancer-classification.git
cd genomic-cancer-classification
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### 4. Download and Prepare the Dataset

- Visit the Kaggle dataset page (https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq?select=TCGA-PANCAN-HiSeq-801x20531)
- Download data.csv and labels.csv
- Place them inside the data/ folder at the project root.

## Running the Pipeline

### 1. Data preprocessing

```bash
python src/data_preprocessing.py
```

- Merges data.csv and labels.csv
- Uses z-score normalisation
- Drops low-variance genes
- Saves cleaned X and y

### 2. Dimensionality reduction

```bash
python src/dimensionality_reduction.py
```

- Applies PCA to retain 95% variance
- Saves reduced feature matrix

### 3. Training the model

```bash
python -m src.train
```

- Trains using Random Forest and XGBoost classifiers and compares them
- Saves the trained models

### 4. Evaluating the model

```bash
python src/evaluation.py
```

- Loads the saved model
- Prints accuracy, classification report, confusion matrix
