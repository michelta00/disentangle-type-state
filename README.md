# Disentangling Cell Type and Cell State in Multi-Condition scRNA-seq Data

A comprehensive analysis pipeline for disentangling cell type and cell state variation in multi-condition single-cell RNA-sequencing data using interpretable machine learning approaches.

## Project Overview

This project implements and compares multiple computational approaches for separating cell type (stable biological identity) and cell state (dynamic condition-specific changes) in single-cell RNA-sequencing datasets. The analysis includes:

- **Patches**: A VAE-based approach for interpretable disentanglement of biological variation ([Beker *et al.*, 2025](https://www.biorxiv.org/content/10.1101/2024.12.23.630186v1))
- **DISCoVeR**: Deep generative modeling for discovering latent representations ([Slavutsky *et al.*, 2025](https://arxiv.org/abs/2506.17182))
- **Feature Selection Scores (FSS)**: Classical statistical approaches for gene selection ([Wang *et al.*, 2025](https://www.biorxiv.org/content/10.1101/2024.11.29.626057v2))

The pipeline evaluates these methods on both simulated ([Wang *et al.*, 2025](https://www.biorxiv.org/content/10.1101/2024.11.29.626057v2)) and experimental datasets ([Kang et al., 2018](https://www.nature.com/articles/nbt.4042)), using multiple downstream analyses including clustering (ARI), mixing (LISI), principal component regression (PCR), and mutual information (MI) metrics.

This repository is associated with the ETH Zurich semester project:\
**"Exploring Methods for Disentangling Cell Type and Cell State in Multi-Condition scRNA-seq Data"**\
Student: Michel Tarnow\
Supervisors: Jiayi Wang, Prof. Dr. Mark D. Robinson

## Repository Structure

```
.
├── data/                   # Data storage (gitignored)
│   ├── kang/                 # Experimental data (Kang18)
│   │   ├── 00-raw/             # Raw data
│   │   ├── 01-pro/             # Processed data
│   │   ├── 02-sco/             # Gene scores
│   │   ├── 03-sel/             # Gene selections
│   │   └── 04-emb/             # Embeddings
│   └── sim/                  # Simulated data
│       └── ...                 # Same structure as kang/
├── logs/                   # Log files (gitignored)
├── notebooks/              # Analysis notebooks
│   ├── python/               # Python notebooks
│   │   ├── 00-data_*.ipynb
│   │   ├── 01-patches_*.ipynb
│   │   ├── 02-discover_*.ipynb
│   │   └── 03-mutual_info_*.ipynb
│   └── r/                    # R notebooks
│       ├── 00-dat_*.qmd
│       ├── 01-sco_*.qmd
│       ├── 02-sel_*.qmd
│       └── 03-da_*.qmd
├── outs/                   # Output figures and results
│   ├── kang/
│   └── sim/
├── src/                    # Source code
│   ├── discover/             # DISCoVeR implementation
│   ├── python/               # Python utilities
│   └── r/                    # R utilities
├── environment.yml         # Conda environment for Python
├── renv.lock               # R package dependencies
└── README.md
```

## Getting Started

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- R (≥ 4.0)
- Git

### Setting Up the Python Environment

1. **Create and activate the conda environment:**

```bash
# Using conda
conda env create -f environment.yml
conda activate type-state
```

2. **Verify installation:**

```bash
python -c "import scanpy, torch, pyro; print('Python environment ready!')"
```

The Python environment includes:
- `scanpy` for single-cell analysis
- `pytorch` and `pyro` for deep learning
- `scladder` (Patches implementation)
- `scikit-learn`, `numpy`, `pandas` for data processing
- `matplotlib`, `seaborn` for visualization

### Setting Up the R Environment

This project uses `renv` for R package management.

1. **Open R in the project directory and restore packages:**

```r
# Install renv if not already installed
install.packages("renv")

# Restore the R environment
renv::restore()
```

2. **Verify installation:**

```r
library(SingleCellExperiment)
library(scater)
library(tidyverse)
print("R environment ready!")
```

The R environment includes:
- `SingleCellExperiment`, `scater`, `scran` for single-cell analysis
- `tidyverse` for data manipulation and visualization
- Bioconductor packages for genomic analysis
- `ggplot2`, `pheatmap`, `viridis` for visualization

### Alternative: Manual R Setup

If `renv::restore()` encounters issues, you can manually install key packages:

```r
install.packages(c("tidyverse", "BiocManager", "renv"))
BiocManager::install(c("SingleCellExperiment", "scater", "scran", 
                       "splatter", "edgeR"))
```

## Analysis Workflow

The analysis follows a structured pipeline for both Kang (experimental) and simulation datasets:

### 0. Data Retrieval
- Datasets are not included in this repository due to their size
- Datasets, FSS scores, and FSS selections can be obtained by running the snakemake workflow from [https://github.com/HelenaLC/type-state](https://github.com/HelenaLC/type-state)

### 1. Data Preparation (`00-dat_*.qmd` / `00-data_*.ipynb`)
- Load and preprocess raw data
- Quality control and normalization
- Export to appropriate formats

### 2. Model Training and Gene Scoring
- **Patches** (`01-patches_*.ipynb`): Train VAE models with interpretable loadings
- **Scores** (`01-sco_*.qmd`): Load classical feature selection scores and compute Patches-based selection scores

### 3. Gene Selection (`02-sel_*.qmd`)
- Combine scores from multiple methods
- Create gene selection strategies
- Export selections for downstream analysis

### 4. Downstream Analysis (`03-da_*.qmd` / `03-mutual_info_*.ipynb`)
- UMAP visualization
- Clustering evaluation (ARI)
- Mixing (LISI)
- Principal component regression (PCR)
- Mutual information analysis

### 5. DISCoVeR Analysis (`02-discover_*.ipynb`)
- Train DISCoVeR models
- Extract embeddings
- Comparative evaluation
