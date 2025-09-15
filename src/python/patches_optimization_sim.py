import scanpy as sc
import logging
from patches_optuna import optimize_patches


# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info("Running Patches optimization on simulation data...")
logging.info("Loading and preprocessing data...")

# load data
adata = sc.read_h5ad("../../data/sim/01-pro/t100,s80,b0.h5ad")

# full adata
adata_full = adata.copy()

# highly variable gene selection on log-normalized data
adata.X = adata.layers["logcounts"]
sc.pp.highly_variable_genes(adata, n_top_genes=1500)
adata_hvg = adata[:, adata.var["highly_variable"]].copy()

# reset to raw counts for model input (as stated in docs)
adata_full.X = adata_full.layers["counts"]
adata_hvg.X = adata_hvg.layers["counts"]
logging.info(f"Full data shape: {adata_full.shape}, HVG data shape: {adata_hvg.shape}")

logging.info("Finished loading and preprocessing data.")
logging.info("Running Patches model with different hyperparameters and all genes...")

# define parameters
factors = ["cluster_id", "group_id"]
batch_key = "sample_id"
random_seed = 42
convergence_threshold = 1e-4
convergence_window = 50
min_lr = 1e-5
max_lr = 1e-2
epochs = [100, 200, 400, 500]
batch_sizes = [64, 128, 256]
n_trials = 2

# optuna optimization for full data
best_params_all = optimize_patches(
    adata_full,
    factors=factors,
    batch_key=batch_key,
    random_seed=random_seed,
    convergence_threshold=convergence_threshold,
    convergence_window=convergence_window,
    min_lr=min_lr,
    max_lr=max_lr,
    epochs=epochs,
    batch_sizes=batch_sizes,
    n_trials=n_trials,
)

logging.info("Finished running Patches model with different hyperparameters and all genes.")
logging.info(f"Best parameters for full data: {best_params_all}")

logging.info("Running Patches model with different hyperparameters and highly variable genes...")

# optuna optimization for hvg data
best_params_hvg = optimize_patches(
    adata_hvg,
    factors=factors,
    batch_key=batch_key,
    random_seed=random_seed,
    convergence_threshold=convergence_threshold,
    convergence_window=convergence_window,
    min_lr=min_lr,
    max_lr=max_lr,
    epochs=epochs,
    batch_sizes=batch_sizes,
    n_trials=n_trials,
)

logging.info("Finished running Patches model with different hyperparameters and highly variable genes.")
logging.info(f"Best parameters for HVG data: {best_params_hvg}")

logging.info("Finished Patches optimization on simulated data.")
