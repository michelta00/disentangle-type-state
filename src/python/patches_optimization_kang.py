import scanpy as sc
import logging
from patches_optuna import optimize_patches


# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info("Running Patches optimization on Kang18 data...")
logging.info("Loading and preprocessing data...")

# load data
adata = sc.read_h5ad("../../data/kang/01-pro/Kang18.h5ad")
adata.layers["counts"] = adata.X.copy()

logging.info(f"Data shape: {adata.shape}")

logging.info("Finished loading and preprocessing data.")

# define parameters
factors_clu_con = ["cluster_id", "group_id"]
factors_con = ["group_id"]
batch_key = "sample_id"
random_seed = 42
convergence_threshold = 1e-4
convergence_window = 50
min_lr = 1e-4
max_lr = 1e-2
epochs = [100, 500, 1000, 1500, 2000]
batch_sizes = [32, 64, 128, 256]
n_trials = 20

logging.info("Running Patches model with different hyperparameters and conditions only...")

# optuna optimization for full data
best_params_con = optimize_patches(
    adata,
    factors=factors_con,
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

logging.info("Finished running Patches model with different hyperparameters and conditions only.")
logging.info(f"Best parameters for conditions only: {best_params_con}")

logging.info("Running Patches model with different hyperparameters and groups and conditions...")

# optuna optimization for hvg data
best_params_clu_con = optimize_patches(
    adata,
    factors=factors_clu_con,
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

logging.info("Finished running Patches model with different hyperparameters and groups and conditions.")
logging.info(f"Best parameters for groups and conditions: {best_params_clu_con}")

logging.info("Finished Patches optimization on Kang18 data.")
