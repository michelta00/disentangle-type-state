from ladder.scripts import InterpretableWorkflow
import scanpy as sc
import pandas as pd
import pprint
import optuna

print("\n Running Patches optimization on simulated data...\n")

print("\n Loading and preprocessing data...\n")

# read in data
adata = sc.read_h5ad("../../data/sim/01-pro/t100,s80,b0.h5ad")

# optional: highly variable gene selection on log-normalized data
# adata.X = adata.layers["logcounts"]
# sc.pp.highly_variable_genes(adata, n_top_genes=1500)
# adata = adata[:, adata.var["highly_variable"]].copy()

# reset to raw counts for model input (as stated in docs)
adata.X = adata.layers["counts"]
print(adata)

print("\n Finished loading and preprocessing data.\n")

print("\n Running Patches model with different hyperparameters...\n")

# optuna optimization
def objective(trial):
    # suggest hyperparameters
    epochs = trial.suggest_categorical("epochs", [50, 100, 200, 500, 1000, 1500, 2000])
    # epochs = trial.suggest_categorical("epochs", [1, 2, 5, 10])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # setup workflow
    workflow = InterpretableWorkflow(adata.copy(), verbose=True, random_seed=42)
    factors = ["group_id", "cluster_id"]
    
    # run Patches
    workflow.prep_model(factors, batch_key="sample_id", minibatch_size=batch_size, model_type='Patches', model_args={'ld_normalize' : True})
    workflow.run_model(max_epochs=epochs, convergence_threshold=1e-5, convergence_window=50)
    workflow.write_embeddings()

    # evaluate model
    scores = workflow.evaluate_reconstruction()
    score = scores["Profile Correlation"][0]
    return score

study = optuna.create_study(direction="maximize")  
study.optimize(objective, n_trials=20)

print("\n Finished running Patches model with different hyperparameters.\n")

print(study.best_params)

print("\n Finished Patches optimization on simulated data.\n")
