from ladder.scripts import InterpretableWorkflow
import scanpy as sc
import pandas as pd
import pprint

# read in data and preprocess
print("Reading in data and starting to preprocess...")
adata = sc.read_h5ad("../../data/sim/01-pro/t100,s80,b0.h5ad")
adata.X = adata.layers["logcounts"]

sc.pp.highly_variable_genes(adata, n_top_genes=1500)

adata = adata[:, adata.var["highly_variable"]].copy()
adata.X = adata.layers["counts"] # model input should be raw counts (stated in docs)
print(adata)
print("Preprocessing finished.")
print("=======================================================")

# run Patches
num_epochs = [50, 100, 200, 500, 1000, 1500, 2000]
results = {}
workflow = InterpretableWorkflow(adata.copy(), verbose=True, random_seed=42)
factors = ["group_id", "cluster_id"]
workflow.prep_model(factors, batch_key="sample_id", model_type='Patches', model_args={'ld_normalize' : True})

print("=======================================================")

for epoch in num_epochs:
    print(f"Running model for {epoch} epochs...")
    workflow.run_model(max_epochs=epoch, convergence_threshold=1e-5, convergence_window=100)
    workflow.write_embeddings()
    results[str(epoch)] = workflow.evaluate_reconstruction()
    print("=======================================================")

pprint.pprint(results)

# save results to a CSV file
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.to_csv("../../data/sim/02-patches/patches_results.csv")