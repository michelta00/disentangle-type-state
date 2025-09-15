from ladder.scripts import InterpretableWorkflow
import optuna
import scanpy as sc
import logging


# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class Objective:
    def __init__(
        self,
        adata,
        factors,
        batch_key,
        random_seed,
        convergence_threshold,
        convergence_window,
        min_lr,
        max_lr,
        epochs,
        batch_sizes,
    ):
        """
        Objective class for Optuna hyperparameter optimization of Patches model.
        
        Parameters:
        ----------
        adata : AnnData
            The annotated data matrix.
        factors : list
            List of condition classes.
        batch_key : str
            Key for batch information in adata.obs.
        random_seed : int
            Random seed for reproducibility.
        convergence_threshold : float
            Threshold for convergence.
        convergence_window : int
            Window size for convergence checking.
        min_lr : float
            Minimum learning rate for optimization.
        max_lr : float
            Maximum learning rate for optimization.
        epochs : list
            List of epoch options for optimization.
        batch_sizes : list
            List of batch size options for optimization.
        """
        self.adata = adata
        self.factors = factors
        self.batch_key = batch_key
        self.random_seed = random_seed
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.epochs = epochs
        self.batch_sizes = batch_sizes

    def __call__(self, trial):
        """
        Objective function for Optuna optimization. Runs the Patches model with suggested
        hyperparameters and evaluates its performance.

        Parameters:
        ----------
        trial : optuna.trial.Trial
            A trial object for suggesting hyperparameters.

        Returns:
        -------
        score : float
            The evaluation score (Profile Correlation) of the trained model.
        """
        # suggest hyperparameters
        lr = trial.suggest_float("lr", self.min_lr, self.max_lr, log=True)
        epochs = trial.suggest_categorical("epochs", self.epochs)
        batch_size = trial.suggest_categorical("batch_size", self.batch_sizes)

        # setup workflow
        logging.info(f"Running trial with lr: {lr}, epochs: {epochs}, batch_size: {batch_size}, data shape: {self.adata.shape}")
        workflow = InterpretableWorkflow(
            self.adata.copy(), 
            verbose=True, 
            random_seed=self.random_seed
        )

        # run Patches
        workflow.prep_model(
            self.factors,
            batch_key=self.batch_key,
            minibatch_size=batch_size,
            model_type="Patches",
            model_args={"ld_normalize": True},
            optim_args={"lr": lr},
        )
        workflow.run_model(
            max_epochs=epochs, 
            convergence_threshold=self.convergence_threshold, 
            convergence_window=self.convergence_window
        )

        # evaluate model
        scores = workflow.evaluate_reconstruction()
        score = scores["Profile Correlation"][0]
        return score
    
def optimize_patches(
    adata,
    factors,
    batch_key,
    random_seed,
    convergence_threshold,
    convergence_window,
    min_lr,
    max_lr,
    epochs,
    batch_sizes,
    n_trials,
):
    """
    Optimize Patches model using Optuna.

    Parameters:
    ----------
    adata : AnnData
        The annotated data matrix.
    factors : list
        List of condition classes.
    batch_key : str
        Key for batch information in adata.obs.
    random_seed : int
        Random seed for reproducibility.
    convergence_threshold : float
        Threshold for convergence.
    convergence_window : int
        Window size for convergence checking.
    min_lr : float
        Minimum learning rate for optimization.
    max_lr : float
        Maximum learning rate for optimization.
    epochs : list
        List of epoch options for optimization.
    batch_sizes : list
        List of batch size options for optimization.
    n_trials : int
        Number of trials for optimization.

    Returns:
    -------
    best_params : dict
        Best hyperparameters found by Optuna.
    """
    objective = Objective(
        adata,
        factors,
        batch_key,
        random_seed,
        convergence_threshold,
        convergence_window,
        min_lr,
        max_lr,
        epochs,
        batch_sizes,
    )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params
