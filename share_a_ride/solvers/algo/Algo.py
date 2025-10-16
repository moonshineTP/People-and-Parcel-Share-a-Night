from typing import Callable, Dict, Any, List, Optional, Tuple

from share_a_ride.problem import ShareARideProblem
from share_a_ride.solution import Solution


algo_name = {
    # Exact
    "exhaustive_solver" : "Exhaustive",
    "branch_and_bound_solver" :"Branch and Bound",
    "milp_solver" : "MILP",
    "cp_solver" : "CP",

    # Metaheuristics
    "greedy_balanced_solver" : "Greedy Balanced",
    "iterative_greedy_balanced_solver" : "Iterative Greedy Balanced",
}


def algo_name_router(solver: Callable) -> str:
    """
    Return the name of the algorithm-based solver.
    """
    try:
        return algo_name[solver.__name__]
    except KeyError:
        raise ValueError(f"Unknown solver: {solver.__name__}")


class AlgoSolver():
    """
    Container for algorithm-based solvers in the share_a_ride problem.
    """
    def __init__(
            self, problem: Optional[ShareARideProblem] = None,
            algo: Callable[..., Tuple[Optional[Solution], Dict[str, Any]]] = None,
            args: Dict[str, Any] = None,
            hyperparams: Dict[str, Any] = None,
        ):
        """
        Initialize the solver with problem instance and common parameters.
        Params:
            - problem: ShareARideProblem instance to solve
            - algo: Callable implementing the solving algorithm
            - args: Additional arguments for the algorithm
                (e.g. time_limit, verbose, seed)
            - hyperparams: Dictionary of algorithm-specific hyperparameters
                (e.g. T, steps, ratio, policy model, etc.)
        Optionals:
            - desc: Description of the solver
            - hyperparams: Dictionary of algorithms hyperparameters (if applicable)
        """
        # Solver state
        self.algo = algo
        self.args = args or {}
        self.hyperparams = hyperparams or {}

        # Metadata
        self.name = algo_name_router(algo)
        self.desc = f"{self.name} Solver"


    def solve(
            self,
            problem: ShareARideProblem
        ) -> Tuple[Optional[Solution], Dict[str, Any]]:
        """
        Main solving method to execute the solver.
        Params:
            - problem: ShareARideProblem instance to solve
        Returns: (solution, info): tuple where
        - solution: Best Solution object found (or None if none found)
        - info: Dictionary with statistics including:
            + elapsed_time: Total time taken
            + status: "done" or "timeout"
            + solver-specific metrics
        """
        sol, info = self.algo(problem, **self.args, **self.hyperparams)

        return sol, info


    def tune(
            self,
            problem: ShareARideProblem,
            n_trials: int,
            lb_hyperparams: Dict[str, Any],
            ub_hyperparams: Dict[str, Any],
        ) -> Dict[str, Any]:
        """
        Hyperparameter tuning method to be implemented by each solver.

        Params:
            - problems: List of ShareARideProblem instances for tuning              
            - n_trials: Number of tuning trials to perform
            - lb_hyperparams: Lower bounds for hyperparameters. 
                If the hyperparameter is non-numeric, this should be a list of possible values.
            - ub_hyperparams: Upper bounds for hyperparameters.
                If the hyperparameter is non-numeric, this should be a list of possible values.

        Returns:
            best_hyperparams: Dictionary of best found hyperparameters
        """

        try:
            import optuna
        except ImportError as exc:
            raise ImportError("Optuna is required for hyperparameter tuning." \
                                " Please install it via 'pip install optuna'.") from exc


        def objective(trial: optuna.Trial) -> float:
            # Suggest hyperparameters based on init_hyperparams
            suggested_hyperparams = {}
            for key, lb_val in lb_hyperparams.items():
                if isinstance(lb_val, int):
                    suggested_hyperparams[key] = \
                        trial.suggest_int(key, lb_val, ub_hyperparams[key])
                elif isinstance(lb_val, float):
                    suggested_hyperparams[key] = \
                        trial.suggest_float(key, lb_val, ub_hyperparams[key])
                elif isinstance(lb_val, bool):
                    suggested_hyperparams[key] = \
                        trial.suggest_categorical(key, [True, False])
                else:   # Keep non-numeric values as is
                    suggested_hyperparams[key] = lb_val

            # Run solver with suggested hyperparameters
            sol, info = self.algo(problem, **self.args, **suggested_hyperparams)

            # Return objective value (max_cost to minimize)
            if sol is None:
                return float('inf')
            return sol.max_cost

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        return study.best_params


    def save(self, filepath: str) -> None:
        """
        Save the solver configuration to a file.
        
        Params:
            - filepath: Path to save the solver configuration
        """
        pass


    def load(self, filepath: str) -> None:
        """
        Load the solver configuration from a file.
        
        Params:
            - filepath: Path to load the solver configuration from
        """
        pass


class HybridAlgoSolver():
    pass
