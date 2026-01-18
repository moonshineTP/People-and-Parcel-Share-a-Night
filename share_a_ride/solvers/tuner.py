"""
Module for hyperparameter tuning of ride-sharing solvers.
It includes methods to define parameter grids for various solvers.
"""
from typing import Dict, Any, List, Optional
import optuna
from share_a_ride.solvers.classes import Solver, SolverParams, SolverName, SolverMode
from share_a_ride.core.problem import ShareARideProblem




def define_grid(name: SolverName, mode: SolverMode, trial: optuna.trial.Trial) -> Dict[str, Any]:
    """
    Define hyperparameter grid for a given solver.

    Args:
        name (SolverName): The solver name.
        mode (SolverMode): The solver mode.
        trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
        dict: A dictionary of hyperparameters for the solver.
    """
    # Get parameter ranges
    ranges = SolverParams.tune_params(name, mode)
    grid: Dict[str, Any] = {}

    # Sample hyperparameters
    for key, interval in ranges.items():
        if not isinstance(interval, (list, tuple)):
            grid[key] = trial.suggest_int(key, interval, interval, log=True)
        elif isinstance(interval[0], int):
            steps = (interval[1] - interval[0]) // 5 + 1
            grid[key] = trial.suggest_int(key, interval[0], interval[1], step=steps)
        else:
            grid[key] = trial.suggest_float(key, interval[0], interval[1], log=True)

    return grid




def tune_solver(
        problems: List[ShareARideProblem],
        solver_name: SolverName,
        solver_mode: SolverMode,

        n_trials: int,
        n_repeats: int,

        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ) -> Dict[str, Any]:
    """
    Tune a solver on a list of problems.

    Args:
        problems (List[ShareARideProblem]): List of problems to tune on.
        solver_name (SolverName): Name of the solver to tune.
        solver_mode (SolverMode): Mode of the solver.
        n_trials (int): Number of trials for optimization.
        n_repeats (int): Number of repeats for each problem instance.
        study_name (Optional[str]): Name of the study.
        storage (Optional[str]): Storage URL for the study.

    Returns:
        Dict[str, Any]: The best hyperparameters found.
    """
    solver = Solver(solver_name, solver_mode)
    def objective(trial: optuna.trial.Trial) -> float:
        params = define_grid(solver_name, solver_mode, trial)
        total_cost = 0.0

        for pid, problem in enumerate(problems):
            for sid in range(n_repeats):
                solution, _ = solver.run(
                    problem,
                    seed=(209 * pid + 31 * sid + 107),
                    verbose=False,
                    **params
                )

                if solution is not None:
                    print(
                        f"[Problem {pid + 1}] Run completed successfully "
                        f"with max cost: {solution.max_cost}"
                    )
                    total_cost += solution.max_cost
                else:
                    total_cost += 1e9

        return total_cost

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=n_trials)

    return study.best_params




# ================ Playground ================
if __name__ == "__main__":
    from share_a_ride.data.classes import Dataset
    from share_a_ride.data.extractor import extract_dataset

    dts = extract_dataset(Dataset.CMT)
    tune_solver(
        problems=list(dts.values()),
        solver_name=SolverName.ACO,
        solver_mode=SolverMode.HEAVY,
        n_trials=20,
        n_repeats=2,
        study_name="aco_tuning",
    )
