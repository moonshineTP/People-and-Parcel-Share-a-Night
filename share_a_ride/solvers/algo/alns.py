"""
Module using an Adaptive Large Neighborhood Search (ALNS) algorithm
This is the baseline, so we use the alns package and not our own implementation.
"""

from typing import Optional, List, Dict, Any, Tuple


import numpy.random as rnd
try:
    from alns import ALNS
    from alns.accept import HillClimbing
    from alns.select import RouletteWheel
    from alns.stop import MaxRuntime
except ImportError:
    from alns.ALNS import ALNS
    from alns.accept.HillClimbing import HillClimbing
    from alns.select.RouletteWheel import RouletteWheel
    from alns.stop import MaxRuntime

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, Solution
from share_a_ride.solvers.algo.greedy import (
    greedy_solver
)
from share_a_ride.solvers.algo.astar import astar_solver
from share_a_ride.solvers.algo.mcts import mcts_solver
from share_a_ride.solvers.operator.destroy import destroy_operator




# ================ ALNSPartialSolution ================
class ALNSPartialSolution:
    """
    A wrapper around PartialSolution to be used in ALNS.
    """
    def __init__(self, partial: PartialSolution):
        self.partial = partial

        assert self.partial.is_valid(), "Initial partial solution is not valid."

    def copy(self) -> 'ALNSPartialSolution':
        """
        Create a deep copy of the wrapper.
        """
        return ALNSPartialSolution(self.partial.copy())

    def objective(self) -> float:
        """
        Return the max cost of the partial solution for minimizing it.
        """
        return float(self.partial.max_cost)




# ================ Destroy Operator ================
def _destroy_1(
        state,
        rng: rnd.Generator,
    ):
    working = state.copy()
    partial = working.partial
    assert partial.is_completed(), "Current solution must be complete before destruction."

    destroyed_partial, _, _ = destroy_operator(
        sol=partial,
        destroy_proba=0.2,
        destroy_steps=5,
        seed=int(rng.integers(0, 1000000))
    )

    return ALNSPartialSolution(destroyed_partial)


def _destroy_2(
        state,
        rng: rnd.Generator,
    ):
    working = state.copy()
    partial = working.partial
    assert partial.is_completed(), "Current solution must be complete before destruction."

    destroyed_partial, _, _ = destroy_operator(
        sol=partial,
        destroy_proba=0.3,
        destroy_steps=10,
        seed=int(rng.integers(0, 1000000))
    )

    return ALNSPartialSolution(destroyed_partial)


def _destroy_3(
        state,
        rng: rnd.Generator,
    ):
    working = state.copy()
    partial = working.partial
    assert partial.is_completed(), "Current solution must be complete before destruction."

    destroyed_partial, _, _ = destroy_operator(
        sol=partial,
        destroy_proba=0.4,
        destroy_steps=15,
        seed=int(rng.integers(0, 1000000))
    )

    return ALNSPartialSolution(destroyed_partial)


# ================ Repair Operator ================
def _repair_1(
        state,
        rng: rnd.Generator,
    ):
    working = state.copy()
    partial = working.partial

    repaired_sol, _info = astar_solver(
        partial.problem,
        partial,
        seed=int(rng.integers(0, 1000000))
    )
    assert repaired_sol is not None
    repaired_partial = PartialSolution.from_solution(repaired_sol)

    return ALNSPartialSolution(repaired_partial)


def _repair_2(
        state,
        rng: rnd.Generator,
    ):
    working = state.copy()
    partial = working.partial

    repaired_sol, _info = mcts_solver(
        partial.problem,
        partial,
        seed=int(rng.integers(0, 1000000))
    )
    assert repaired_sol is not None
    repaired_partial = PartialSolution.from_solution(repaired_sol)

    return ALNSPartialSolution(repaired_partial)




# ================ ALNS Execution Function ================
def alns_solver(        # pylint: disable=W0102
        problem: ShareARideProblem,
        initial_solution: Optional[PartialSolution] = None,
        scores: List[float] = [10, 4, 1, 0],
        decay: float = 0.8,

        time_limit: float = 30.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:

    """
    Adaptive Large Neighborhood Search (ALNS) solver for the Share-a-Ride problem.
    """

    # //// Generate initial solution
    if initial_solution is None:
        if verbose:
            print("Generating initial solution...")

        sol, solver_stats = greedy_solver(
            problem,
            # iterations=1000,
            # time_limit=5.0,
            # seed=28*seed if seed is not None else None,
        )
        assert sol is not None and sol.is_valid(), "Initial solution is not valid."
        initial_solution  = PartialSolution.from_solution(sol)

    init_cost = initial_solution.max_cost
    init_sol = ALNSPartialSolution(initial_solution)

    if verbose:
        print("Initial max cost:", init_cost)


    # //////// Create and configure ALNS ////////
    # Build alns model
    rng = rnd.default_rng(seed)
    alns = ALNS(rng)

    # Configure alns operators
    alns.add_destroy_operator(_destroy_1)   # type: ignore
    alns.add_destroy_operator(_destroy_2)   # type: ignore
    alns.add_destroy_operator(_destroy_3)   # type: ignore
    alns.add_repair_operator(_repair_1)   # type: ignore
    alns.add_repair_operator(_repair_2)   # type: ignore

    # Configure alns components
    select = RouletteWheel(scores, decay=decay, num_destroy=3, num_repair=2)
    accept = HillClimbing()
    stop = MaxRuntime(time_limit)


    # ///////// Run ALNS ////////
    if verbose:
        print("Starting ALNS...")
    result = alns.iterate(init_sol, select, accept, stop)
    if verbose:
        print("ALNS finished.")


    # ///////// Retrieve Results ////////
    # Retrieve the final solution
    best: ALNSPartialSolution = result.best_state       # type: ignore
    sol = best.partial.to_solution()
    assert sol is not None and sol.is_valid(), "Final solution is not valid."

    if verbose:
        print()
        print("Final solution:")
        sol.stdin_print(verbose=True)
        print()

    # Retrieve statistics
    alns_stats = result.statistics
    solver_stats: Dict[str, Any] = {
        "destroy_operator_counts": dict(alns_stats.destroy_operator_counts),
        "repair_operator_counts": dict(alns_stats.repair_operator_counts),
        "total_runtime": alns_stats.total_runtime,
        "objectives": alns_stats.objectives,
    }

    # Logging
    if verbose:
        print()
        print("[ALNS] Completed")
        print("ALNS Statistics:")
        print(solver_stats)
        print("------------------------------")
        print()

    return sol, solver_stats




# ================ Playground ================
if __name__ == "__main__":
    from share_a_ride.solvers.algo.utils import test_problem

    # def objective(trial: optuna.Trial) -> float:
    #     """
    #     Objective function for Optuna hyperparameter tuning.
    #     Runs ACO with suggested hyperparameters and returns average cost
    #     over multiple runs.
    #     """
    #     # ACO hyperparameters tuning
    #     scores = [
    #         trial.suggest_float("score_1", 8.0, 16.0, step = 0.5),
    #         trial.suggest_float("score_2", 2.0, 4.0, step=0.2),
    #         trial.suggest_float("score_3", 0.5, 2.0, step=0.1),
    #         0.0
    #     ]
    #     decay = trial.suggest_float("decay", 0.7, 0.95, step=0.05)

    #     costs = []
    #     for i in range(3):
    #         sol, _info = alns_solver(
    #             main_prob,
    #             scores=scores,
    #             decay=decay,
    #             time_limit=40.0,
    #             seed=123 + 321 * i,
    #             verbose=False,
    #         )
    #         assert sol is not None and sol.is_valid(), "Obtained solution is not valid."

    #         costs.append(sol.max_cost)

    #     return sum(costs) / len(costs) if costs else float("inf")


    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=30)

    # print("Best trial:")
    # best = study.best_trial
    # print(f"  Value: {best.value}")
    # print("  Params: ")
    # for key, value in best.params.items():
    #     print(f"    {key}: {value}")

    # Solve with ALNS
    solution, info = alns_solver(
        test_problem,
        time_limit=60.0,
        seed=42,
        verbose=True,
    )
