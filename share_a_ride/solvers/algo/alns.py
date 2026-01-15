"""
Module using an Adaptive Large Neighborhood Search (ALNS) algorithm
This is the baseline, so we use the alns package and not our own implementation.
"""
import time
from typing import Optional, List, Dict, Any, Tuple

from numpy import random as rnd
from numpy import flip
try:
    from alns.ALNS import ALNS
    from alns.accept import HillClimbing
    from alns.select import RouletteWheel
    from alns.stop import MaxRuntime
except ImportError:
    from alns.ALNS import ALNS
    from alns.accept.HillClimbing import HillClimbing
    from alns.select.RouletteWheel import RouletteWheel
    from alns.stop import MaxRuntime

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, Solution, PartialSolutionSwarm
from share_a_ride.solvers.algo.greedy import iterative_greedy_solver
from share_a_ride.solvers.algo.beam import beam_solver
from share_a_ride.solvers.algo.greedy import iterative_greedy_solver
from share_a_ride.solvers.algo.astar import astar_solver
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

    repaired_sol, _info = beam_solver(
        partial.problem,
        PartialSolutionSwarm([partial]),
        r_intra=0.95,
        f_intra=0.1,
        r_inter=0.99,
        seed=int(rng.integers(0, 1000000))
    )
    assert repaired_sol
    repaired_partial = PartialSolution.from_solution(repaired_sol)

    return ALNSPartialSolution(repaired_partial)


def _repair_2(
        state,
        rng: rnd.Generator,
    ):
    working = state.copy()
    partial = working.partial

    repaired_sol, _info = iterative_greedy_solver(
        partial.problem,
        partial,
        iterations=3000,
        time_limit=5.0,
        seed=int(rng.integers(0, 1000000))
    )
    assert repaired_sol
    repaired_partial = PartialSolution.from_solution(repaired_sol)

    return ALNSPartialSolution(repaired_partial)


def _repair_3(
        state,
        rng: rnd.Generator,
    ):
    working = state.copy()
    partial = working.partial

    repaired_sol, _info = astar_solver(
        partial.problem,
        partial,
        width=3,
        cutoff_depth=6,
        cutoff_size=300,
        time_limit=5.0,
        seed=int(rng.integers(0, 1000000))
    )
    assert repaired_sol
    repaired_partial = PartialSolution.from_solution(repaired_sol)

    return ALNSPartialSolution(repaired_partial)



# ================ ALNS Execution Function ================
def alns_solver(        # pylint: disable=W0102
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        scores: Optional[List[float]] = None,
        decay: float = 0.8,

        time_limit: float = 30.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Adaptive Large Neighborhood Search (ALNS) solver for the Share-a-Ride problem.
    """
    start = time.time()

    if scores is None:
        scores = [10, 4, 1, 0]

    # //// Generate initial solution
    if partial is None:
        if verbose:
            print("Generating initial solution...")

        sol, solver_stats = iterative_greedy_solver(
            problem,
            iterations=3000,
            time_limit=3.0,
            seed=seed,
        )
        assert sol is not None and sol.is_valid(), "Initial solution is not valid."
        partial  = PartialSolution.from_solution(sol)

    init_cost = partial.max_cost
    init_sol = ALNSPartialSolution(partial)

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
    alns.add_repair_operator(_repair_3)   # type: ignore

    # Configure alns components
    select = RouletteWheel(scores, decay=decay, num_destroy=3, num_repair=3)
    accept = HillClimbing()
    stop = MaxRuntime(min(time_limit * 0.9, time_limit - 10))


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

    # Retrieve statistics
    alns_stats = result.statistics
    status = "done"
    if alns_stats.total_runtime >= time_limit:
        status = "overtime"

    # Build solver stats
    elapsed = time.time() - start

    solver_stats: Dict[str, Any] = {
        "destroy_operator_counts": dict(alns_stats.destroy_operator_counts),
        "repair_operator_counts": dict(alns_stats.repair_operator_counts),
        "total_runtime": alns_stats.total_runtime,
        "objectives": flip(alns_stats.objectives).tolist()[:5],
        "status": status,
        "time": elapsed
    }

    # Logging
    if verbose:
        print()
        print("[ALNS] Completed")
        for key, value in solver_stats.items():
            print(f" - {key}: {value}")
        print("------------------------------")
        print()

    return sol, solver_stats




# ================ Playground ================
if __name__ == "__main__":
    from share_a_ride.solvers.algo.utils import test_problem

    solution, info = alns_solver(
        test_problem,
        time_limit=960,
        seed=84,
        verbose=True,
    )
