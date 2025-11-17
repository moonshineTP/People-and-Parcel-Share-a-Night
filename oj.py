"""
    Dedicated for online judge systems submission.
    Use inliner.py for inlining modules of this file into one file.
    The final output is submission.py
"""
import sys
import time
import random

from typing import List, Optional

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import Solution, PartialSolution
from share_a_ride.solvers.operator.relocate import relocate_operator
from share_a_ride.solvers.algo.greedy import greedy_balanced_solver
from share_a_ride.solvers.algo.beam import beam_search_solver, parsol_scorer
from share_a_ride.solvers.algo.mcts import (
    mcts_solver,
    PartialSolutionSwarm,
    Action,
)
from share_a_ride.solvers.utils.sampler import sample_from_weight
from share_a_ride.solvers.utils.weighter import softmax_weighter


def read_instance() -> ShareARideProblem:
    """
    Read instance from standard input.
    """

    N, M, K = map(int, sys.stdin.readline().strip().split())
    q = list(map(int, sys.stdin.readline().split()))
    Q = list(map(int, sys.stdin.readline().split()))
    D = [[0] * (2 * N + 2 * M + 1) for _ in range(2 * N + 2 * M + 1)]
    for r in range(2 * N + 2 * M + 1):
        line = sys.stdin.readline().strip()
        D[r] = list(map(int, line.split()))

    return ShareARideProblem(N, M, K, q, Q, D)


def main(verbose: bool = False):
    prob: ShareARideProblem = read_instance()


    # //////// MCTS Solver ////////
    def v_func(
        parsol: PartialSolution,
    ) -> float:
        """Value function provided by negative `parsol_scorer` in beam search"""
        cost = parsol_scorer(parsol)
        return -cost    # IMPORTANT: Use negative cost as value for minimization problems


    def stochastic_select_policy(
        _ps: PartialSolution,
        actions: List[Action],
    ) -> Optional[Action]:
        """Sample an action using a low-temperature softmax over incremental cost."""
        rng = random.Random()
        if not actions:
            return None
        increments = [float(action[3]) for action in actions]
        weights = softmax_weighter(increments, T=0.1)
        chosen_idx = sample_from_weight(rng, weights)
        return actions[chosen_idx]


    def sim_policy(
        ps: PartialSolution,
    ) -> Optional[PartialSolution]:
        """Greedy balanced simulation policy"""
        sim_solution, _ = greedy_balanced_solver(
            ps.problem,
            premature_routes=[r.copy() for r in ps.routes],
            verbose=False,
        )
        return ps if sim_solution is None else PartialSolution.from_solution(sim_solution)


    def def_policy(
        ps: PartialSolution,
    ) -> Optional[Solution]:
        """
        Beam search defense policy.
        """
        beam_solution, _ = beam_search_solver(
            ps.problem,
            cost_function=parsol_scorer,
            initial=PartialSolutionSwarm([ps]),
        )

        return beam_solution


    sol, _ = mcts_solver(
        problem=prob,
        partial=None,
        value_function=v_func,
        selection_policy=stochastic_select_policy,
        simulation_policy=sim_policy,
        defense_policy=def_policy,

        width=3,
        uct_c=5.0,
        max_iters=100000,

        seed=42,
        time_limit=200.0,
        verbose=verbose,
    )

    assert sol, "No solution found by MCTS."
    if verbose:
        print()
        print(f"Cost after MCTS: {sol.max_cost:.2f}")
        print("===============================")


    # //////// Post-processing improvement ////////
    st1 = time.time()
    par = PartialSolution.from_solution(sol)
    new_par, modified, n_relocates = relocate_operator(
        partial=par,
        steps=None,
        mode='first',
        seed=111,
        verbose=verbose
    )
    sol = new_par.to_solution()
    assert sol, "No solution found after relocate."
    if verbose:
        print()
        print(f"Total relocate performed: {n_relocates}")
        print(f"Cost after relocate: {sol.max_cost:.2f}")
        print(f"Time for relocate: {time.time() - st1:.2f} seconds")
        print("===============================")

    sol.stdin_print(verbose=verbose)


if __name__ == "__main__":
    main(verbose=False)
