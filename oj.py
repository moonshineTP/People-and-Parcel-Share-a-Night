"""
    Dedicated for online judge systems submission.
    Use inliner.py for inlining modules of this file into one file.
    The final output is submission.py
"""
import sys
import time

from share_a_ride.solvers.algo.greedy import iterative_greedy_balanced_solver
from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution
from share_a_ride.solvers.operator.swap import inter_swap_operator, intra_swap_operator


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
    sol, info1 = iterative_greedy_balanced_solver(
        prob=prob,
        iterations=100000, time_limit=60.0, seed=42, verbose=True,
        destroy_proba=0.5, destroy_steps=min(6, prob.num_nodes // (2 * prob.K) + 1), destroy_T=1.0,
        rebuild_proba=0.25, rebuild_steps=2, rebuild_T=10.0
    )
    assert sol is not None, "No solution found by IG solver."
    if verbose:
        print(f"Initial solution cost: {sol.max_cost:.2f}")
        print()
        print()

    st1 = time.time()
    par = PartialSolution.from_solution(sol)
    new_par, modified, n_relocates = inter_swap_operator(
        partial=par,
        steps=None,
        mode='first',
        seed=100,
        verbose=True
    )

    sol = new_par.to_solution()
    assert sol, "No solution found after int."
    if verbose:
        print(f"Total inter-swap performed: {n_relocates}")
        print(f"Cost after inter-swap: {sol.max_cost:.2f}")
        print(f"Time for inter-swap: {time.time() - st1:.2f} seconds")
        print()
        print()

    st2 = time.time()
    par = PartialSolution.from_solution(sol)
    new_par, modified, n_relocates = intra_swap_operator(
        partial=par,
        steps=None,
        mode='best',
        seed=200,
        verbose=True
    )

    sol = new_par.to_solution()
    assert sol, "No solution found after relocates."
    if verbose:
        print(f"Total relocates performed: {n_relocates}")
        print(f"Cost after relocates: {sol.max_cost:.2f}")
        print(f"Time for relocates: {time.time() - st2:.2f} seconds")
        print()
        print()

    sol.stdin_print(verbose=True)


if __name__ == "__main__":
    main(verbose=True)
