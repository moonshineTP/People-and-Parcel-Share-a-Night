"""
    Dedicated for online judge systems submission.
    Use inliner.py for inlining modules of this file into one file.
    The final output is submission.py
"""
import sys

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.solvers.algo.aco import aco_solver


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
    """
    Main execution function.
    """
    prob: ShareARideProblem = read_instance()

    # //////// Initial solution ////////
    sol, _ = aco_solver(
        prob,
        cutoff=10,
        num_ants=(
            500 if prob.num_nodes <= 100
            else 150 if prob.num_nodes <= 250
            else 50 if prob.num_nodes <= 500
            else 25 if prob.num_nodes <= 1000
            else 10
        ),
        runs=(
            100 if prob.num_nodes <= 100
            else 75 if prob.num_nodes <= 250
            else 50 if prob.num_nodes <= 500
            else 20 if prob.num_nodes <= 1000
            else 10
        ),
        width=(
            10 if prob.num_nodes <= 100
            else 8 if prob.num_nodes <= 250
            else 6 if prob.num_nodes <= 500
            else 4 if prob.num_nodes <= 1000
            else 2
        ),

        seed=42,
        time_limit=250.0,
        verbose=verbose,
    )

    assert sol, "No solution found by ACO."

    sol.stdin_print(verbose=verbose)


if __name__ == "__main__":
    main(verbose=True)
