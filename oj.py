"""
    Dedicated for online judge systems submission.
    Use inliner.py for inlining modules of this file into one file.
    The final output is submission.py
"""
import sys

from share_a_ride.solvers.algo.Algo import AlgoSolver
from share_a_ride.solvers.algo.greedy import iterative_greedy_balanced_solver
from share_a_ride.problem import ShareARideProblem
from share_a_ride.utils.generator import generate_instance_coords


def read_instance() -> tuple:
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

def main():
    prob: ShareARideProblem = read_instance()
    sol, info1 = iterative_greedy_balanced_solver(
        prob=prob, 
        iterations=10000, time_limit=10.0, seed=42, verbose=0,
        destroy_proba=0.5, destroy_steps=min(6, prob.num_nodes // (2 * prob.K) + 1), destroy_T=1.0,
        rebuild_proba=0.3, rebuild_steps=2, rebuild_T=5.0
    )

    # print(f"Enumeration info: {info1}")
    sol.stdin_print(verbose=0)


    # sol_list3, info3 = exhaustive_enumerate(
    #     prob, max_solutions=500000,time_limit=10.0, verbose=True
    # )
    # if sol_list3:
    #     sol_list3[0].pretty_print(verbose=1)
    # print("Enumeration info:", info3)
    # print(route_cost_from_sequence(sol_list3[0].routes[0], prob.D, verbose=1))


if __name__ == "__main__":
    main()
