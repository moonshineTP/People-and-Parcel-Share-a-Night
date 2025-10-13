"""
    Dedicated for online judge systems submission.
    Use inliner.py for inlining modules of this file into one file.
    The final output is submission.py
"""
import sys

from share_a_ride.solvers.algo.bnb import branch_and_bound
from share_a_ride.solvers.algo.greedy import greedy_balanced_solver
from share_a_ride.solvers.algo.greedy import iterative_greedy_balanced_solver
from share_a_ride.problem import ShareARideProblem
from share_a_ride.solution import Solution
from share_a_ride.utils.probgen import generate_instance_coords


def read_instance() -> tuple:
    """Read instance from standard input."""

    N, M, K = map(int, sys.stdin.readline().strip().split())
    q = list(map(int, sys.stdin.readline().split()))
    Q = list(map(int, sys.stdin.readline().split()))
    D = [[0] * (2 * N + 2 * M + 1) for _ in range(2 * N + 2 * M + 1)]
    for r in range(2 * N + 2 * M + 1):
        line = sys.stdin.readline().strip()
        D[r] = list(map(int, line.split()))

    return ShareARideProblem(N, M, K, q, Q, D)

def main():
    prob: ShareARideProblem = generate_instance_coords(
        area=100.0, N = 10, M = 40, K = 5,
        qlmbd=12.0, Qlmbd=18.0,
        seed=42
    )
    prob.stdin_print()

    # prob: ShareARideProblem = read_instance()
    sol: Solution = None
    sol, info1 = iterative_greedy_balanced_solver(
        prob=prob, iterations=1000, time_limit=10.0,
        destroy_ratio=0.5, destroy_steps=6,     # tune these
        rebuild_prob=0.3, rebuild_steps=2,      # tune these
        temperature=1.0,
        seed=42, verbose=True
    )

    print(f"Enumeration info: {info1}")
    sol.pretty_print(verbose=0)


    # sol_list3, info3 = exhaustive_enumerate(
    #     prob, max_solutions=500000,time_limit=10.0, verbose=True
    # )
    # if sol_list3:
    #     sol_list3[0].pretty_print(verbose=1)
    # print("Enumeration info:", info3)
    # print(route_cost_from_sequence(sol_list3[0].routes[0], prob.D, verbose=1))

if __name__ == "__main__":
    main()