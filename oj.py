"""
    Dedicated for online judge systems submission.
    Use inliner.py for inlining modules of this file into one file.
    The final output is submission.py
"""
import sys
from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.solvers.algo.greedy import iterative_greedy_solver
# from share_a_ride.solvers.algo.astar import astar_solver
# from share_a_ride.solvers.algo.aco import aco_solver
# from share_a_ride.solvers.algo.mcts import mcts_solver
# from share_a_ride.solvers.algo.beam import beam_solver



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

    problem: ShareARideProblem = read_instance()
    solution, _info = iterative_greedy_solver(problem, verbose=verbose)
    assert solution is not None, "No solution found"
    solution.stdin_print(verbose=verbose)




if __name__ == "__main__":
    main(verbose=True)
