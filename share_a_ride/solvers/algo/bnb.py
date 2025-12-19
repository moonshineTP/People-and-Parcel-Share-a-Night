"""
Branch-and-bound solver for Share-a-Ride problem.
"""
import time
import heapq
from typing import List, Optional, Tuple, Dict, Any

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, Solution
from share_a_ride.solvers.algo.astar import _default_pred_function
from share_a_ride.solvers.algo.utils import (
    enumerate_actions_greedily, apply_general_action
)




# //// Admissive lower-bounds for branch-and-bound
def _admissive_lb(partial: PartialSolution) -> float:
    """
    Admissive lower bound function for branch-and-bound.
    Currently uses the default A* predicate function.
    """
    return partial.max_cost + _default_pred_function(partial)




# //// Bnb APIs
def bnb_enumerator(
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        max_solutions: int = 100000,
        imcumbent: Optional[int] = None,
        time_limit: float = 30.0,
        verbose: bool = False,
    ) -> Tuple[List[Solution], Dict[str, Any]]:

    """
    Branch-and-bound enumeration of optimal solution for the given problem.
    Add a cache mechanism to enumerate action more greedily and prune by
    admissive lower-bound.

    Returns (best_solution, info) tuple where
    - best_solution: the optimal Solution object found (or None if timeout)
    """
    start = time.time()
    end = start + time_limit
    if not partial:
        partial = PartialSolution(problem)
    if not imcumbent:
        imcumbent = 10**18  # effectively unlimited


    # //// Backtracking
    # Terminal case
    if not partial.is_pending():
        sol = partial.to_solution()
        assert sol, "Generation of solution from terminal partial failed"
        return [sol], {
            "num_actions": 0,
            "num_solutions": 1,
            "best_cost": sol.max_cost,
            "time": time.time() - start,
            "status": "done"
        }

    # Expansion case
    solutions_heap: List[Tuple[int, int, Solution]] = []  # in ascending order of cost
    solutions_count = 0
    actions_done = 0
    timeout = False

    for action in enumerate_actions_greedily(partial):
        actions_done += 1
        if time.time() > end:
            timeout = True
            break

        # Apply action
        partial_next = partial.copy()
        apply_general_action(partial_next, action)

        # Check admissive lower-bound
        if _admissive_lb(partial_next) >= imcumbent:
            continue    # prune by admissive lower-bound

        # Recursive call
        sols_next, info_sub = bnb_enumerator(
            problem,
            partial_next,
            max_solutions=max_solutions - solutions_count,
            imcumbent=imcumbent,
            time_limit=end - time.time(),
            verbose=False
        )

        # Insertion sort for solutions based on cost
        for sol_next in sols_next:
            if sol_next.max_cost < imcumbent:
                imcumbent = sol_next.max_cost

            heapq.heappush(solutions_heap, (sol_next.max_cost, solutions_count, sol_next))
            solutions_count += 1

            if solutions_count >= max_solutions:
                break

        # Update stats
        actions_done += info_sub["num_actions"]

        if solutions_count >= max_solutions:
            break

    # Extract solutions from heap
    solutions = [item[2] for item in solutions_heap]

    # Elapse
    elapsed = time.time() - start
    stats = {
        "num_actions": actions_done,
        "num_solutions": solutions_count,
        "best_cost": solutions[0].max_cost if solutions else 'N/A',
        "time": elapsed,
        "status": "timeout" if timeout else "done"
    }

    # logging
    if verbose:
        print(
            f"[BnB] Completed. "
            f"Actions: {stats['num_actions']}, "
            f"Solutions: {stats['num_solutions']}, Time: {stats['time']:.2f}s, "
            f"Best Cost: {stats['best_cost']}, "
            f"Status: {stats['status']}"
        )

    # return results
    return solutions, stats


def bnb_solver(
        problem: ShareARideProblem,
        partial: PartialSolution,
        time_limit: float = 30.0,
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Branch-and-bound solver that returns only the best solution found.
    Uses bnb_enumerator internally.
    """
    solutions, info = bnb_enumerator(    # pylint: disable=W0621
        problem,
        partial,
        max_solutions=1,
        time_limit=time_limit,
        verbose=verbose
    )

    if solutions:
        best_solution = solutions[0]
        if verbose:
            print(
                f"[BnB Solver] Best solution found. "
                f"Cost: {best_solution.max_cost}, "
                f"Time taken: {info['time']:.2f}s"
            )
    else:
        best_solution = None
        if verbose:
            print("[BnB Solver] No feasible solution found.")

    return best_solution, info




# ================ Playground ================
if __name__ == "__main__":
    from share_a_ride.solvers.algo.greedy import iterative_greedy_solver
    from share_a_ride.solvers.algo.utils import bnb_problem


    imcumbent_sol, _ = iterative_greedy_solver(
        bnb_problem,
        iterations=1000,
        time_limit=10.0,
        verbose=True
    )
    sols, info = bnb_enumerator(
        bnb_problem,
        max_solutions=1000,
        imcumbent = imcumbent_sol.max_cost if imcumbent_sol else None,
        time_limit=30.0,
        verbose=True
    )
