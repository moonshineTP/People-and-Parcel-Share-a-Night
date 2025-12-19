"""
Exhaustive search solver for the Share-a-Ride problem.
"""
import time
import heapq
from typing import List, Tuple, Any, Dict, Optional

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import Solution, PartialSolution
from share_a_ride.solvers.algo.utils import enumerate_actions_greedily, apply_general_action



def exhaust_enumerator(
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        max_solutions: int = 100000,
        imcumbent: Optional[int] = None,
        time_limit: float = 30.0,
        verbose: bool = False
    ) -> Tuple[List[Solution], Dict[str, Any]]:
    """
    Exhaustive enumeration of all feasible solutions that is the descendant of
    the given partial solution.
    
    Use PartialSolution APIs to enumerate and maneuver the search space.

    Returns (solutions, info) tuple where
    - solutions: list of valid Solution objects found (up to max_solutions)
    - info: dictionary with statistics
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
            "time": time.time() - start,
            "status": "done"
        }

    # Expansion case
    solutions_heap: List[Tuple[int, int, Solution]] = []  # in ascending order of cost
    solutions_count = 0
    actions_done = 0
    timeout = False
    for action in enumerate_actions_greedily(partial):
        if time.time() > end:
            timeout = True
            break

        # Apply action
        partial_next = partial.copy()
        apply_general_action(partial_next, action)
        actions_done += 1

        # Recursive call
        sols_next, info_sub = exhaust_enumerator(
            problem,
            partial_next,
            max_solutions=max_solutions - solutions_count,
            imcumbent=imcumbent,
            time_limit=end - time.time(),
            verbose=False
        )

        # Heap insertion for solutions based on cost
        for sol_next in sols_next:
            if sol_next.max_cost >= imcumbent:
                continue    # prune by incumbent

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

    # stats
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
            f"[Exhaustive] Completed. "
            f"Actions: {stats['num_actions']}, "
            f"Solutions: {stats['num_solutions']}, Time: {stats['time']:.2f}s, "
            f"Best Cost: {stats['best_cost']}, "
            f"Status: {stats['status']}"
        )

    # return results
    return solutions, stats




def exhaust_solver(
        problem: ShareARideProblem,
        partial: PartialSolution,
        time_limit: float = 30.0,
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Exhaustive search solver that returns only the best solution found.
    Uses exhaustive_enumerate internally.
    """
    solutions, info = exhaust_enumerator(    # pylint: disable=W0621
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
                f"[Exhaustive Solver] Best solution found. "
                f"Cost: {best_solution.max_cost}, "
                f"Time taken: {info['time']:.2f}s"
            )
    else:
        best_solution = None
        if verbose:
            print("[Exhaustive Solver] No feasible solution found.")

    return best_solution, info




# ================= Playground ==================
if __name__ == "__main__":
    from share_a_ride.solvers.algo.greedy import iterative_greedy_solver
    from share_a_ride.solvers.algo.utils import exhaustive_problem

    imcumbent_sol, _ = iterative_greedy_solver(
        exhaustive_problem,
        iterations=1000,
        time_limit=10.0,
        verbose=True
    )
    sols, info = exhaust_enumerator(
        exhaustive_problem,
        max_solutions=100000,
        imcumbent = imcumbent_sol.max_cost if imcumbent_sol else None,
        time_limit=30.0,
        verbose=True
    )
