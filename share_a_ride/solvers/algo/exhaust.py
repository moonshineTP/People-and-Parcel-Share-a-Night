"""
Exhaustive solver for the Share-a-Ride problem.

This module implements an exhaustive search algorithm by:
- Enumerating all canonical, symmetry-breaking assignments of pickup/delivery pairs to taxis.
- For each route assignment, generate all feasible serving sequences.
- Use Descartes product to combine taxi routes into full solutions. 
"""
import time
from typing import List, Tuple, Any, Dict, Iterator, Optional, Callable, Union

from share_a_ride.core import solution
from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, Solution
from share_a_ride.core.utils import route_cost_from_sequence




def _assign_pairs_canonical(
        partial: PartialSolution,
    ) -> Iterator[List[List[Union[int, Tuple[int, int]]]]]:
    """
    Assign pickup/delivery pairs into routes canonically with symmetry breaking.
    Uses the canonical assignment approach: first assign each item to a taxi ID,
    then convert to actual routes.
    """
    problem = partial.problem
    N = problem.N       # pylint: disable=invalid-name
    M = problem.M       # pylint: disable=invalid-name
    K = problem.K       # pylint: disable=invalid-name
    num_reqs = N + M   # Total items to assign (N passengers + M parcels)


    # //// Inner dfs generation
    # - idx: current item index to assign
    # - used: number of taxis used so far
    # - cur: current assignment being built, cur[req_id] = taxi_id.
    assignment: List[int] = []
    def do_assign(req_idx: int, taxi_used: int) -> Iterator[List[int]]:
        """Assign items to taxi IDs with symmetry breaking."""
        if req_idx == num_reqs:
            yield assignment[:]
            return

        # Assign to existing taxis
        for t in range(taxi_used):
            assignment.append(t)
            yield from do_assign(req_idx + 1, taxi_used)
            assignment.pop()

        # Open a new taxi if available (symmetry breaking)
        if taxi_used < K:
            assignment.append(taxi_used)
            yield from do_assign(req_idx + 1, taxi_used + 1)
            assignment.pop()

    do_assign(0, 0)

    # //// Convert taxi ID assignments to actual route assignments
    for taxi_assignment in do_assign(0, 0):
        # Initialize routes for all K taxis
        routes: List[List[Union[int, Tuple[int, int]]]] = [[] for _ in range(K)]

        # Assign passengers (first N items in the assignment)
        for pid in range(1, N + 1):
            taxi_id = taxi_assignment[pid - 1]
            routes[taxi_id].append(problem.pserve(pid))

        # Assign parcels (next M items in the assignment)
        for lid in range(1, M + 1):
            taxi_id = taxi_assignment[N + lid - 1]
            routes[taxi_id].append(problem.lpick(lid))
            routes[taxi_id].append(problem.ldrop(lid))

        yield routes




def _assign_serve_regular(
        partial: PartialSolution,
        routeid: int,
        servelist: List[Union[int, Tuple[int, int]]],
        tle: Callable[[], bool],
        incumbent: int,
    ) -> Iterator[List[int]]:
    """
    Given a servelist of remaining services has to be done (pserve/lpick/ldrop)
    and a set of parcels onboard, assign them to the respective route, satisfying
    precedence and capacity constraints.
    """
    problem = partial.problem
    capacity = problem.Q[routeid]

    # Initial route, load, cost
    initroute = partial.routes[routeid]
    initload = partial.states[routeid]['load']
    initcost = partial.route_costs[routeid]


    # //// Inner dfs procedure
    # - route: current route being built
    # - load: current load of the vehicle
    # - current_cost: current cost of the route

    # Global mutable sets for backtracking
    parcels_onboard = set(partial.states[routeid]['parcels'])
    remaining_serves = {*servelist}

    def do_serve(
            route: List[int],
            load: int,
            current_cost: int,
        ) -> Iterator[List[int]]:

        if current_cost > incumbent:
            return

        # Base case: all services done and no parcels onboard
        if not remaining_serves and not parcels_onboard:
            final_cost = current_cost + problem.D[route[-1]][0]
            if final_cost <= incumbent:
                yield route + [0]  # return to depot
            return


        # //// Option 1: Try to drop a parcel currently onboard
        # Iterate over a copy since we modify the set
        for lid in list(parcels_onboard):
            if tle():
                return

            # Extract move details
            drop_node = problem.ldrop(lid)
            dist = problem.D[route[-1]][drop_node]
            qty = problem.q[lid - 1]

            # Compute new state
            new_cost = current_cost + dist
            new_route = route + [drop_node]
            new_load = load - qty

            if new_cost > incumbent:
                continue

            # Apply move
            parcels_onboard.remove(lid)

            # Recurse
            yield from do_serve(new_route, new_load, new_cost)

            # Backtrack
            parcels_onboard.add(lid)


        # //// Option 2: Try to serve next item from remaining
        # Iterate over a copy since we modify the set
        for item in list(remaining_serves):
            # Apply move
            remaining_serves.remove(item)

            # Passenger serve: (pickup, delivery)
            if isinstance(item, tuple):
                pp, pd = item
                dist = problem.D[route[-1]][pp] + problem.D[pp][pd]
                new_cost = current_cost + dist

                if new_cost > incumbent:
                    remaining_serves.add(item) # Backtrack
                    continue

                yield from do_serve(route + [pp, pd], load, new_cost)

                # Backtrack
                remaining_serves.add(item)
                continue

            # Parcel serve: node
            node = item
            dist = problem.D[route[-1]][node]
            new_cost = current_cost + dist

            if new_cost > incumbent:
                remaining_serves.add(item) # Backtrack
                continue

            if problem.is_lpick(node):
                lid = problem.rev_lpick(node)
                qty = problem.q[lid - 1]
                new_load = load + qty

                if new_load <= capacity:
                    parcels_onboard.add(lid)
                    yield from do_serve(route + [node], new_load, new_cost)
                    parcels_onboard.remove(lid) # Backtrack

            elif problem.is_ldrop(node):
                lid = problem.rev_ldrop(node)
                qty = problem.q[lid - 1]
                new_load = load - qty

                if lid in parcels_onboard:
                    parcels_onboard.remove(lid)
                    yield from do_serve(route + [node], new_load, new_cost)
                    parcels_onboard.add(lid) # Backtrack

            # Backtrack
            remaining_serves.add(item)

    yield from do_serve(initroute, initload, initcost)




def _assign_route_sequential(
        partial: PartialSolution,
        all_candidates: List[List[List[int]]],
        tle: Callable[[], bool],
    ) -> Iterator[Solution]:
    """
    Given all taxi route assignments, combine them into full solutions.
    """
    problem = partial.problem


    # //// Inner dfs procedure
    # - t_idx: current taxi index
    current_routes: List[List[int]] = []
    def do_product(t_idx: int):
        if t_idx == problem.K:
            sol = Solution(problem, list(current_routes))
            if sol.is_valid():
                yield sol
            return

        # Try all feasible routes for current taxi
        for route in all_candidates[t_idx]:
            if tle():
                return

            current_routes.append(route)
            yield from do_product(t_idx + 1)
            current_routes.pop()


    yield from do_product(0)




# ================ Main APIs ================
def exhaust_enumerator(
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        n_return: int = 1000,
        incumbent: Optional[int] = None,
        time_limit: float = 30.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[Solution], Dict[str, Any]]:
    """
    Exhaustive enumeration of all feasible solutions.
    The procedure is as follows:
    - First, use ordered DFS to assign pickup/delivery pairs to taxis. This order
    breaks symmetries and ensures canonical assignments only.
    - For each taxi, generate all feasible serving sequences using another DFS
      that respects precedence and capacity constraints.
    - Combine taxi routes into full solutions using DFS to simulate Descartes product.

    Params:
    - problem: ShareARideProblem instance
    - partial: PartialSolution instance to start from
    - n_return: number of best solutions to return (must be non-negative)
    - incumbent: cost of the best known solution (to prune search)
    - time_limit: time limit for the enumeration (in seconds)
    - verbose: whether to print information during execution

    Returns (solutions, info) tuple where
    - solutions: list of valid Solution objects found (up to max_solutions)
    - info: dictionary with statistics
    """
    if verbose:
        print("[Exhaust] Starting exhaustive enumeration...")

    start = time.time()
    end = start + time_limit
    K = problem.K   # pylint: disable=invalid-name
    if not partial:
        partial = PartialSolution(problem)
    if not incumbent:
        incumbent = 10**18  # effectively unlimited

    def tle() -> bool:
        return time.time() >= end


    # //// Main enumeration loop ////
    solutions: List[Solution] = []
    solutions_count = 0
    solutions_keep = 0

    assignments_count = 0
    timeout = False
    worst_cost = incumbent

    # Extract canonical assignments (materialize generator to avoid double-consumption)
    assignments = list(_assign_pairs_canonical(partial))
    if verbose:
        print(f"[Exhaust] Extracted {len(assignments)} canonical assignments.")
        print("[Exhaust] Starting enumeration of assignments...")

    # Build solutions for each assignment
    for assignment in assignments:
        if tle():
            timeout = True
            break

        # For each taxi, generate all feasible serving sequences
        candidates: List[List[List[int]]] = [
            list(_assign_serve_regular(
                partial,
                t_idx,
                assignment[t_idx],
                tle,
                incumbent
            )) for t_idx in range(K)
        ]

        # Combine taxi routes into full solutions
        solutions_assigned = list(_assign_route_sequential(
            partial,
            candidates,
            tle
        ))
        solutions_count += len(solutions_assigned)
        solutions_assigned.sort(key=lambda s: s.max_cost)
        assignments_count += 1

        # //// Insertion sort into solutions list
        for sol in solutions_assigned:
            if solutions_keep < n_return:
                # List not full yet, insert in sorted position
                insert_pos = 0
                while insert_pos < solutions_keep and solutions[insert_pos].max_cost < sol.max_cost:
                    insert_pos += 1
                solutions.insert(insert_pos, sol)
                solutions_keep += 1
                if solutions_keep == n_return:
                    worst_cost = solutions[-1].max_cost

            elif sol.max_cost < worst_cost:
                # Better than worst, insert in sorted position and remove worst
                insert_pos = 0
                while insert_pos < solutions_keep and solutions[insert_pos].max_cost < sol.max_cost:
                    insert_pos += 1
                solutions.insert(insert_pos, sol)
                solutions.pop()  # Remove worst (last element)
                worst_cost = solutions[-1].max_cost

        # Logging
        if verbose:
            print(
                f"[Exhaust] Assignments processed: {assignments_count}, "
                f"Solutions found so far: {solutions_count}"
            )


    # Final stats
    elapsed = time.time() - start
    stats = {
        "solutions_found": solutions_count,
        "best_cost": solutions[0].max_cost if solutions else 'N/A',
        "time": elapsed,
        "status": "timeout" if timeout else "done"
    }

    # Logging
    if verbose:
        print()
        print(f"[Exhaust] Enumeration finished in {elapsed:.2f} seconds.")
        print(f"[Exhaust] Total solutions found: {solutions_count}.")
        if solutions:
            print(f"[Exhaust] Best solution cost: {solutions[0].max_cost}.")
        else:
            print("[Exhaust] No feasible solutions found.")
        print(f"[Exhaust] Time elapsed: {elapsed:.2f} seconds.")
        print("------------------------------")
        print()


    return solutions, stats




def exhaust_solver(
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        incumbent: Optional[int] = None,
        time_limit: float = 30.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Exhaustive search solver that returns only the best solution found.
    Uses exhaustive_enumerate internally.
    """
    if not partial:
        partial = PartialSolution(problem)
    if not incumbent:
        incumbent = 10**18  # effectively unlimited

    solutions, stats = exhaust_enumerator(
        problem,
        partial,
        1,
        incumbent,
        time_limit,
        verbose=verbose
    )

    best_solution = solutions[0] if solutions else None

    return best_solution, stats




# ================= Playground ==================
if __name__ == "__main__":
    from share_a_ride.solvers.algo.greedy import iterative_greedy_solver
    from share_a_ride.solvers.algo.utils import exact_problems

    for probid, prob in enumerate(exact_problems):
        if probid != 1:
            continue
        incumbent_sol, _ = iterative_greedy_solver(
            prob,
            iterations=2000,
            time_limit=20.0,
            seed=42,
            verbose=False
        )
        assert incumbent_sol
        print(f"Problem {probid}: Initial greedy solution cost = {incumbent_sol.max_cost}")
        solution, info = exhaust_solver(
            prob,
            incumbent = incumbent_sol.max_cost if incumbent_sol else None,
            time_limit=240.0,
            verbose=True
        )
        print(f"Problem {probid}: Best solution cost = {solution.max_cost if solution else 'N/A'}")
