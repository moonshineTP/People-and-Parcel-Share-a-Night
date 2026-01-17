"""
Branch-and-bound solver for Share-a-Ride problem.

This module implements a branch-and-bound algorithm that uses depth-first search (DFS)
to explore:
- The assignment of delivery pairs to taxis and
- The construction of routes for each taxi.
This approach incorporates lower-bound pruning to efficiently navigate the solution space,
while avoiding duplicate states and ensuring valid solutions.
"""
import time
import heapq
from typing import List, Optional, Tuple, Dict, Any, Union, Iterator, Callable, Set
from concurrent.futures import ThreadPoolExecutor

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, Solution




def mst_lowerbound(action_nodes: List[Tuple[int, int]], D: List[List[int]]) -> int:
    """
    Compute a Minimum Spanning Tree (MST) lower bound for the given action nodes.

    Params:
    - action_nodes: List of action nodes, represented by a tuple of (phys_in, phys_out)
    where phys_in and phys_out are physical node indices of the service.
    - D: Distance matrix.

    Returns:
    - The total weight of the MST as an integer.
    """
    if len(action_nodes) <= 1:
        return 0

    # Track internal sum of action nodes
    internal_cost = 0
    for node in action_nodes:
        if node[0] != node[1]:
            internal_cost += D[node[0]][node[1]]

    # Tracking structures
    remaining = set(action_nodes[1:])
    current = action_nodes[0]      # First node in the MST
    best = {            # Frontier best edge costs
        node: min(D[current[1]][node[0]], D[node[0]][current[1]])
        for node in remaining
    }
    total_cost = internal_cost

    # Loop for remaining nodes
    while remaining:
        # Pick next node with smallest edge cost
        next_node = min(remaining, key=lambda node: best[node])

        # Add its edge cost to total weight
        edge_cost = best[next_node]
        total_cost += edge_cost
        remaining.remove(next_node)

        # Update best costs for remaining nodes
        for node in remaining:
            cost = min(D[next_node[1]][node[0]], D[node[0]][next_node[1]])
            if cost < best[node]:
                best[node] = cost


    return total_cost




def _assign_pairs_canonical(
        partial: PartialSolution,
        incumbent: int,
        c: float = 0.75,
    ) -> Iterator[List[List[Union[int, Tuple[int, int]]]]]:
    """
    Assign pickup/delivery pairs into routes canonically with symmetry breaking.

    Uses the canonical assignment approach: first assign each item to a taxi ID,
    then convert to actual routes. Also, enforces symmetry breaking by only
    allowing new taxis to be opened in order (i.e., taxi 0 first, then taxi 1, etc.).

    Also, prunes assignments that cannot improve upon best_cost using a MST-based lower bound.
    This bound is derived fron the Greedy Incremental Tree estimate, which is quite close
    to the actual cost (though not guaranteed to be a lower bound).
    """
    problem = partial.problem
    N = problem.N       # pylint: disable=invalid-name
    M = problem.M       # pylint: disable=invalid-name
    K = problem.K       # pylint: disable=invalid-name
    num_reqs = N + M   # Total items to assign (N passengers + M parcels)


    # //// DFS to generate canonical taxi ID assignments
    # - idx: current item index to assign
    # - used: number of taxis used so far
    # - cur: current assignment being built, cur[req_id] = taxi_id.
    start_actnode: List[List[Tuple[int, int]]] = []     # Starting action nodes for each taxi
    for state in partial.states:
        start_actnode.append([(state["pos"], state["pos"])])

    assignment = []   # Current assignment being built
    actnode_sets = [    # Action nodes per taxi
        set(start) for start in start_actnode
    ]
    greedy_costs = [0 for _ in range(K)]  # Greedy costs per taxi




    # ================ Helper Functions ================
    def do_increment(
        taxi_id: int,
        new_node: Tuple[int, int]
    ):
        """
        Increase greedy cost for adding new_node to taxi_id's route.
        """
        current_nodes = actnode_sets[taxi_id]
        min_inc = min(
            min(problem.D[existing[1]][new_node[0]], problem.D[new_node[1]][existing[0]])
            for existing in current_nodes
        )
        greedy_costs[taxi_id] += min_inc
        actnode_sets[taxi_id].add(new_node)


    def greedy_increment(
        taxi_id: int,
        req_id: int,
    ):
        """
        Decrease greedy cost for removing req_id's nodes from taxi_id's route.
        """
        if req_id <= N:
            do_increment(taxi_id, problem.pserve(req_id))
        else:
            lid = req_id - N
            do_increment(taxi_id, (problem.lpick(lid), problem.lpick(lid)))
            do_increment(taxi_id, (problem.ldrop(lid), problem.ldrop(lid)))


    def do_decrement(
        taxi_id: int,
        new_node: Tuple[int, int]
    ):
        """
        Decrease greedy cost for removing new_node from taxi_id's route.
        """
        current_nodes = actnode_sets[taxi_id]
        current_nodes.remove(new_node)
        if not current_nodes:
            return
        min_inc = min(
            min(problem.D[existing[1]][new_node[0]], problem.D[new_node[1]][existing[0]])
            for existing in current_nodes
        )
        greedy_costs[taxi_id] -= min_inc


    def greedy_decrement(
        taxi_id: int,
        req_id: int,
    ):
        """
        Decrease greedy cost for removing req_id's nodes from taxi_id's route.
        """
        if req_id <= N:
            do_decrement(taxi_id, problem.pserve(req_id))
        else:
            lid = req_id - N
            do_decrement(taxi_id, (problem.lpick(lid), problem.lpick(lid)))
            do_decrement(taxi_id, (problem.ldrop(lid), problem.ldrop(lid)))




    # ================ DFS Assignment Function ================
    def do_assign(
            req_id: int,
            taxi_used: int,
        ) -> Iterator[List[int]]:
        """Assign items to taxi IDs with symmetry breaking."""
        if req_id == num_reqs:
            yield assignment[:]
            return

        # Assign to existing taxis
        for taxi_id in range(taxi_used):
            # // Update assignment and greedy cost
            assignment.append(taxi_id)
            greedy_increment(taxi_id, req_id)

            # // Prune using greedy cost
            if partial.route_costs[taxi_id] + greedy_costs[taxi_id] * c >= incumbent:
                assignment.pop()
                greedy_decrement(taxi_id, req_id)
                continue

            # // Recurse
            yield from do_assign(req_id + 1, taxi_used)

            # // Backtrack
            assignment.pop()
            greedy_decrement(taxi_id, req_id)

        # Open a new taxi if available (symmetry breaking)
        if taxi_used < K:
            # // Update assignment and greedy cost
            assignment.append(taxi_used)
            greedy_increment(taxi_used, req_id)

            # // Prune new-taxi branch as well (undo changes before continuing)
            if partial.route_costs[taxi_used] + greedy_costs[taxi_used] * c >= incumbent:
                assignment.pop()
                greedy_decrement(taxi_used, req_id)
            else:
                # // Recurse
                yield from do_assign(req_id + 1, taxi_used + 1)

                # // Backtrack
                assignment.pop()
                greedy_decrement(taxi_used, req_id)


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

        # Yield valid route assignment
        yield routes




def _assign_serve_regular(
        partial: PartialSolution,
        routeid: int,
        servelist: List[Union[int, Tuple[int, int]]],
        end_time: float,
        incumbent: int,
    ) -> Optional[List[int]]:
    """
    Given a servelist of remaining services has to be done (pserve/lpick/ldrop)
    and a set of parcels onboard, assign them to the respective route and return
    the best valid route found within the incumbent cost.
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
    parcels_onboard = set(partial.states[routeid]['parcels'])
    remaining_serves = set(servelist)
    actnodes_todo: Set[Tuple[int, int]] = set()
    for lid in parcels_onboard:
        dnode = problem.ldrop(lid)
        actnodes_todo.add((dnode, dnode))
    for item in servelist:
        if isinstance(item, tuple):
            actnodes_todo.add(item)
        else:
            actnodes_todo.add((item, item))

    best_route = None
    best_cost = incumbent

    def tle() -> bool:
        return time.time() >= end_time

    def do_serve(
            route: List[int],
            load: int,
            current_cost: int,
        ) -> None:
        nonlocal best_route, best_cost

        # Pruning with MST Lower Bound
        all_actnodes = [(route[-1], route[-1])] + list(actnodes_todo)
        lb = mst_lowerbound(all_actnodes, problem.D)
        if current_cost + lb >= best_cost:
            return

        # Terminal case: all services done and no parcels onboard
        if not remaining_serves and not parcels_onboard:
            final_cost = current_cost + problem.D[route[-1]][0]
            if final_cost < best_cost:
                best_cost = final_cost
                best_route = route + [0]  # return to depot
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

            # Apply move
            parcels_onboard.remove(lid)
            remaining_serves.remove(drop_node)
            actnodes_todo.remove((drop_node, drop_node))

            # Recurse
            do_serve(new_route, new_load, new_cost)

            # Backtrack
            parcels_onboard.add(lid)
            remaining_serves.add(drop_node)
            actnodes_todo.add((drop_node, drop_node))


        # //// Option 2: Try to serve next item from remaining
        # Iterate over a copy since we modify the set
        for item in list(remaining_serves):
            if tle():
                return

            # Extract move details
            if isinstance(item, tuple):
                actnode = item
            else:
                actnode = (item, item)

            # Apply move
            remaining_serves.remove(item)
            actnodes_todo.remove(actnode)


            # // Passenger serve: (pickup, delivery)
            if isinstance(item, tuple):
                pp, pd = item
                dist = problem.D[route[-1]][pp] + problem.D[pp][pd]
                new_cost = current_cost + dist

                # Recurse
                do_serve(route + [pp, pd], load, new_cost)

                # Backtrack
                remaining_serves.add(item)
                actnodes_todo.add(actnode)
                continue


            # // Parcel serve: node
            node = item
            dist = problem.D[route[-1]][node]
            new_cost = current_cost + dist

            if problem.is_lpick(node):      # Pickup node
                lid = problem.rev_lpick(node)
                qty = problem.q[lid - 1]
                new_load = load + qty

                if new_load <= capacity:
                    # Apply move: add parcel onboard and track its drop node
                    parcels_onboard.add(lid)

                    do_serve(route + [node], new_load, new_cost)

                    # Backtrack
                    parcels_onboard.remove(lid)

            elif problem.is_ldrop(node):    # Dropoff node
                lid = problem.rev_ldrop(node)
                qty = problem.q[lid - 1]
                new_load = load - qty

                if lid in parcels_onboard:
                    # Apply move: remove parcel from onboard
                    parcels_onboard.remove(lid)

                    do_serve(route + [node], new_load, new_cost)

                    # Backtrack
                    parcels_onboard.add(lid)

            # Backtrack
            remaining_serves.add(item)
            actnodes_todo.add(actnode)


    # //// Perform DFS and return best route found
    do_serve(initroute, initload, initcost)
    return best_route




def _process_single_assignment(
        assignment: List[List[Union[int, Tuple[int, int]]]],
        partial: PartialSolution,
        end_time: float,
        worst_cost: int,
    ) -> Optional[Solution]:
    """
    Process a single assignment. Returns solution or None.
    """
    if time.time() >= end_time:
        return None

    # Extract ordered routes for each taxi
    routes: List[List[int]] = []
    for routeid, servelist in enumerate(assignment):
        route = _assign_serve_regular(
            partial,
            routeid,
            servelist,
            end_time,
            worst_cost
        )
        if route is None:
            return None

        routes.append(route)

    return Solution(partial.problem, routes)




def bnb_enumerator(
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        n_return: int = 10,
        incumbent: Optional[int] = None,
        threads: int = 5,
        time_limit: float = 30.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[Solution], Dict[str, Any]]:
    """
    Branch-and-bound search solver for Share-a-Ride problem.

    The procedure is as follows:
    - First, use ordered DFS to assign pickup/delivery pairs to taxis. This order
    breaks symmetries and ensures canonical assignments only. This is aided by
    a greedy cost function to prune assignments that cannot improve upon the incumbent.
    - For each taxi, use a dfs to search for the best route.
    - Combine taxi routes into full solutions using DFS to simulate Descartes product.

    Params:
    - problem: The Share-a-Ride problem instance to solve.
    - partial: An optional initial partial solution to start from.
    - n_return: The maximum number of solutions to return.
    - incumbent: An optional incumbent solution cost to prune against.
    - time_limit: The maximum time (in seconds) to run the solver.
    - verbose: Whether to print progress information.

    Returns (solutions, info) tuple where
    - solutions: list of valid Solution objects found (up to max_solutions)
    - info: dictionary with statistics
    """

    if verbose:
        print("[BnB] Starting exhaustive enumeration...")

    start = time.time()
    end = start + time_limit
    if not partial:
        partial = PartialSolution(problem)
    if not incumbent:
        incumbent = 10**18  # effectively unlimited

    def tle() -> bool:
        return time.time() >= end


    # Extract canonical assignments (materialize generator to avoid double-consumption)
    if verbose:
        print("[BnB] Extracting canonical assignments...")
    assignments = list(_assign_pairs_canonical(partial, incumbent))
    num_assignments = len(assignments)
    if verbose:
        print(f"[BnB] Extracted {num_assignments} canonical assignments.")


    # //// Main enumeration loop ////
    solutions_count = 0
    assignments_count = 0
    solutions_keep = 0
    timeout = False
    worst_cost = incumbent
    top_heap: List[Tuple[int, int, Solution]] = []

    if verbose:
        print(f"[BnB] Initiating parallel assignment processing with {threads} threads...")

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(_process_single_assignment, assignment, partial, end, worst_cost)
            for assignment in assignments
        ]
        if verbose:
            print(f"[BnB] Submitted {len(futures)} assignment tasks to thread pool.")

        for future in futures:
            if tle():
                timeout = True
                break

            # Logging
            assignments_count += 1
            if verbose and assignments_count % 10 == 0:
                print(
                    f"[BnB] Processing {assignments_count}/{num_assignments} assignments. "
                    f"Time elapsed: {time.time() - start:.2f} seconds."
                )

            sol = future.result()
            if sol is None:
                continue

            solutions_count += 1
            cost = sol.max_cost

            # Logging
            if verbose:
                print(
                    f"[BnB] Assignment {assignments_count} yielded a valid solution "
                    f"with cost {cost}. "
                    f"Solutions found so far: {solutions_count}"
                )

            # Use heap for O(log n) top-k maintenance
            if solutions_keep < n_return:
                heapq.heappush(top_heap, (-cost, assignments_count, sol))
                solutions_keep += 1
                if solutions_keep == n_return:
                    worst_cost = -top_heap[0][0]  # max cost in heap
            elif cost < worst_cost:
                heapq.heapreplace(top_heap, (-cost, assignments_count, sol))
                worst_cost = -top_heap[0][0]


    # Extract solutions from heap, sorted by cost ascending
    solutions = [item[2] for item in sorted(top_heap, key=lambda x: -x[0])]

    if verbose:
        print(f"[BnB] Found {solutions_count} valid solutions.")


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
        print(f"[BnB] Enumeration finished in {elapsed:.2f} seconds.")
        print(f"[BnB] Total solutions found: {solutions_count}.")
        if solutions:
            print(f"[BnB] Best solution cost: {solutions[0].max_cost}.")
        else:
            print("[BnB] No feasible solutions found.")
        print(f"[BnB] Time elapsed: {elapsed:.2f} seconds.")
        print("------------------------------")
        print()


    return solutions, stats




def bnb_solver(
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        incumbent: Optional[int] = None,
        threads: int = 8,
        time_limit: float = 30.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Branch-and-bound solver wrapper for Share-a-Ride problem.
    This function calls the bnb_enumerator to find the best solution.

    Params:
    - problem: The Share-a-Ride problem instance to solve.
    - partial: An optional initial partial solution to start from.
    - incumbent: An optional incumbent solution cost to prune against.
    - time_limit: The maximum time (in seconds) to run the solver.
    - verbose: Whether to print progress information.
    """
    if not partial:
        partial = PartialSolution(problem)
    if not incumbent:
        incumbent = 10**18  # effectively unlimited

    solutions, stats = bnb_enumerator(
        problem,
        partial,
        1,
        incumbent,
        threads,
        time_limit,
        verbose=verbose
    )

    best_solution = solutions[0] if solutions else None

    return best_solution, stats




# ================= Playground ==================
if __name__ == "__main__":
    from share_a_ride.solvers.algo.greedy import iterative_greedy_solver
    from share_a_ride.solvers.algo.utils import exact_problems


    # //// Example usage and testing
    for probid, prob in enumerate(exact_problems, start=1):
        incumbent_sol, _ = iterative_greedy_solver(
            prob,
            iterations=2000,
            time_limit=20.0,
            seed=42,
            verbose=False
        )
        assert incumbent_sol
        print(f"Problem {probid}: Initial greedy solution cost = {incumbent_sol.max_cost}")
        solution, info = bnb_solver(
            problem=prob,
            incumbent = incumbent_sol.max_cost + 1 if incumbent_sol else None,
            time_limit=1200.0,
            verbose=True
        )
        print(f"Problem {probid}: Best solution cost = {solution.max_cost if solution else 'N/A'}")
        print("=========================================")
        print()