"""
HGS Solver for Share-a-Ride Problem using PyVRP.
"""
from random import Random
import time
from collections import Counter
from typing import Tuple, Optional, Dict, Any, List

from pyvrp import Solution as PyvrpSolution
from pyvrp import Model, Route, Result
from pyvrp import RandomNumberGenerator, GeneticAlgorithm, Population, PenaltyManager
from pyvrp.stop import MaxRuntime
from pyvrp.search import LocalSearch, compute_neighbours
from pyvrp.diversity import broken_pairs_distance
from pyvrp.crossover import selective_route_exchange

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import Solution, PartialSolutionSwarm
from share_a_ride.solvers.algo.beam import beam_enumerator




def hgs_solver(
        problem: ShareARideProblem,
        swarm: Optional[PartialSolutionSwarm] = None,
        iterations: int = 8,
        n_partials: int = 40,
        time_limit: float = 30.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Solve the Share-a-Ride problem using PyVRP (HGS).
    
    Implements a Binary Search over the maximum route duration/distance to minimize
    the maximum cost (Min-Max objective), as PyVRP minimizes total cost by default.
    
    Constraints handled:
    - Capacity (Parcel volume and Passenger count)
    - Precedence (Pickup before Delivery)
    - Same Vehicle (Pickup and Delivery on same route)
    - Cost limits (via Binary Search)
    """

    start = time.time()
    N = problem.N  # Number of passengers   # pylint: disable=invalid-name
    M = problem.M  # Number of parcels      # pylint: disable=invalid-name


    # //// Mapping capacity <-> vehicle type index
    node_to_pyvrp = {}
    for pid in range(1, N + 1):
        p, d = problem.pserve(pid)
        node_to_pyvrp[p] = pid
        node_to_pyvrp[d] = -1  # Drop node not directly represented
    for lid in range(1, M + 1):
        p = problem.lpick(lid)
        node_to_pyvrp[p] = N + lid
        d = problem.ldrop(lid)
        node_to_pyvrp[d] = N + M + lid

    cap_to_pyvrp_type = {}
    pyvrp_type_to_cap = {}
    current_type_idx = 0
    for cap in problem.Q:
        if cap not in cap_to_pyvrp_type:
            cap_to_pyvrp_type[cap] = current_type_idx
            pyvrp_type_to_cap[current_type_idx] = cap
            current_type_idx += 1



    # /// Mapping Pyvrp locations to physical locations
    # Note: location 0 is depot and a passenger node is a collapsed node (pickup+drop)
    num_points = 1 + N + 2 * M
    map_departure = [0] * num_points
    map_arrival = [0] * num_points
    curr = 1

    # Passengers (collapsed clients): arrive at pickup, leave from drop
    for pid in range(1, N + 1):
        p, d = problem.pserve(pid)
        map_arrival[curr] = p
        map_departure[curr] = d
        curr += 1

    # Parcel pickups (arrive/leave same location)
    for lid in range(1, M + 1):
        p = problem.lpick(lid)
        map_arrival[curr] = p
        map_departure[curr] = p
        curr += 1

    # Parcel drops (arrive/leave same location)
    for lid in range(1, M + 1):
        d = problem.ldrop(lid)
        map_arrival[curr] = d
        map_departure[curr] = d
        curr += 1


    # //// Solver for a fixed max distance limit ////
    def solve_instance(
            max_dist_limit: int,
            run_time: float,
            initial_data: Optional[List[List[Tuple[int, List[int]]]]] = None,
            seed_inner: Optional[int] = None
        ) -> Result:

        # Initialize PyVRP Model and Data
        model = Model()
        clients = []


        # //// Add depot
        depot_x, depot_y = (problem.coords[0] if problem.coords else (0, 0))
        depot = model.add_depot(x=depot_x, y=depot_y)


        # //// Add clients
        # A. Passengers
        for pid in range(1, N + 1):
            pick_node, drop_node = problem.pserve(pid)
            x, y = (problem.coords[pick_node] if problem.coords else (0, 0))
            inner_cost = problem.D[pick_node][drop_node]

            client = model.add_client(
                x=x, y=y,
                pickup=[0],
                delivery=[0],
                service_duration=inner_cost
            )
            clients.append(client)

        # B. Parcel Pickups
        for lid in range(1, M + 1):
            node_idx = problem.lpick(lid)
            x, y = (problem.coords[node_idx] if problem.coords else (0, 0))
            qty = problem.q[lid - 1]

            client = model.add_client(
                x=x, y=y,
                pickup=[qty],
                delivery=[0],
                service_duration=0
            )
            clients.append(client)

        # C. Parcel Dropoffs
        for lid in range(1, M + 1):
            node_idx = problem.ldrop(lid)
            x, y = (problem.coords[node_idx] if problem.coords else (0, 0))
            qty = problem.q[lid - 1]

            client = model.add_client(
                x=x, y=y,
                pickup=[0],
                delivery=[qty],
                service_duration=0
            )
            clients.append(client)


        # //// Add vehicle
        cap_counts = Counter(problem.Q)
        current_type_idx = 0
        for cap_val, count in cap_counts.items():
            # Simple 1D Capacity: [ParcelWeightLimit]
            model.add_vehicle_type(
                num_available=count,
                capacity=[cap_val],
                max_distance=max_dist_limit
            )
            current_type_idx += 1


        # //// Add edges
        vrp_locs = [depot] + clients
        for fid in range(num_points):
            for tid in range(num_points):
                if fid == tid:
                    continue

                u_phys = map_departure[fid]
                v_phys = map_arrival[tid]
                dist = problem.D[u_phys][v_phys]
                model.add_edge(vrp_locs[fid], vrp_locs[tid], distance=dist)


        # //// Integrate Initial Solutions
        # Defensive: If no initial solutions provided, solve as is.
        if not initial_data:
            res = model.solve(stop=MaxRuntime(run_time))
            return res

        # Prepare GA components
        data = model.data()
        rng = RandomNumberGenerator(seed=seed_inner if seed_inner else 42)
        pm = PenaltyManager(([100], 100, 100))
        pop = Population(broken_pairs_distance)
        neighbours = compute_neighbours(data)
        ls = LocalSearch(data, rng, neighbours)


        # Prepare initial PyvrpSolutions
        init_pyvrpsols = []
        for sol_routes in initial_data:
            # Reset vehicle availability for each solution
            pyvrp_routes = []

            for k, route in sol_routes:
                # Convert physical route to PyVRP client indices
                converted_route = []
                for node in route:
                    if node == 0:
                        continue
                    if node in node_to_pyvrp:
                        pyvrp_idx = node_to_pyvrp[node]
                        if pyvrp_idx != -1:
                            converted_route.append(pyvrp_idx)

                # Get pyvrp_vtype
                cap = problem.Q[k]
                pyvrp_vtype = cap_to_pyvrp_type[cap]

                # Create Route object
                route_obj = Route(data, converted_route, pyvrp_vtype)
                pyvrp_routes.append(route_obj)

            init_sol = PyvrpSolution(data, pyvrp_routes)
            init_pyvrpsols.append(init_sol)


        # //// Solve with GA
        ga = GeneticAlgorithm(
            data=data,
            penalty_manager=pm,
            rng=rng,
            population=pop,
            search_method=ls,
            crossover_op=selective_route_exchange,
            initial_solutions=init_pyvrpsols,
        )
        res = ga.run(stop=MaxRuntime(run_time))

        return res




    # ================= Binary Search =================
    # Because PyVRP minimizes total cost and do not support Min-Max objective directly,
    # we perform a binary search over the maximum allowed route distance to minimize
    # the maximum route cost.


    # //// Initial Solution Sampling
    if verbose:
        print("[HGS] Generating initial solution population")

    # Generate initial swarm if not provided
    if swarm is not None:
        init_swarm = swarm
    else:
        init_swarm, _ = beam_enumerator(
            problem,
            n_partials=n_partials,
            n_return=20,
            r_intra=1.2,
            r_inter=1.2,
            time_limit=10**9,   # Effectively no limit
            seed=11*seed if seed is not None else None,
            verbose=verbose,
        )

    init_best_sol = init_swarm.opt()
    init_best_cost = init_best_sol.max_cost if init_best_sol else 10**18
    # [End of initial solution generation]

    if verbose:
        print(f"[HGS] Best Initial solution found. Max Cost: {init_best_cost}")
    else:
        print("[HGS] No Initial solution found.")


    # //// Collect and format initial solutions
    initial_sol_data: List[List[Tuple[int, List[int]]]] = []
    for partial in init_swarm.partial_lists:
        if not partial.is_completed():
            continue

        sol_routes = []
        for k, route in enumerate(partial.routes):
            # r is [0, node1, node2, ..., 0]
            r_indices = [node for node in route if node != 0]
            if r_indices:
                sol_routes.append((k, r_indices))
        if sol_routes:
            initial_sol_data.append(sol_routes)

    if verbose:
        print(f"[HGS] Extracted {len(initial_sol_data)} solutions for seeding.")


    # //// Phase 1: Initial Feasible Solution
    if verbose:
        print("[HGS] Trying to run a first solve...")

    # We use a large limit to ensure we find *some* feasible solution first.
    t1 = max(1.0, time_limit * 0.3)
    res = solve_instance(
        10**18, t1, initial_data=initial_sol_data
    )
    if not res.is_feasible():
        if verbose:
            print("[HGS] Could not find feasible solution in initial phase.")

        if init_best_cost < 10**18:
            if verbose:
                print("[HGS] Falling back to initial solution.")
            return init_best_sol, {"method": "initial", "cost": init_best_cost, "status": "done"}

        return None, {"error": "Infeasible", "status": "error"}

    # Calculate current max cost (distance + service_duration for passenger inner trips)
    def get_route_cost(route) -> int:
        """Calculate full route cost including passenger inner trips."""
        return route.distance() + route.service_duration()

    current_max = max((get_route_cost(r) for r in res.best.routes()), default=0)
    if verbose:
        print(f"[HGS] Initial Feasible Max Cost: {current_max}")

    # Track the best HGS solution from Phase 1
    best_hgs_res = res
    best_hgs_max_cost = current_max


    # //// Phase 2: Binary Search
    best_max_cost = current_max
    low: int = int(current_max / 1.2)       # Christian's suggestion
    high: int = current_max                 # We want to improve, not loosen
    # Note: best_hgs_res/best_hgs_max_cost track the overall best HGS solution (Phase 1 + 2)
    remaining_time = time_limit - (time.time() - start)
    time_per_iter = max(0, remaining_time - 2.0) / iterations \
        if iterations > 0 else 0

    #  Logging
    if verbose and iterations > 0:
        print(
            f"[HGS] Starting Binary Search with {iterations} iterations, "
            f"{time_per_iter:.2f}s per iter."
        )

    # Main Binary Search Loop
    for i in range(iterations):
        # Breaking criterion
        if high - low < 1.1:
            break

        # Define mid point
        mid = (low + high) // 2
        if verbose:
            print(f"[HGS] [Iter {i+1}] Testing limit {mid}...")

        # Solve with current mid as max distance limit
        res_iter = solve_instance(
            mid, time_per_iter, initial_sol_data,
            seed_inner=seed + 101*i + i * i if seed else None
        )

        # Analyze result
        if res_iter.is_feasible():      # Mid is feasible, new best
            # Extract actual max cost (distance + service_duration)
            actual_max = max((get_route_cost(r) for r in res_iter.best.routes()), default=0)

            # Update binary search bounds
            if actual_max < best_max_cost:
                best_max_cost = actual_max
            high = min(mid, actual_max)

            # Update overall best HGS solution
            if actual_max < best_hgs_max_cost:
                best_hgs_res = res_iter
                best_hgs_max_cost = actual_max

            # Logging
            if verbose:
                print(f"  -> Feasible. Max cost found: {actual_max}")

        else:                           # Mid is infeasible, loosen limit
            low = int(1.01 * mid) + 1

            # Logging
            if verbose:
                print("  -> Infeasible.")


    # //// Reconstruction
    # Use the best HGS solution found (from Phase 1 or Phase 2)
    res = best_hgs_res
    if not res.is_feasible():
        return None, {"error": "Infeasible", "status": "error"}

    if verbose:
        print(f"[HGS] Best HGS Max Cost: {best_hgs_max_cost}")
    used_routes: List[Route] = res.best.routes()
    routes_by_cap: Dict[int, List[List[int]]] = {}

    # Extract routes by capacity
    for route in used_routes:
        vidx = route.vehicle_type()    # It's a method returning index
        cap = pyvrp_type_to_cap[vidx]

        # Add new entry if needed
        if cap not in routes_by_cap:
            routes_by_cap[cap] = []

        # Extract sequence in #Solution format
        seq = [0]
        for cid in route:       # cid is in PyVRP indexing
            node_arr = map_arrival[cid]         # re-map to physical location
            node_dep = map_departure[cid]

            if node_arr != node_dep:    # Passenger node
                seq.append(node_arr)
                seq.append(node_dep)
            else:                       # Parcel pickup/drop node
                seq.append(node_arr)
        seq.append(0)                   # Return to depot following #Solution format

        # Append the sequence to the list for this capacity
        routes_by_cap[cap].append(seq)

    # Map routes back to vehicles
    final_routes: List[List[int]] = [[0,0]] * problem.K
    for k in range(problem.K):
        cap = problem.Q[k]
        if routes_by_cap[cap]:
            final_routes[k] = routes_by_cap[cap].pop(0)

    # Create Solution Object
    sol = Solution(problem, final_routes)


    # //// Finalization
    # Compare HGS solution with Initial beam search solution
    # Select the best among: beam search (init_best_sol), and HGS result (sol)
    hgs_max_cost = sol.max_cost if sol.is_valid() else 10**18

    if verbose:
        print(f"[HGS] Beam search best: {init_best_cost}, HGS best: {hgs_max_cost}")

    # Choose the better solution
    if sol.max_cost == 0 or not sol.is_valid() or init_best_cost < hgs_max_cost:
        if verbose:
            print(
                f"[HGS] Beam search {init_best_cost} better than HGS {hgs_max_cost}. "
                "Returning Beam search solution."
            )
        sol = init_best_sol

    assert sol is not None and sol.is_valid()

    # Summary
    stats = {
        "cost": res.cost(),
        "runtime": res.runtime,
        "iterations": iterations,
        "best_max_cost": sol.max_cost,
        "status": "done" if time.time() - start < time_limit else "overtime"
    }

    # Logging
    if verbose:
        print()
        print(
            "[HGS] Finished. "
            f"Final Max Cost: {sol.max_cost}."
        )
        print("------------------------------")
        print()

    # Return
    return sol, stats




# ========== Playground ==========
if __name__ == "__main__":
    from share_a_ride.solvers.algo.utils import test_problem

    # Solve with HGS
    solution, info = hgs_solver(test_problem, time_limit=300.0, verbose=True)
    assert solution is not None and solution.is_valid()

    solution.stdin_print()
