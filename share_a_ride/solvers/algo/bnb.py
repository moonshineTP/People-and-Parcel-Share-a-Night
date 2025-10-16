import time
from math import inf
from typing import List, Optional, Tuple, Dict, Any

from share_a_ride.problem import ShareARideProblem
from share_a_ride.solution import Solution
from share_a_ride.utils.helper import route_cost_from_sequence

# ---------------------- Branch-and-bound (DFS on pair->taxi with LB) -------------


def _deduplicate(nodes: List[int]) -> List[int]:
    """Return nodes with first-occurrence order preserved."""
    return list(dict.fromkeys(nodes))


def _mst_weight(nodes: List[int], D: List[List[int]]) -> int:
    """Return the MST weight over ``nodes`` using undirected edge costs."""
    if len(nodes) <= 1:
        return 0

    remaining = set(nodes[1:])
    current = nodes[0]
    best = {
        node: min(D[current][node], D[node][current])
        for node in remaining
    }
    weight = 0

    while remaining:
        next_node = min(remaining, key=lambda node: best.get(node, inf))
        edge_cost = best[next_node]
        weight += edge_cost
        remaining.remove(next_node)

        for node in remaining:
            cost = min(D[next_node][node], D[node][next_node])
            if cost < best[node]:
                best[node] = cost

    return weight


def _nn_lower_bound(nodes: List[int], D: List[List[int]]) -> int:
    """Compute a valid lower bound for covering ``nodes`` and returning to depot.

    The bound is built as:
      1. A minimum spanning tree (Prim) on the requested nodes.
      2. The cheapest edge leaving the depot to any node.
      3. The cheapest edge returning from any node to the depot.

    For any feasible tour that starts and ends at the depot and visits every
    node exactly once, the cost must be at least the sum of these three terms.
    """

    nodes = _deduplicate(nodes)
    if not nodes:
        return 0

    # Cheapest edges connecting depot to the set
    min_out = min(D[0][node] for node in nodes)
    min_in = min(D[node][0] for node in nodes)

    mst_weight = _mst_weight(nodes, D)

    return mst_weight + min_out + min_in


def _completion_lower_bound(current: int, nodes: List[int], D: List[List[int]]) -> int:
    """Lower bound on the extra cost to visit ``nodes`` from ``current`` and return."""

    nodes = _deduplicate(nodes)
    if not nodes:
        return D[current][0]

    mst_weight = _mst_weight(nodes, D)
    min_to_depot = min(D[node][0] for node in nodes)

    # Option 1: continue directly to the closest remaining node
    direct = min(D[current][node] for node in nodes) + mst_weight + min_to_depot

    # Option 2: return to depot first, then serve remaining nodes
    via_depot = D[current][0] + _nn_lower_bound(nodes, D)

    return min(direct, via_depot)



def branch_and_bound_solver(
        problem: ShareARideProblem,
        time_limit: float = 30.0,
        verbose: bool = False,
        target_cost: Optional[int] = None
    ) -> Tuple[Solution, Dict[str, Any]]:

    """
    Branch-and-bound:
        - DFS over pair assignments (symmetry-broken),
        - DFS over route construction per taxi,
        - Lower-bound pruning in both phases.
    Returns:
        (best_solution, info): tuple where
        - best_solution: best valid Solution object found (or None if none found)
        - info: dictionary with statistics:
            + assign_pairs: number of pair-to-taxi assignment nodes explored
            + build_nodes: number of route-construction nodes explored
            + pruned: number of nodes pruned by lower bounds
            + routes: number of taxi routes constructed
            + time: total time taken
            + status: "done" or "timeout"
    """

    start = time.time()
    N, M, K = problem.N, problem.M, problem.K
    pairs = [("P", i) for i in range(1, N + 1)] + [("L", j) for j in range(1, M + 1)]
    num_pairs = len(pairs)

    best_sol = None
    best_cost = float("inf") if target_cost is None else target_cost

    stats = {"assign_pairs": 0, "build_nodes": 0, "pruned": 0, "routes": 0}
    assignment = [-1] * num_pairs

    def tle() -> bool:
        return time_limit is not None and (time.time() - start) > time_limit


    # --------------- Helpers ---------------

    # Lower bound for pair->taxi assignment phase
    def lower_bound_assignment(pair_idx: int) -> float:
        """Lower bound on the max route cost after assigning pairs 0..pair_idx."""
        taxi_nodes = [[] for _ in range(K)]

        # Assign nodes to taxis based on current partial assignment
        for j in range(pair_idx + 1):
            kind, pid = pairs[j]
            if kind == "P":
                taxi_nodes[assignment[j]].extend(
                    [problem.ppick(pid), problem.pdrop(pid)]
                )
            else:
                taxi_nodes[assignment[j]].extend(
                    [problem.parc_pick(pid), problem.parc_drop(pid)]
                )

        # Lower bound contributed by each taxi's currently assigned requests
        return max(
            _nn_lower_bound(nodes, problem.D) if nodes else 0
            for nodes in taxi_nodes
        )


    # Lower bound for route construction per taxi
    def lower_bound_build(taxi_pairs: List[List[Tuple[str, int]]]) -> float:
        """Lower bound on the max cost required by the remaining taxis."""

        costs = []
        for t_pairs in taxi_pairs:
            if not t_pairs:
                costs.append(0)
                continue

            nodes: List[int] = []
            for kind, idx in t_pairs:
                if kind == "P":
                    nodes.extend([problem.ppick(idx), problem.pdrop(idx)])
                else:
                    nodes.extend([problem.parc_pick(idx), problem.parc_drop(idx)])

            est = _nn_lower_bound(nodes, problem.D)
            costs.append(est)

        return max(costs)


    # --------------- Main DFS functions ---------------

    # DFS phase 1: assign pairs to taxis
    def dfs_assign_delivery_pair(pair_idx: int) -> Optional[str]:
        """
            DFS generator to assign delivery pairs to taxis.
            Use symmetry-breaking and early branch-and-bound pruning.
        """
        nonlocal best_sol, best_cost

        # timeout
        if tle():
            return "timeout"

        # If assigning finished,
        # prepare taxi_pairs and enter route-construction DFS
        if pair_idx >= num_pairs:
            taxi_pairs = [[] for _ in range(K)]
            for i, tx in enumerate(assignment):
                taxi_pairs[tx].append(pairs[i])
            used_taxis = max(assignment) + 1 if num_pairs > 0 else 0
            return dfs_build_taxi_route(0, [], [], taxi_pairs, used_taxis)


        # Else: assign this pair and recurse
        stats["assign_pairs"] += 1

        # symmetry-breaking: taxis used so far plus one new if allowed
        used_taxis = (max(assignment[:pair_idx]) + 1) if pair_idx > 0 else 0
        taxi_considered = used_taxis + 1 if used_taxis < K else used_taxis

        for t in range(taxi_considered):
            if tle():
                return "timeout"
            # assign pair to taxi t
            assignment[pair_idx] = t

            # LB on partial assignment (only assigned pairs 0..pair_idx)
            lb1 = lower_bound_assignment(pair_idx)
            if lb1 >= best_cost:
                stats["pruned"] += 1
                continue

            # recurse to next pair
            res = dfs_assign_delivery_pair(pair_idx + 1)
            if res == "timeout":
                return "timeout"

        assignment[pair_idx] = -1  # backtrack
        return None


    # ---------- Phase 2: recursive route-building DFS ----------
    def dfs_build_taxi_route(t: int,
                current_routes: List[List[int]],
                current_costs: List[float],
                taxi_pairs: List[List[Tuple[str, int]]],
                used_taxis: int) -> Optional[str]:
        """
        Recursively build solution for taxi t, then recurse to t+1.
        taxi_pairs and used_taxis are prepared once when entering phase 2.
        """
        nonlocal best_sol, best_cost

        # timeout
        if tle():
            return "timeout"

        # ____ if all taxis assigned -> finalize solution ____
        if t == used_taxis:
            # pad unused taxis with empty routes
            full_routes = list(current_routes) + [[0, 0]] * (K - used_taxis)
            full_costs = list(current_costs) + [0] * (K - used_taxis)

            # submit solution
            sol = Solution(problem, full_routes, full_costs)
            assert sol.is_valid()

            # update best solution if improved
            if sol.max_cost < best_cost:
                best_cost = sol.max_cost
                best_sol = sol
                if verbose:
                    print(f"[update] best={best_cost:.2f}  time={time.time()-start:.1f}s")

            return None


        # ____ else -> build route for taxi t ____
        stats["build_nodes"] += 1

        # Before exploring taxi t, calculate an optimistic lower bound over remaining taxis.
        # current_cost is the largest cost so far.
        current_cost = max(current_costs) if current_costs else 0

        # Use lower_bound_build on the remaining taxi pairs (from taxi t to used_taxis-1).
        remaining_lb = lower_bound_build(taxi_pairs[t:used_taxis])
        # Join the two to get overall optimistic cost.
        optimistic_cost = max(current_cost, remaining_lb)
        if optimistic_cost >= best_cost:
            stats["pruned"] += 1
            return None

        # Items for this taxi.
        this_taxi_pairs = taxi_pairs[t]
        n = len(this_taxi_pairs)
        # Precompute types and IDs.
        types = [p[0] for p in this_taxi_pairs]
        ids = [p[1] for p in this_taxi_pairs]
        # Precompute relevant nodes for pickups and drops.
        pickup_nodes = [problem.ppick(i) if tp == "P" else problem.parc_pick(i)
                            for tp, i in this_taxi_pairs]
        drop_nodes = [problem.pdrop(i) if tp == "P" else problem.parc_drop(i)
                            for tp, i in this_taxi_pairs]

        # Nested DFS for route building on taxi t.
        def _dfs_route(
                seq: List[int],         # current sequence of nodes (including depots)
                cost: float,            # current cost of seq
                load: int,              # current parcel load
                passenger: int,         # current passenger onboard (0 if none)
                picked: List[bool],     # which pairs have been picked
                dropped: List[bool]     # which pairs have been dropped
            ) -> Optional[str]:

            # Check timeout.
            if tle():
                return "timeout"

            # If all pairs have been completed (dropped), we have a complete route.
            if all(dropped):
                # Increment routes stats.
                stats["routes"] += 1

                # submit this route
                cost += problem.D[seq[-1]][0]
                final_route = seq + [0]

                assert route_cost_from_sequence(final_route, problem.D) == cost

                current_routes.append(final_route)
                current_costs.append(cost)

                # recurse to next taxi
                res = dfs_build_taxi_route(t + 1, current_routes,
                        current_costs, taxi_pairs, used_taxis
                )
                if res == "timeout":
                    return "timeout"

                # backtrack
                current_routes.pop()
                current_costs.pop()

                # default return
                return None

            # Compute optimistic lower bound over remaining nodes.
            remaining_nodes = []
            for i in range(n):
                if tle():
                    return "timeout"
                if not picked[i]:
                    remaining_nodes.append(pickup_nodes[i])
                elif not dropped[i]:
                    remaining_nodes.append(drop_nodes[i])

            # Compute optimistic cost using completion lower bound
            optimistic = cost + _completion_lower_bound(seq[-1], remaining_nodes, problem.D)
            
            # Prune if optimistic cost is not better than best found.
            if optimistic >= best_cost:
                stats["pruned"] += 1
                return None  # prune

            # Try possible pickup moves.
            for i in range(n):
                if tle():
                    return "timeout"
                if not picked[i]:
                    # For passenger pickup: ensure taxi doesn't carry a passenger already.
                    if types[i] == "P" and passenger >= 1:
                        continue
                    # For parcel pickup: check capacity.
                    if types[i] == "L" and load + problem.q[ids[i] - 1] > problem.Q[t]:
                        continue

                    # apply pickup
                    picked[i] = True
                    next_node = pickup_nodes[i]
                    prev_node = seq[-1]
                    new_cost = cost + problem.D[prev_node][next_node]
                    new_load = load
                    new_passenger = passenger
                    if types[i] == "P":
                        new_passenger = 1
                    else:
                        new_load += problem.q[ids[i] - 1]
                    seq.append(next_node)

                    # continue DFS
                    ret = _dfs_route(seq, new_cost, new_load, new_passenger, picked, dropped)
                    if ret == "timeout":
                        return "timeout"

                    # backtrack
                    seq.pop()
                    picked[i] = False

            # Try possible drop moves.
            for i in range(n):
                if tle():
                    return "timeout"
                if picked[i] and not dropped[i]:
                    # apply drop
                    next_node = drop_nodes[i]
                    prev_node = seq[-1]
                    new_cost = cost + problem.D[prev_node][next_node]
                    new_load = load
                    new_passenger = passenger
                    if types[i] == "P":
                        new_passenger = 0
                    else:
                        new_load -= problem.q[ids[i] - 1]
                    dropped[i] = True
                    seq.append(next_node)

                    # continue DFS
                    ret = _dfs_route(seq, new_cost, new_load, new_passenger, picked, dropped)
                    if ret == "timeout":
                        return "timeout"

                    # backtrack
                    seq.pop()
                    dropped[i] = False

            return None

        # Initialize the DFS with empty route and initial state.
        ret = _dfs_route([0], 0, 0, 0, [False] * n, [False] * n)
        if ret == "timeout":
            return "timeout"
        return None


    # ---------- entry: start the assignment DFS ----------
    res = dfs_assign_delivery_pair(0)


    # ---------- finish and return stats ----------
    elapsed = time.time() - start
    info = {
        "assign_pairs": stats.get("assign_pairs", 0),
        "build_nodes": stats.get("build_nodes", 0),
        "pruned": stats.get("pruned", 0),
        "routes": stats.get("routes", 0),
        "time": elapsed,
        "status": "timeout" if res == "timeout" else "done"
    }

    if best_sol:
        if not best_sol.is_valid():
            best_sol = None

    assert best_sol.is_valid() if best_sol else True
    return best_sol, info
