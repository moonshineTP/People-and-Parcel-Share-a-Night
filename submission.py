# ----- file oj_final.py begin -----
"""
    Dedicated for online judge systems submission.
    Use inliner.py for inlining modules of this file into one file.
    The final output is submission.py
"""
import sys


# ----- file share_a_ride/solvers/bnb.py begin -----
import time
from typing import List, Optional, Tuple, Dict, Any


# ----- file share_a_ride/problem.py begin -----
from typing import List


class ShareARideProblem:
    def __init__(self, N: int, M: int, K: int,
                parcel_qty: List[int], vehicle_caps: List[int],
                dist: List[List[int]]):

        # Basic parameters
        self.N = N
        self.M = M
        self.K = K
        self.q = list(parcel_qty)
        self.Q = list(vehicle_caps)
        self.D = [row[:] for row in dist]
        self.num_nodes = 2*N + 2*M + 1

        # index helpers
        self.ppick = lambda i: i
        self.pdrop = lambda i: N + M + i
        self.parc_pick = lambda j: N + j
        self.parc_drop = lambda j: 2*N + M + j

        self.rev_ppick = lambda i: i
        self.rev_pdrop = lambda n: n - (N + M)
        self.rev_parc_pick = lambda n: n - N
        self.rev_parc_drop = lambda n: n - (2 * N + M)

        # Node check helpers
        self.is_ppick = lambda x: 1 <= x <= N
        self.is_pdrop = lambda x: N + M + 1 <= x <= 2 * N + M
        self.is_parc_pick = lambda x: N + 1 <= x <= N + M
        self.is_parc_drop = lambda x: 2 * N + M + 1 <= x <= 2 * (N + M)

    def copy(self):
        return ShareARideProblem(self.N, self.M, self.K,
            list(self.q), list(self.Q), [row[:] for row in self.D]
        )

    def stdin_print(self):
        print(self.N, self.M, self.K)
        print(*self.q)
        print(*self.Q)
        for row in self.D:
            print(*row)

    def pretty_print(self, verbose: int = 0):
        print(f"Share-a-Ride: N={self.N} passengers, M={self.M} parcels, "
            f"K={self.K}, num_nodes={self.num_nodes}")

        if verbose >= 1:
            print("Parcel quantities (q):", self.q)
            print("Vehicle capacities (Q):", self.Q)
            print("Distance matrix D:")
            for row in self.D:
                print(" ", row)
# ----- file share_a_ride/problem.py end -----

# ----- file share_a_ride/solution.py begin -----
from typing import List

# ----- file utils/helper.py begin -----
from typing import List

# ---------------------- Helper functions ----------------------------------------
def route_cost_from_sequence(
        seq: List[int], 
        D: List[List[int]], 
        verbose: bool = False
    ) -> int:

    prev, total_cost = 0, 0
    for node in seq[1:]:
        if verbose: 
            print(D[prev][node], end=" ")

        total_cost += D[prev][node]
        prev = node
        
    if verbose:
        print()

    return total_cost

# ----- file utils/helper.py end -----

class Solution:
    """
    Solution object representing K routes.
    - routes: list of lists of node indices (each route includes depot 0 at start and end)
    - route_lengths: list of ints
    - max_length: int (objective to minimize)
    """
    def __init__(self, problem: "ShareARideProblem",
                routes: List[List[int]], route_lengths: List[int]):

        if problem is None:
            raise ValueError("Problem instance cannot be None.")

        assert len(routes) == len(route_lengths)
        self.problem = problem
        self.routes = routes
        self.route_costs = route_lengths
        self.max_cost = max(route_lengths) if route_lengths else 0

    def is_valid(self) -> bool:
        """
            Check depot, precedence, and capacity constraints.
        """

        prob = self.problem
        N, M, K = prob.N, prob.M, prob.K
        if len(self.routes) != K:
            return False

        for route_idx, route in enumerate(self.routes):
            # must start and end at depot
            if not (route[0] == 0 and route[-1] == 0):
                return False

            # track passengers and parcels
            passenger_onboard = set()
            parcel_onboard = set()
            parcel_load = 0

            visited_pass_pick = set()
            visited_parc_pick = set()

            for node in route[1:-1]:

                # passenger pickup
                if prob.is_ppick(node):
                    id = prob.rev_ppick(node)

                    # check
                    if id in visited_pass_pick:
                        return False
                    if len(passenger_onboard) >= 1:
                        return False

                    # add
                    passenger_onboard.add(id)
                    visited_pass_pick.add(id)

                # passenger drop
                elif prob.is_pdrop(node):
                    id = prob.rev_pdrop(node)

                    # check
                    if id not in passenger_onboard:
                        return False

                    # remove
                    passenger_onboard.remove(id)

                # parcel pickup
                elif prob.is_parc_pick(node):
                    jd = prob.rev_parc_pick(node)

                    # check
                    if jd in visited_parc_pick or jd in parcel_onboard:
                        return False
                    parcel_load += prob.q[jd - 1]
                    if parcel_load > prob.Q[route_idx]:
                        return False

                    # add
                    visited_parc_pick.add(jd)
                    parcel_onboard.add(jd)

                # parcel drop
                elif prob.is_parc_drop(node):
                    jd = prob.rev_parc_drop(node)

                    # check
                    if jd not in parcel_onboard:
                        return False

                    # remove
                    assert parcel_load - prob.q[jd - 1] >= 0
                    parcel_load -= prob.q[jd - 1]
                    parcel_onboard.remove(jd)

            # at end of route
            if passenger_onboard:
                return False
            if parcel_load != 0:
                return False

        return True

    def pretty_print(self, verbose: int = 0):
        """
            Print the solution in the specified format.
            Verbose option for more details.
            - verbose=0: for online judge
            - verbose=1: detailed route info
        """
        if verbose:
            print(f"*** Max route cost: {self.max_cost} ***")

        print(self.problem.K)
        
        for route, cost in zip(self.routes, self.route_costs):
            if verbose:
                print(f"- Route cost: {cost}")
            print(len(route))
            print(" ".join(map(str, route)))

        
# ----- file share_a_ride/solution.py end -----
# from utils.helper import route_cost_from_sequence

# ---------------------- Branch-and-bound (DFS on pair->taxi with LB) -------------

# Nearest-neighbor heuristic for lower bound
def _nn_lower_bound(nodes: List[int], D: List[List[int]]) -> int:
    """
        Nearest-neighbor heuristic: short route covering nodes + depot ư/ no constraints.
        Used as a fast optimistic lower bound in BnB.
    """

    if not nodes:
        return 0

    unvisited = set(nodes)

    pos, total = 0, 0
    while unvisited:
        nxt = min(unvisited, key=lambda x: D[pos][x])
        total += D[pos][nxt]
        pos = nxt
        unvisited.remove(nxt)
    total += D[pos][0]

    return total


def branch_and_bound(
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


    # --------------- Helpers ---------------

    # Lower bound for pair->taxi assignment phase
    def lower_bound_assignment(pair_idx: int) -> float:
        """
            Compute optimistic lower bound for partial assignment.
            Use NN bound on each taxi's assigned nodes so far.
        """
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

        # Use NN bound for each taxi
        return max(_nn_lower_bound(nodes, problem.D) if nodes else 0 for nodes in taxi_nodes)


    # Lower bound for route construction per taxi
    def lower_bound_build(taxi_pairs: List[List[Tuple[str, int]]]) -> float:
        """LB based on current best partial routes per taxi."""

        costs = []
        for t_pairs in taxi_pairs:
            if not t_pairs:
                costs.append(0)
                continue

            est = _nn_lower_bound(
                [problem.ppick(i) if type == "P" else problem.parc_pick(i)
                    for type, i in t_pairs], problem.D
            )
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
        if time.time() - start > time_limit:
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
        if time.time() - start > time_limit:
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
        def dfs_route(
                seq: List[int],         # current sequence of nodes (including depots)
                cost: float,            # current cost of seq
                load: int,              # current parcel load
                passenger: int,         # current passenger onboard (0 if none)
                picked: List[bool],     # which pairs have been picked
                dropped: List[bool]     # which pairs have been dropped
            ) -> Optional[str]:

            # Check timeout.
            if time.time() - start > time_limit:
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
                if not picked[i]:
                    remaining_nodes.append(pickup_nodes[i])
                elif not dropped[i]:
                    remaining_nodes.append(drop_nodes[i])
            
            # Compute optimistic cost
            optimistic = cost + _nn_lower_bound(remaining_nodes, problem.D)
            
            # Prune if optimistic cost is not better than best found.
            if optimistic >= best_cost:
                stats["pruned"] += 1
                return None  # prune

            # Try possible pickup moves.
            for i in range(n):
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
                    ret = dfs_route(seq, new_cost, new_load, new_passenger, picked, dropped)
                    if ret == "timeout":
                        return "timeout"
                    
                    # backtrack
                    seq.pop()
                    picked[i] = False

            # Try possible drop moves.
            for i in range(n):
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
                    ret = dfs_route(seq, new_cost, new_load, new_passenger, picked, dropped)
                    if ret == "timeout":
                        return "timeout"

                    # backtrack
                    seq.pop()
                    dropped[i] = False

            return None

        # Initialize the DFS with empty route and initial state.
        ret = dfs_route([0], 0, 0, 0, [False] * n, [False] * n)
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
# ----- file share_a_ride/solvers/bnb.py end -----

# ----- file share_a_ride/solvers/greedy.py begin -----
import time
# from share_a_ride.problem import ShareARideProblem
# from share_a_ride.solution import Solution
# from utils.helper import route_cost_from_sequence

def greedy_balanced_solver(prob: ShareARideProblem, verbose: bool = False) -> tuple:
    """
    Greedy balanced heuristic:
        - At each step, choose the taxi with smallest current route cost
        - For that taxi, evaluate all feasible next actions (pick/drop).
        - Choose the action with minimal added distance (inc).
        - Repeat until all passengers and parcels are served.

    Returns:
        (sol, info): tuple where
        - sol: Solution object with routes and costs.
        - Info dictionary contains:
            + iterations: number of main loop iterations
            + actions_evaluated: total number of actions evaluated
            + elapsed_time: total time taken
    """
    start_time = time.time()
    N, M, K = prob.N, prob.M, prob.K

    # state
    remaining_pass_pick = set(range(1, N + 1))
    remaining_parc_pick = set(range(1, M + 1))
    taxi_current_pass = [0] * K
    taxi_current_parc = [set() for _ in range(K)]

    taxi_states = [
        {   
            "route": [0], 
            "pos": 0,
            "cost": 0, 
            "load": 0, 
            "passenger": 0, 
            "ended": False
        }
        for _ in range(K)
    ]

    # Statistics tracking
    stats = {
        "iterations": 0,
        "actions_evaluated": 0,
    }


    def possible_actions(t_state: dict, t_idx: int):
        """
            Generate feasible next actions for a taxi t.
            Returns a list of (action_type, node_index, incremental_cost).
        """
        pos, actions = t_state["pos"], []

        # passenger pickup
        for i in list(remaining_pass_pick):
            if t_state["passenger"] == 0:  # no passenger onboard
                inc = prob.D[pos][prob.ppick(i)]
                actions.append(("pickP", i, inc))

        # passenger drop
        if t_state["passenger"] > 0:  # passenger onboard
            inc = prob.D[pos][prob.pdrop(t_state["passenger"])]
            actions.append(("dropP", t_state["passenger"], inc))

        # parcel pickup
        for j in list(remaining_parc_pick):
            qj = prob.q[j - 1]
            if t_state["load"] + qj <= prob.Q[t_idx]:
                inc = prob.D[pos][prob.parc_pick(j)]
                actions.append(("pickL", j, inc))

        # parcel drop
        for j in taxi_current_parc[t_idx]:
            inc = prob.D[pos][prob.parc_drop(j)]
            actions.append(("dropL", j, inc))

        return actions


    def apply_action(t_state: dict, t_idx: int, kind: str, node_idx: int, inc: int):
        """Apply an action to taxi t and update global sets."""
        nonlocal remaining_pass_pick, remaining_parc_pick
        nonlocal taxi_current_pass, taxi_current_parc

        if kind == "pickP":
            t_state["route"].append(prob.ppick(node_idx))
            t_state["passenger"] = node_idx
            remaining_pass_pick.remove(node_idx)
            taxi_current_pass[t_idx] = node_idx

        elif kind == "dropP":
            t_state["route"].append(prob.pdrop(node_idx))
            t_state["passenger"] = 0
            taxi_current_pass[t_idx] = 0

        elif kind == "pickL":
            t_state["route"].append(prob.parc_pick(node_idx))
            t_state["load"] += prob.q[node_idx - 1]
            remaining_parc_pick.remove(node_idx)
            taxi_current_parc[t_idx].add(node_idx)

        elif kind == "dropL":
            t_state["route"].append(prob.parc_drop(node_idx))
            t_state["load"] -= prob.q[node_idx - 1]
            taxi_current_parc[t_idx].remove(node_idx)

        else:
            raise ValueError(f"Unknown action kind: {kind}")

        t_state["cost"] += inc
        t_state["pos"] = t_state["route"][-1]


    # Main loop: execute actions until all pickups and drops are resolved.
    while ( remaining_pass_pick or any(taxi_current_pass)
            or remaining_parc_pick or any(taxi_current_pass)):

        stats["iterations"] += 1

        # List all available taxis
        available_taxis = [
            t_idx for t_idx, t_state in enumerate(taxi_states)
            if not t_state["ended"]
        ]

        # Select taxi with minimal route cost.
        argmin_t_idx = min(available_taxis, key=lambda t_idx: taxi_states[t_idx]["cost"])
        argmin_t_state = taxi_states[argmin_t_idx]

        # Get feasible actions and record how many were evaluated.
        actions = possible_actions(argmin_t_state, argmin_t_idx)
        stats["actions_evaluated"] += len(actions)

        if verbose:
            print(f"Taxi with min cost: {argmin_t_idx}")
            print(f"Actions available: {actions}")

        # If no feasible action exists, send taxi back to depot.
        if not actions:
            assert argmin_t_state["passenger"] == 0
            assert argmin_t_state["load"] == 0

            argmin_t_state["route"].append(0)
            argmin_t_state["pos"] = 0
            argmin_t_state["cost"] += prob.D[argmin_t_state["pos"]][0]
            argmin_t_state["ended"] = True

            continue

        # Select action with minimal incremental cost.
        kind, idx, inc = min(actions, key=lambda x: x[2])
        apply_action(argmin_t_state, argmin_t_idx, kind, idx, inc)
        
        if verbose:
            print(f"Taxi: {argmin_t_idx}: {argmin_t_state['route']}")
            print()

    for t_state in taxi_states:
        if not t_state["ended"]:
            t_state["route"].append(0)
            t_state["ended"] = True

    if verbose:
        print("All tasks completed.")


    # Build final solution.
    routes, route_costs = [], []
    for t_state in taxi_states:
        routes.append(t_state["route"])
        cost = route_cost_from_sequence(t_state["route"], prob.D)
        route_costs.append(cost)

    sol = Solution(prob, routes, route_costs)
    elapsed = time.time() - start_time
    info = {
        "iterations": stats["iterations"],
        "actions_evaluated": stats["actions_evaluated"],
        "elapsed_time": elapsed
    }
    if sol:
        if not sol.is_valid():
            sol = None


    assert sol.is_valid() if sol else True
    return sol, info
# ----- file share_a_ride/solvers/greedy.py end -----
# from share_a_ride.problem import ShareARideProblem

# ----- file utils/generators.py begin -----
import math
import random
from typing import Optional, List, Tuple

# from share_a_ride.problem import ShareARideProblem

# ----- file utils/visualization.py begin -----
from typing import List, Tuple

def visualize_instance(
    coords: List[Tuple[float, float]], N: int, M: int, K: int
):
    """Visualize instance with matplotlib, if available."""
    try:
        import matplotlib.pyplot as plt

        xs = [x for x, _ in coords]
        ys = [y for _, y in coords]

        plt.figure(figsize=(6, 6))
        plt.scatter(xs[1:], ys[1:], label="Nodes")
        plt.scatter(xs[0], ys[0], c="red", label="Depot")

        for idx, (x, y) in enumerate(coords):
            plt.text(x, y, str(idx), fontsize=8)

        plt.title(f"Instance coords N={N}, M={M}, K={K}")
        plt.legend()
        plt.show()

    except Exception as e:
        print("Visualization failed (matplotlib not available):", e)
# ----- file utils/visualization.py end -----


def _generate_cost_value(
        i: int, j: int, D: List[List[int]], rng: random.Random, 
        low: int, high: int, lmbd: Optional[float], asymmetric: bool
    ) -> int:
    """Generate a single distance value, symmetric if needed."""
    if i == j:
        return 0
    if asymmetric:
        if lmbd is not None:
            return _sample_poisson(rng, low, high, lmbd)
        return rng.randint(low, high)
    if j < i:
        return D[j][i]
    if lmbd is not None:
        return _sample_poisson(rng, low, high, lmbd)
    return rng.randint(low, high)


def _sample_poisson(
    rng: random.Random,
    low: int,
    high: int,
    lmbd: float,
    ) -> int:
    """Sample from Poisson(lmbd) until result is in [low, high]."""

    while True:
        # Parameters
        L = math.exp(-lmbd)
        k = 0
        p = 1.0

        # Generate Poisson sample using Knuth's algorithm
        while p > L:
            k += 1
            p *= rng.random()
        value = k - 1

        # Check range and return
        if low <= value <= high:
            return value


# ---------------------- Test instance generators --------------------------------
def random_distance_matrix(
        n: int,
        low: int = 5,
        high: int = 20,
        lmbd: Optional[float] = None,
        asymmetric: bool = False,
        seed: Optional[int] = None
    ) -> List[List[int]]:
    """
    Generate a random symmetric or asymmetric distance matrix.
    """
    rng = random.Random(seed)
    D = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            D[i][j] = _generate_cost_value(i, j, D, rng, low, high, lmbd, asymmetric)

    return D


def euclidean_distance_matrix(
        coords: List[Tuple[float, float]]
    ) -> List[List[int]]:
    """Compute pairwise Euclidean distance matrix from coordinates, rounding distances to the nearest integer."""
    n = len(coords)
    D = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            dist = int(round(math.hypot(
                coords[i][0] - coords[j][0],
                coords[i][1] - coords[j][1]
            )))
            D[i][j] = D[j][i] = dist
    return D


def generate_instance_lazy(
        N: int, M: int, K: int,
        low: int = 10, high: int = 50, lmbd: float = 20.0,
        qlow: int = 5, qhigh: int = 15, qlmbd: float = 10.0,
        Qlow: int = 15, Qhigh: int = 30, Qlmbd: float = 20.0,
        use_poisson: bool = False,
        seed: Optional[int] = None
    ) -> ShareARideProblem:
    """Generate random instance using lazy distance matrix."""
    rng = random.Random(seed)
    n_nodes = 2*N + 2*M + 1

    if use_poisson:
        q = [_sample_poisson(rng, qlow, qhigh, qlmbd) for _ in range(M)]
        Q = [_sample_poisson(rng, Qlow, Qhigh, Qlmbd) for _ in range(K)]
        D = random_distance_matrix(n_nodes, low=low, high=high, lmbd=lmbd,
                               asymmetric=True, seed=seed)

    else:
        q = [rng.randint(qlow, qhigh) for _ in range(M)]
        Q = [rng.randint(Qlow, Qhigh) for _ in range(K)]
        D = random_distance_matrix(n_nodes, low=low, high=high, lmbd=None,
                               asymmetric=True, seed=seed)
    
    return ShareARideProblem(N, M, K, q, Q, D)


def generate_instance_coords(
        N: int, M: int, K: int,
        area: float = 100.0,
        qlow: int = 5, qhigh: int = 15, qlmbd: float = 10.0,
        Qlow: int = 15, Qhigh: int = 30, Qlmbd: float = 20.0,
        seed: Optional[int] = None,
        visualize: bool = False
    ) -> ShareARideProblem:

    """
    Generate instance with coordinates and optional visualization with matplotlib.
    """
    
    rng = random.Random(seed)
    total_points = 1 + 2 * N + 2 * M

    # Generate depot and random coordinates for all other points
    coords = [(area / 2.0, area / 2.0)] + [
        (rng.random() * area, rng.random() * area)
        for _ in range(total_points - 1)
    ]

    D = euclidean_distance_matrix(coords)
    q = [rng.randint(qlow, qhigh) for _ in range(M)]
    Q = [rng.randint(Qlow, Qhigh) for _ in range(K)]
    prob = ShareARideProblem(N, M, K, q, Q, D)

    if visualize:
        visualize_instance(coords, N, M, K)

    return prob
# ----- file utils/generators.py end -----


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
    # prob: ShareARideProblem = generate_instance_coords(5, 5, 3, qlow=5, qhigh=12, qlmbd=7)
    # prob.stdin_print()

    prob: ShareARideProblem = read_instance()
    sol, info1 = greedy_balanced_solver(prob, verbose=False)

    if not sol:
        print("No solution in phase 1")

    if (not sol) or (prob.num_nodes <= 21):
        sol, info2 = branch_and_bound(
            prob,
            time_limit=10.0,
            target_cost=sol.max_cost if sol else None
        )

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
# ----- file oj_final.py end -----
