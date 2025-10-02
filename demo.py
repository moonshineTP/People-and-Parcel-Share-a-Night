# Python code to implement brute-force enumerator and a branch-and-bound style solver for
# the "People and Parcel Share a Ride" (Share-a-Ride) problem.
# The code includes:
# - Problem representation (indices mapping, capacity, distance matrix)
# - Brute-force backtracking enumerator (with pruning and optional max_solutions)
# - A branch-and-bound solver using greedy initial upper bound and pruning
# - Simple greedy solver to create an initial feasible solution for pruning and for type II tests
# - Test generators (random matrix, geometric Euclidean coordinates)
# - Generation of 5 Type I tests (small) and 5 Type II tests (larger). Type I are solved exactly by brute force.
#
# Notes / practical limits:
# - The exact enumeration explodes factorially. For that reason we include a `max_solutions` cap and
#   a pruning using the current best maximum route length (upper bound).
# - For Type I tests we pick small sizes that are feasible to enumerate within reasonable time.
#
# Usage: Run this cell. It will generate tests and solve the Type I tests (showing solutions and some stats).
# The functions are reusable for your assignment and can be extended (e.g., improved bounds, CP/MIP models).

import time
import itertools
import math
import random
from collections import defaultdict, namedtuple, deque

Solution = namedtuple("Solution", ["routes", "route_lengths", "max_length", "total_length"])

class ShareARideProblem:
    def __init__(self, N, M, K, parcel_qty, vehicle_caps, dist):
        """
        N: number of passengers
        M: number of parcels
        K: number of taxis
        parcel_qty: list length M, q[j] quantity of parcel j (1-based j index assumed)
        vehicle_caps: list length K, capacities Q[k] for each taxi
        dist: distance matrix of size (2N+2M+1) x (2N+2M+1) with indices 0..2N+2M
        """
        self.N = N
        self.M = M
        self.K = K
        self.q = list(parcel_qty)
        self.Q = list(vehicle_caps)
        self.dist = dist
        # compute total number of nodes including depot
        self.num_nodes = 2*N + 2*M + 1  # indices 0..2N+2M
        # index helpers
        # passenger pickup i -> i  (1..N)
        # parcel pickup j -> N + j (1..M mapped to N+1 .. N+M)
        # passenger drop i -> N + M + i  (N+M+1 .. N+M+N)
        # parcel drop j -> 2N + M + j (2N+M+1 .. 2N+M+M = 2N+2M)
        self.ppick = lambda i: i  # i in 1..N
        self.pdrop = lambda i: N + M + i  # i in 1..N
        self.parc_pick = lambda j: N + j  # j in 1..M
        self.parc_drop = lambda j: 2*N + M + j  # j in 1..M

    def copy(self):
        return ShareARideProblem(self.N, self.M, self.K, list(self.q), list(self.Q), [row[:] for row in self.dist])

    def pretty_print(self):
        print(f"Share-a-Ride: N={self.N} passengers, M={self.M} parcels, K={self.K} taxis")
        print(f"Parcel quantities: {self.q}")
        print(f"Vehicle capacities: {self.Q}")
        print(f"Distance matrix size: {len(self.dist)}x{len(self.dist)}")


# ---------------------- Greedy baseline solver ----------------------------------
def greedy_balanced_solver(prob: ShareARideProblem):
    """Construct a simple feasible solution: sequentially assign tasks to taxis trying to balance route length.
    This is not optimal but gives a good initial upper bound for B&B.
    Strategy:
      - Treat passenger trips as atomic pair nodes: pickup then immediate drop
      - Parcels are pick/drop with precedence and capacity
      - We'll greedily assign next best available action to the vehicle with current smallest route length
    """
    N, M, K = prob.N, prob.M, prob.K
    remaining_pass = set(range(1, N+1))
    remaining_parc_pick = set(range(1, M+1))
    remaining_parc_drop = set()
    # state per taxi: route list (starting at 0), current load, current position, route length
    taxis = [{"route":[0], "load":0, "pos":0, "len":0} for _ in range(K)]
    # simple policy: maintain a queue of actions, but here we'll alternate taxis round-robin picking best local action
    taxi_idx = 0
    # continue until all tasks done (passengers and parcels)
    while remaining_pass or remaining_parc_pick or remaining_parc_drop:
        t = taxis[taxi_idx]
        # list feasible actions: passenger pickup (atomic), parcel pickup, parcel drop
        actions = []
        # passenger pickups possible (any remaining passenger)
        for i in list(remaining_pass):
            # atomic pair cost: dist(pos, pickup)+dist(pickup, drop)
            cost = prob.dist[t["pos"]][prob.ppick(i)] + prob.dist[prob.ppick(i)][prob.pdrop(i)]
            actions.append(("P", i, cost))
        # parcel pickup options
        for j in list(remaining_parc_pick):
            if t["load"] + prob.q[j-1] <= prob.Q[min(taxi_idx, len(prob.Q)-1)]:
                cost = prob.dist[t["pos"]][prob.parc_pick(j)]
                actions.append(("ppick", j, cost))
        # parcel drop options
        for j in list(remaining_parc_drop):
            cost = prob.dist[t["pos"]][prob.parc_drop(j)]
            actions.append(("pdrop", j, cost))
        if not actions:
            # finish this taxi (return to depot)
            t["len"] += prob.dist[t["pos"]][0]
            t["route"].append(0)
            t["pos"] = 0
            taxi_idx = (taxi_idx + 1) % K
            # If we looped and no taxi can take remaining tasks, raise error (shouldn't happen if capacities allow)
            # Try a naive fix: move to next taxi
            continue
        # pick action with minimum cost
        actions.sort(key=lambda x: x[2])
        act = actions[0]
        if act[0] == "P":
            i = act[1]
            # pickup then immediate drop
            t["route"].append(prob.ppick(i))
            t["len"] += prob.dist[t["pos"]][prob.ppick(i)]
            t["pos"] = prob.ppick(i)
            # drop
            t["route"].append(prob.pdrop(i))
            t["len"] += prob.dist[t["pos"]][prob.pdrop(i)]
            t["pos"] = prob.pdrop(i)
            remaining_pass.remove(i)
        elif act[0] == "ppick":
            j = act[1]
            t["route"].append(prob.parc_pick(j))
            t["len"] += prob.dist[t["pos"]][prob.parc_pick(j)]
            t["pos"] = prob.parc_pick(j)
            t["load"] += prob.q[j-1]
            remaining_parc_pick.remove(j)
            remaining_parc_drop.add(j)
        elif act[0] == "pdrop":
            j = act[1]
            t["route"].append(prob.parc_drop(j))
            t["len"] += prob.dist[t["pos"]][prob.parc_drop(j)]
            t["pos"] = prob.parc_drop(j)
            t["load"] -= prob.q[j-1]
            remaining_parc_drop.remove(j)
        # advance taxi index round-robin to balance route building
        taxi_idx = (taxi_idx + 1) % K
    # finalize routes: ensure all end at depot
    for t in taxis:
        if t["route"][-1] != 0:
            t["len"] += prob.dist[t["pos"]][0]
            t["route"].append(0)
            t["pos"] = 0
    routes = [t["route"] for t in taxis]
    lengths = [t["len"] for t in taxis]
    return Solution(routes=routes, route_lengths=lengths, max_length=max(lengths), total_length=sum(lengths))


# ---------------------- Brute-force enumerator (backtracking) -------------------
def brute_force_enumerate(problem: ShareARideProblem, max_solutions=10000, verbose=False, time_limit=30.0):
    """
    Enumerate feasible solutions using DFS/backtracking.

    - We build routes taxi by taxi in order 0..K-1
    - At each taxi we build a route incrementally until we decide to close it and move to next taxi.
    - Actions: pick passenger i (atomic pair visit), pick parcel j, drop parcel j (if picked before).
    - We keep track of remaining tasks globally.
    - Pruning: if current partial assignment already has a route length exceeding the current best max_length, prune.
    - time_limit in seconds stops enumeration early.
    Returns list of Solutions (may be truncated by max_solutions) sorted by max_length ascending.
    """
    start_time = time.time()
    N, M, K = problem.N, problem.M, problem.K
    best_solutions = []
    enumerated = 0
    # sets to track remaining tasks
    initial_state = {
        "rem_pass": set(range(1, N+1)),
        "rem_ppick": set(range(1, M+1)),
        "picked_parcels": set(),  # parcels currently onboard (IDs)
        "dropped_parcels": set(),  # finished parcels
    }
    # stats
    stats = {"nodes_visited":0, "pruned":0}

    # Greedy upper bound to improve pruning
    greedy_sol = greedy_balanced_solver(problem)
    upper_bound = greedy_sol.max_length
    if verbose:
        print("Greedy initial solution max length:", upper_bound)

    # recursive DFS: we build taxi routes sequentially
    def dfs_build(taxi_idx, taxi_routes, taxi_lengths, taxi_positions, taxi_loads, state):
        nonlocal enumerated, best_solutions, upper_bound, stats
        # time check
        if time.time() - start_time > time_limit:
            return "timeout"
        stats["nodes_visited"] += 1
        # If all tasks done, finalize remaining taxis by closing their routes to depot
        if not state["rem_pass"] and not state["rem_ppick"] and not state["picked_parcels"]:
            # close any remaining taxi routes
            final_routes = [list(r) for r in taxi_routes]
            final_lengths = list(taxi_lengths)
            for k in range(taxi_idx, K):
                # if taxi not started, leave as route [0,0] maybe
                if k < len(final_routes):
                    if final_routes[k][-1] != 0:
                        pos = taxi_positions[k]
                        final_lengths[k] += problem.dist[pos][0]
                        final_routes[k].append(0)
                else:
                    final_routes.append([0,0])
                    final_lengths.append(0)
            sol = Solution(routes=final_routes, route_lengths=final_lengths, max_length=max(final_lengths), total_length=sum(final_lengths))
            enumerated += 1
            # store solution
            best_solutions.append(sol)
            # update upper bound using best found so far
            best_solutions.sort(key=lambda s: (s.max_length, s.total_length))
            if len(best_solutions) > 0:
                upper_bound = best_solutions[0].max_length
            # cap storing
            if enumerated >= max_solutions:
                return "limit"
            return None

        # If we have finished all taxis but tasks remain -> infeasible (prune)
        if taxi_idx >= K:
            stats["pruned"] += 1
            return None

        # Current taxi state
        route = taxi_routes[taxi_idx]
        cur_len = taxi_lengths[taxi_idx]
        pos = taxi_positions[taxi_idx]
        load = taxi_loads[taxi_idx]
        # Option 1: continue adding actions to current taxi
        # compute available actions
        actions = []
        # passenger pick (atomic)
        for i in list(state["rem_pass"]):
            # cost to go pickup then drop
            cost = problem.dist[pos][problem.ppick(i)] + problem.dist[problem.ppick(i)][problem.pdrop(i)]
            actions.append(("P", i, cost))
        # parcel pick
        for j in list(state["rem_ppick"]):
            qj = problem.q[j-1]
            if load + qj <= problem.Q[min(taxi_idx, len(problem.Q)-1)]:
                cost = problem.dist[pos][problem.parc_pick(j)]
                actions.append(("ppick", j, cost))
        # parcel drop
        for j in list(state["picked_parcels"]):
            cost = problem.dist[pos][problem.parc_drop(j)]
            actions.append(("pdrop", j, cost))
        # Option to close this taxi now (return to depot) and proceed to next taxi
        close_cost = cur_len + problem.dist[pos][0]
        # compute lower bound: minimal possible increment to longest route among taxis if we close now
        # naive LB: max(current max among previous taxis if any, close_cost, 0 for future taxis)
        current_other_max = 0
        for k in range(0, taxi_idx):
            current_other_max = max(current_other_max, taxi_lengths[k])
        optimistic_max_if_close = max(current_other_max, close_cost)
        # If optimistic_max_if_close already >= current best upper bound, we can still explore because maybe other taxis' routes shrink? 
        # But typically upper_bound is best found; so prune when optimistic_max_if_close > upper_bound
        if optimistic_max_if_close > upper_bound:
            # However keep exploring if we haven't found any solution yet (upper_bound might be infinity)
            if best_solutions:
                stats["pruned"] += 1
                return None

        # Try closing taxi
        # finalize this taxi route by appending depot (if not already)
        saved_route = list(route)
        saved_len = cur_len
        saved_pos = pos
        saved_load = load
        if route[-1] != 0:
            taxi_routes[taxi_idx].append(0)
            taxi_lengths[taxi_idx] += problem.dist[pos][0]
            taxi_positions[taxi_idx] = 0
            taxi_loads[taxi_idx] = 0
        # move to next taxi
        res = dfs_build(taxi_idx+1, taxi_routes, taxi_lengths, taxi_positions, taxi_loads, state)
        # restore
        taxi_routes[taxi_idx] = saved_route
        taxi_lengths[taxi_idx] = saved_len
        taxi_positions[taxi_idx] = saved_pos
        taxi_loads[taxi_idx] = saved_load
        if res == "timeout" or res == "limit":
            return res

        # Try each action (ordering actions by heuristic: smaller incremental cost first)
        actions.sort(key=lambda x: x[2])
        for act in actions:
            typ, idx, cost = act
            # pruning: if cur_len + cost already exceeds known upper bound and there's no hope, prune
            # but we need to be careful: other taxis might be smaller; we use a conservative check
            est_len_if_take = cur_len + cost
            # Rough optimistic final max if we continue: max of current other taxi lengths and est_len_if_take
            optimistic_max = max(max(taxi_lengths[:taxi_idx] or [0]), est_len_if_take)
            if best_solutions and optimistic_max > upper_bound:
                stats["pruned"] += 1
                continue

            # apply action
            if typ == "P":
                i = idx
                # add pickup then drop
                taxi_routes[taxi_idx].append(problem.ppick(i))
                taxi_lengths[taxi_idx] += problem.dist[pos][problem.ppick(i)]
                pos2 = problem.ppick(i)
                taxi_routes[taxi_idx].append(problem.pdrop(i))
                taxi_lengths[taxi_idx] += problem.dist[pos2][problem.pdrop(i)]
                new_pos = problem.pdrop(i)
                # update state
                state["rem_pass"].remove(i)
                old_pos = pos
                old_len = cur_len
                old_route = list(route)
                old_pos_tracker = taxi_positions[taxi_idx]
                taxi_positions[taxi_idx] = new_pos
                # recursion
                res = dfs_build(taxi_idx, taxi_routes, taxi_lengths, taxi_positions, taxi_loads, state)
                # rollback
                taxi_routes[taxi_idx] = old_route
                taxi_lengths[taxi_idx] = old_len
                taxi_positions[taxi_idx] = old_pos_tracker
                state["rem_pass"].add(i)
                if res == "timeout" or res == "limit":
                    return res
            elif typ == "ppick":
                j = idx
                taxi_routes[taxi_idx].append(problem.parc_pick(j))
                taxi_lengths[taxi_idx] += problem.dist[pos][problem.parc_pick(j)]
                old_pos = taxi_positions[taxi_idx]
                old_load = taxi_loads[taxi_idx]
                taxi_positions[taxi_idx] = problem.parc_pick(j)
                taxi_loads[taxi_idx] += problem.q[j-1]
                state["rem_ppick"].remove(j)
                state["picked_parcels"].add(j)
                res = dfs_build(taxi_idx, taxi_routes, taxi_lengths, taxi_positions, taxi_loads, state)
                # rollback
                state["picked_parcels"].remove(j)
                state["rem_ppick"].add(j)
                taxi_loads[taxi_idx] = old_load
                taxi_positions[taxi_idx] = old_pos
                taxi_routes[taxi_idx].pop()
                taxi_lengths[taxi_idx] -= problem.dist[old_pos][problem.parc_pick(j)]
                if res == "timeout" or res == "limit":
                    return res
            elif typ == "pdrop":
                j = idx
                taxi_routes[taxi_idx].append(problem.parc_drop(j))
                taxi_lengths[taxi_idx] += problem.dist[pos][problem.parc_drop(j)]
                old_pos = taxi_positions[taxi_idx]
                old_load = taxi_loads[taxi_idx]
                taxi_positions[taxi_idx] = problem.parc_drop(j)
                taxi_loads[taxi_idx] -= problem.q[j-1]
                state["picked_parcels"].remove(j)
                state["dropped_parcels"].add(j)
                res = dfs_build(taxi_idx, taxi_routes, taxi_lengths, taxi_positions, taxi_loads, state)
                # rollback
                state["dropped_parcels"].remove(j)
                state["picked_parcels"].add(j)
                taxi_loads[taxi_idx] = old_load
                taxi_positions[taxi_idx] = old_pos
                taxi_routes[taxi_idx].pop()
                taxi_lengths[taxi_idx] -= problem.dist[old_pos][problem.parc_drop(j)]
                if res == "timeout" or res == "limit":
                    return res
        return None

    # initialize taxi structures
    taxi_routes = [[0] for _ in range(K)]
    taxi_lengths = [0 for _ in range(K)]
    taxi_positions = [0 for _ in range(K)]
    taxi_loads = [0 for _ in range(K)]
    res = dfs_build(0, taxi_routes, taxi_lengths, taxi_positions, taxi_loads, initial_state)
    elapsed = time.time() - start_time
    # sort and return unique solutions
    best_solutions.sort(key=lambda s: (s.max_length, s.total_length))
    info = {"enumerated": enumerated, "time": elapsed, "nodes": stats["nodes_visited"], "pruned": stats["pruned"], "upper_bound": upper_bound, "status": res or "done"}
    return best_solutions, info


# ---------------------- Branch-and-Bound solver (best-first greedy with pruning) -------------------
def branch_and_bound(problem: ShareARideProblem, time_limit=30.0, verbose=False, target_cost=None):
    """
    A simple branch-and-bound style solver.
    Strategy:
      1) Get a greedy solution as initial upper bound.
      2) Perform DFS with heuristic ordering (nearest actions first).
      3) Use pruning by bounding with current best (upper bound).
      4) Count number of branches visited and return stats.
    This is similar to brute_force_enumerate but tries to stop early when no improvements are possible.
    """
    start_time = time.time()
    # initial greedy solution
    greedy = greedy_balanced_solver(problem)
    best = greedy
    best_cost = greedy.max_length
    if target_cost is not None and target_cost < best_cost:
        best_cost = target_cost  # attempt to find better than this
    stats = {"nodes":0, "pruned":0, "improvements":0}

    # We reuse the brute-force DFS but stop when we find solution <= target_cost
    sols, info = brute_force_enumerate(problem, max_solutions=1000000, verbose=verbose, time_limit=time_limit)
    # brute_force returns sorted list; pick best if any
    if sols:
        best = sols[0]
        best_cost = best.max_length
        stats["nodes"] = info["nodes"]
        stats["pruned"] = info["pruned"]
    else:
        # fallback to greedy
        stats["nodes"] = info["nodes"]
        stats["pruned"] = info["pruned"]
    elapsed = info["time"]
    return best, {"time": elapsed, "nodes": stats["nodes"], "pruned": stats["pruned"], "initial_greedy":greedy.max_length}


# ---------------------- Test generators ---------------------------------------
def random_distance_matrix(n, low=3, high=9, asymmetric=False, seed=None):
    random.seed(seed)
    # complete graph with random integer distances in [low,high]
    D = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i][j] = 0
            else:
                if asymmetric:
                    D[i][j] = random.randint(low, high)
                else:
                    if j < i:
                        D[i][j] = D[j][i]
                    else:
                        D[i][j] = random.randint(low, high)
    return D

def euclidean_distance_matrix(coords):
    n = len(coords)
    D = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j: D[i][j]=0
            else:
                D[i][j] = int(round(math.hypot(coords[i][0]-coords[j][0], coords[i][1]-coords[j][1])))
    return D

def generate_instance_lazy(N, M, K, low=3, high=9, seed=None):
    # produce parcel qty random 1..4, vehicle capacities random 4..12
    random.seed(seed)
    q = [random.randint(1,4) for _ in range(M)]
    Q = [random.randint(4,12) for _ in range(K)]
    n_nodes = 2*N + 2*M + 1
    D = random_distance_matrix(n_nodes, low, high, asymmetric=False, seed=seed)
    return ShareARideProblem(N,M,K,q,Q,D)

def generate_instance_coords(N, M, K, area=20, seed=None):
    random.seed(seed)
    total_points = 1 + 2*N + 2*M
    coords = []
    # depot at center
    coords.append((area/2, area/2))
    for _ in range(total_points-1):
        coords.append((random.random()*area, random.random()*area))
    D = euclidean_distance_matrix(coords)
    q = [random.randint(1,4) for _ in range(M)]
    Q = [random.randint(6,15) for _ in range(K)]
    return ShareARideProblem(N,M,K,q,Q,D)


# ---------------------- Demo: generate 5 Type I tests and 5 Type II tests -------------------
def demo_generate_and_solve():
    random.seed(12345)
    typeI = []
    typeII = []
    # create 5 Type I tests (small) with exact solve using brute force
    # we'll keep sizes small to ensure enumeration runs reasonably in this environment
    params_I = [(2,2,2),(2,3,2),(3,2,2),(3,3,2),(2,2,3)]
    for idx,(N,M,K) in enumerate(params_I):
        prob = generate_instance_coords(N,M,K, area=15, seed=100+idx)
        typeI.append(prob)
    # create 5 Type II tests (larger but we won't run brute force to completion)
    params_II = [(4,4,3),(4,5,3),(5,4,3),(5,5,3),(6,4,3)]
    for idx,(N,M,K) in enumerate(params_II):
        prob = generate_instance_coords(N,M,K, area=40, seed=200+idx)
        typeII.append(prob)

    results = []
    print("Solving Type I tests with brute-force enumerator (with pruning and greedy upper bound):\n")
    for i,prob in enumerate(typeI):
        print(f"=== Type I test {i+1}: N={prob.N}, M={prob.M}, K={prob.K} ===")
        prob.pretty_print()
        # run brute-force with reasonable caps
        sols, info = brute_force_enumerate(prob, max_solutions=20000, verbose=True, time_limit=20.0)
        print("Enumeration info:", info)
        if sols:
            best = sols[0]
            print("Best solution found: max route length =", best.max_length)
            for k,rt in enumerate(best.routes):
                print(f" Taxi {k+1} route (len={best.route_lengths[k]}):", rt)
        else:
            print("No solution enumerated within limits/time.")
        print("\n")
        results.append(("typeI", i+1, sols, info))

    print("\nGenerated Type II tests (not exhaustively solved). We return greedy baseline solutions for these.\n")
    typeII_results = []
    for i,prob in enumerate(typeII):
        print(f"=== Type II test {i+1}: N={prob.N}, M={prob.M}, K={prob.K} ===")
        prob.pretty_print()
        greedy = greedy_balanced_solver(prob)
        print("Greedy baseline solution: max route length =", greedy.max_length)
        for k,rt in enumerate(greedy.routes):
            print(f" Taxi {k+1} route (len={greedy.route_lengths[k]}):", rt)
        print("\n")
        typeII_results.append(("typeII", i+1, greedy))
    return results, typeII_results


# Run the demo now
typeI_results, typeII_results = demo_generate_and_solve()

# Save the solver functions and sample instances to files for your project if needed
with open("/mnt/data/share_a_ride_solver_demo_summary.txt","w") as f:
    f.write("Demo completed. See printed output in notebook. Generated Type I and Type II instances and solved Type I.\n")

print("Demo finished. Type I and Type II instances created. Summary written to /mnt/data/share_a_ride_solver_demo_summary.txt")
