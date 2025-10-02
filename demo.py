import time
import itertools
import math
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
import sys

# ---------------------- Data classes -------------------------------------------
class Solution:
    """
    Solution object representing K routes.
    - routes: list of lists of node indices (each route starts at depot implicitly at 0 and ends at 0)
    - route_lengths: list of ints
    - max_length: int (objective to minimize)
    """
    def __init__(self, routes: List[List[int]], route_lengths: List[int]):
        assert len(routes) == len(route_lengths)
        self.routes = routes
        self.route_lengths = route_lengths
        self.max_length = max(route_lengths) if route_lengths else 0

    def is_valid(self) -> bool:
        """Quick validity check: route lengths consistent with routes (not enforced here)."""
        return len(self.routes) == len(self.route_lengths)

    def pretty_print(self):
        print("Solution:")
        for k, (r, L) in enumerate(zip(self.routes, self.route_lengths)):
            print(f" Taxi {k+1}: len={L}, route={r}")
        print(" Max route length (objective) =", self.max_length)


class ShareARideProblem:
    def __init__(self, N:int, M:int, K:int, parcel_qty:List[int], vehicle_caps:List[int], dist:List[List[int]]):
        """
        N: #passengers
        M: #parcels
        K: #taxis
        parcel_qty: length M, q[j-1] is volume of parcel j (1-based index in problem mapping)
        vehicle_caps: length K, Q[k] capacity for taxi k
        dist: distance matrix of size (2*N + 2*M + 1) (indices 0..2N+2M). We store as D (capital) for emphasis.
        """
        self.N = N
        self.M = M
        self.K = K
        self.q = list(parcel_qty)
        self.Q = list(vehicle_caps)
        # store as D; keep alias dist for backward compatibility
        self.D = [row[:] for row in dist]
        self.dist = self.D

        # indices:
        # depot = 0
        # passenger pickup i -> i (1..N)
        # parcel pickup j -> N + j (1..M mapped to N+1 .. N+M)
        # passenger drop i -> N + M + i  (N+M+1 .. N+M+N)
        # parcel drop j -> 2N + M + j    (2N+M+1 .. 2N+2M)
        self.num_nodes = 2*N + 2*M + 1

        # index helpers
        self.ppick = lambda i: i
        self.pdrop = lambda i: N + M + i
        self.parc_pick = lambda j: N + j
        self.parc_drop = lambda j: 2*N + M + j

    def copy(self):
        return ShareARideProblem(self.N, self.M, self.K, list(self.q), list(self.Q), [row[:] for row in self.D])

    def pretty_print(self, verbose:int=0):
        """
        verbose: 0 -> print N, M, K, num_nodes
                 1 -> additionally print q, Q, and D (full matrix)
        """
        print(f"Share-a-Ride: N={self.N} passengers, M={self.M} parcels, K={self.K} taxis, num_nodes={self.num_nodes}")
        if verbose >= 1:
            print("Parcel quantities (q):", self.q)
            print("Vehicle capacities (Q):", self.Q)
            print("Distance matrix D:")
            for row in self.D:
                print(" ", row)


# ---------------------- Utility functions -------------------------------------
def route_length_from_sequence(seq: List[int], D: List[List[int]]) -> int:
    """Compute route length given sequence of nodes (not including starting 0), route assumed: 0 -> seq -> 0"""
    pos = 0
    total = 0
    for node in seq:
        total += D[pos][node]
        pos = node
    total += D[pos][0]
    return total

# ---------------------- Greedy solver (modified) -------------------------------
def greedy_balanced_solver(prob: ShareARideProblem) -> Solution:
    """
    Greedy: always pick the taxi with smallest current route length,
    then add the single best (lowest incremental cost) feasible action for that taxi.
    Actions:
      - passenger atomic: pickup then immediate drop (two nodes appended)
      - parcel pickup or parcel drop (must respect capacity)
    This continues until all tasks done.
    """
    N, M, K = prob.N, prob.M, prob.K
    remaining_pass = set(range(1, N+1))
    remaining_parc_pick = set(range(1, M+1))
    remaining_parc_drop = set()
    # taxi states
    taxis = [{"route": [], "pos":0, "len":0, "load":0} for _ in range(K)]
    # ensure routes start at depot implicitly; we'll keep route as list of visited nodes (exclude initial 0)
    # loop until all tasks done
    while remaining_pass or remaining_parc_pick or remaining_parc_drop:
        # choose taxi with current smallest route length (ties by index)
        taxi_idx = min(range(K), key=lambda t: taxis[t]["len"])
        t = taxis[taxi_idx]
        pos = t["pos"]
        actions = []
        # passenger atomic
        for i in list(remaining_pass):
            inc = prob.D[pos][prob.ppick(i)] + prob.D[prob.ppick(i)][prob.pdrop(i)]
            actions.append(("P", i, inc))
        # parcel pickup
        for j in list(remaining_parc_pick):
            qj = prob.q[j-1]
            if t["load"] + qj <= prob.Q[min(taxi_idx, len(prob.Q)-1)]:
                inc = prob.D[pos][prob.parc_pick(j)]
                actions.append(("ppick", j, inc))
        # parcel drop
        for j in list(remaining_parc_drop):
            inc = prob.D[pos][prob.parc_drop(j)]
            actions.append(("pdrop", j, inc))
        if not actions:
            # finish taxi (return to depot)
            t["len"] += prob.D[pos][0]
            t["pos"] = 0
            # if taxi route empty, leave route empty (means it starts at depot and returns immediately)
            continue
        # pick smallest incremental action
        actions.sort(key=lambda x: x[2])
        act = actions[0]
        if act[0] == "P":
            i = act[1]
            # append pickup and drop
            t["route"].append(prob.ppick(i))
            t["len"] += prob.D[pos][prob.ppick(i)]
            pos = prob.ppick(i)
            t["route"].append(prob.pdrop(i))
            t["len"] += prob.D[pos][prob.pdrop(i)]
            pos = prob.pdrop(i)
            remaining_pass.remove(i)
        elif act[0] == "ppick":
            j = act[1]
            t["route"].append(prob.parc_pick(j))
            t["len"] += prob.D[pos][prob.parc_pick(j)]
            pos = prob.parc_pick(j)
            t["load"] += prob.q[j-1]
            remaining_parc_pick.remove(j)
            remaining_parc_drop.add(j)
        elif act[0] == "pdrop":
            j = act[1]
            t["route"].append(prob.parc_drop(j))
            t["len"] += prob.D[pos][prob.parc_drop(j)]
            pos = prob.parc_drop(j)
            t["load"] -= prob.q[j-1]
            remaining_parc_drop.remove(j)
        # update taxi pos
        t["pos"] = pos

    # finalize: ensure each taxi route returns to depot (add implied)
    route_lists = []
    lengths = []
    for t in taxis:
        if t["route"]:
            # length already includes edges, but ensure returned to depot if not added
            # our t["len"] already accounted for returns when no actions left? To be safe recompute
            L = route_length_from_sequence(t["route"], prob.D)
            route_lists.append([0] + t["route"] + [0])
            lengths.append(L)
        else:
            route_lists.append([0,0])
            lengths.append(0)
    return Solution(route_lists, lengths)


# ---------------------- Brute-force enumerator (exhaustive by PAIR assignment) --
def _assign_pairs_to_taxis_generator(num_pairs:int, K:int):
    """
    Yield each assignment as a tuple of length num_pairs, values in 0..K-1 meaning which taxi.
    This is equivalent to K^(num_pairs) possibilities; caller must ensure num_pairs small enough.
    """
    for t in itertools.product(range(K), repeat=num_pairs):
        yield t

def brute_force_enumerate(problem: ShareARideProblem,
                          max_solutions:int=100000,
                          verbose:bool=False,
                          time_limit:float=30.0) -> Tuple[List[Solution], Dict[str,Any]]:
    """
    Exhaustive enumeration as requested:
      - Treat each passenger (pair) and parcel (pair) as an atomic pair.
      - Enumerate all assignments of these pairs to K taxis (each pair assigned to exactly one taxi).
      - For each taxi, expand to the 2*items nodes and enumerate ALL permutations of those nodes.
      - For each permutation, check precedence (pickup before drop) and capacity (parcel loads).
      - Compute route lengths and collect feasible solutions.
    Returns: list of Solution objects sorted by max_length asc and info dict.
    """
    start = time.time()
    N, M, K = problem.N, problem.M, problem.K
    num_pairs = N + M
    solutions = []
    enumerated_assignments = 0
    enumerated_permutations = 0
    pruned_due_time = False

    # Pre-build a list of pairs: first N are passengers 'P1'..'PN', next M are parcels 'L1'..'LM'
    # But we'll store as tuples with type and index for clarity
    pairs = []
    for i in range(1, N+1):
        pairs.append(("P", i))
    for j in range(1, M+1):
        pairs.append(("L", j))

    # Enumerate assignments
    for assignment in _assign_pairs_to_taxis_generator(num_pairs, K):
        if time.time() - start > time_limit:
            pruned_due_time = True
            break
        enumerated_assignments += 1
        # Build lists of nodes per taxi based on assigned pairs (each pair contributes its two nodes)
        taxi_items = [[] for _ in range(K)]  # list of pair descriptors
        for pair_idx, taxi_id in enumerate(assignment):
            taxi_items[taxi_id].append(pairs[pair_idx])
        # Now for each taxi build the list of nodes (2*len)
        taxi_nodes = []
        feasible_assignment = True
        for t_idx in range(K):
            nodes = []
            for it in taxi_items[t_idx]:
                if it[0] == "P":  # passenger
                    i = it[1]
                    nodes.append(problem.ppick(i))
                    nodes.append(problem.pdrop(i))
                else:
                    j = it[1]
                    nodes.append(problem.parc_pick(j))
                    nodes.append(problem.parc_drop(j))
            taxi_nodes.append(nodes)

        # For each taxi, we need to permute nodes. The full solution is the Cartesian product of permutations
        # But to avoid building huge cartesian products when some taxi has many nodes, use nested loops
        # Prepare list of permutations per taxi
        perms_per_taxi = []
        skip_assignment = False
        for t_idx, nodes in enumerate(taxi_nodes):
            # all permutations of nodes
            if not nodes:
                perms_per_taxi.append([()])  # single empty tuple
            else:
                # generate permutations
                # To reduce duplicate permutations where pairs' pickups and drops swapped incorrectly,
                # we still enumerate all permutations and filter by precedence later.
                perms = list(itertools.permutations(nodes))
                perms_per_taxi.append(perms)
                # quick check size explosion
                total_perm_est = 1
                for p in perms_per_taxi:
                    total_perm_est *= max(1, len(p))
                if total_perm_est > max_solutions * 50:  # heuristic cap to avoid extreme blowup
                    # skip this assignment as too big for brute force
                    skip_assignment = True
                    break
        if skip_assignment:
            continue

        # Now iterate over product of permutations across taxis
        for combo in itertools.product(*perms_per_taxi):
            if time.time() - start > time_limit:
                pruned_due_time = True
                break
            enumerated_permutations += 1
            # combo is a tuple length K, each element is a permutation tuple for that taxi
            valid = True
            route_lists = []
            route_lengths = []
            # For each taxi check precedence + capacity along the route and compute length
            for t_idx, perm_nodes in enumerate(combo):
                seq = list(perm_nodes)
                # Check precedence: for every passenger i, both nodes must be in the sequence and pick occurs before drop
                # Since we assigned pair to same taxi, both must be present
                # For speed, build position map
                pos_map = {node: idx for idx, node in enumerate(seq)}
                # passengers
                for i in range(1, N+1):
                    pick = problem.ppick(i)
                    drop = problem.pdrop(i)
                    # if this pair wasn't assigned to this taxi, nodes won't be both present; skip checks
                    if pick in pos_map or drop in pos_map:
                        # If only one present -> invalid (shouldn't happen because pairs were assigned together), but keep check
                        if (pick in pos_map) != (drop in pos_map):
                            valid = False
                            break
                        if pos_map[pick] >= pos_map[drop]:
                            valid = False
                            break
                if not valid:
                    break
                # parcels: check pickup before drop and capacity along the route
                load = 0
                # We'll compute load stepwise
                for idx_n, node in enumerate(seq):
                    # if it's a parcel pickup
                    # parcel pickups are in range N+1 .. N+M
                    # detect by comparing with parcel_pick mapping
                    if 1 <= node <= problem.N:
                        # passenger pickup: does not change parcel load
                        pass
                    elif problem.N+1 <= node <= problem.N+problem.M:
                        # parcel pickup
                        j = node - problem.N
                        load += problem.q[j-1]
                        if load > problem.Q[min(t_idx, len(problem.Q)-1)]:
                            valid = False
                            break
                    elif problem.N+problem.M+1 <= node <= problem.N+problem.M+problem.N:
                        # passenger drop, no parcel load change
                        pass
                    elif 2*problem.N + problem.M + 1 <= node <= 2*problem.N + 2*problem.M:
                        # parcel drop
                        j = node - (2*problem.N + problem.M)
                        # careful mapping: actually parcel_drop(j) = 2N + M + j
                        # so j = node - (2N+M)
                        j = node - (2*problem.N + problem.M)
                        load -= problem.q[j-1]
                        if load < 0:
                            # shouldn't happen if pickups/drops consistent
                            valid = False
                            break
                    else:
                        # unknown node
                        pass
                if not valid:
                    break
                # check payout precedence for parcels as well
                for j in range(1, problem.M+1):
                    pick = problem.parc_pick(j)
                    drop = problem.parc_drop(j)
                    if pick in pos_map or drop in pos_map:
                        if (pick in pos_map) != (drop in pos_map):
                            valid = False
                            break
                        if pos_map[pick] >= pos_map[drop]:
                            valid = False
                            break
                if not valid:
                    break
                # if passed checks, compute route length
                L = route_length_from_sequence(seq, problem.D)
                route_lists.append([0] + seq + [0])
                route_lengths.append(L)
            if not valid:
                continue
            # All taxis processed and valid -> feasible solution
            sol = Solution(route_lists, route_lengths)
            solutions.append(sol)
            # keep sorted by objective
            solutions.sort(key=lambda s: s.max_length)
            # cap
            if len(solutions) >= max_solutions:
                break
        if pruned_due_time or len(solutions) >= max_solutions:
            break

    elapsed = time.time() - start
    info = {
        "assignments_enumerated": enumerated_assignments,
        "permutations_enumerated": enumerated_permutations,
        "time": elapsed,
        "found": len(solutions),
        "status": "timeout" if pruned_due_time else "done"
    }
    return solutions, info


# ---------------------- Branch-and-bound (DFS on pair->taxi with LB) -------------
def _nn_lower_bound_for_nodes(nodes:List[int], D:List[List[int]]) -> int:
    """
    Simple nearest-neighbor heuristic to estimate a short route length starting and ending at depot (0).
    This is used as a lower-bound heuristic for pruning.
    """
    if not nodes:
        return 0
    unvisited = set(nodes)
    pos = 0
    total = 0
    while unvisited:
        # find nearest
        nxt = min(unvisited, key=lambda x: D[pos][x])
        total += D[pos][nxt]
        pos = nxt
        unvisited.remove(nxt)
    total += D[pos][0]
    return total

def branch_and_bound(problem: ShareARideProblem,
                     time_limit:float=30.0,
                     verbose:bool=False,
                     target_cost:Optional[int]=None) -> Tuple[Solution, Dict[str,Any]]:
    """
    Branch and Bound implemented as DFS on assignments of (N+M) pairs to K taxis.
    For partial assignments, compute a simple LB for each taxi (using NN heuristic) and prune if LB >= current best.
    When complete assignment found, evaluate all permutations per taxi (like brute force) but limited by time.
    """
    start = time.time()
    N, M, K = problem.N, problem.M, problem.K
    pairs = []
    for i in range(1, N+1): pairs.append(("P", i))
    for j in range(1, M+1): pairs.append(("L", j))
    num_pairs = len(pairs)
    best_solution = None
    best_cost = float("inf") if target_cost is None else target_cost
    stats = {"nodes":0, "pruned":0, "evaluated_assignments":0}

    # recursive DFS: assignment[index] = taxi id
    assignment = [-1]*num_pairs

    def dfs_pair(index:int):
        nonlocal best_solution, best_cost, stats
        # time check
        if time.time() - start > time_limit:
            return "timeout"
        if index >= num_pairs:
            # full assignment -> evaluate permutations per taxi (like brute force)
            stats["evaluated_assignments"] += 1
            taxi_items = [[] for _ in range(K)]
            for pair_idx, taxi_id in enumerate(assignment):
                taxi_items[taxi_id].append(pairs[pair_idx])
            # build node lists
            taxi_nodes = []
            for t_idx in range(K):
                nodes = []
                for it in taxi_items[t_idx]:
                    if it[0] == "P":
                        i = it[1]
                        nodes.append(problem.ppick(i)); nodes.append(problem.pdrop(i))
                    else:
                        j = it[1]
                        nodes.append(problem.parc_pick(j)); nodes.append(problem.parc_drop(j))
                taxi_nodes.append(nodes)
            # compute permutations per taxi and check feasiblity, update best_solution
            perms_per_taxi = [list(itertools.permutations(nodes)) if nodes else [()] for nodes in taxi_nodes]
            # if total product too big, skip (prune)
            total_perm = 1
            for p in perms_per_taxi:
                total_perm *= max(1, len(p))
                if total_perm > 1_000_000:  # avoid blowup
                    return None
            for combo in itertools.product(*perms_per_taxi):
                if time.time() - start > time_limit:
                    return "timeout"
                # check feasibility similarly to brute-force
                valid = True
                route_lists = []
                route_lengths = []
                for t_idx, perm_nodes in enumerate(combo):
                    seq = list(perm_nodes)
                    pos_map = {node: idx for idx, node in enumerate(seq)}
                    # precedence passenger
                    for i in range(1, N+1):
                        pick = problem.ppick(i); drop = problem.pdrop(i)
                        if pick in pos_map or drop in pos_map:
                            if (pick in pos_map) != (drop in pos_map): valid=False; break
                            if pos_map[pick] >= pos_map[drop]: valid=False; break
                    if not valid: break
                    # parcels and load
                    load = 0
                    for node in seq:
                        if problem.N+1 <= node <= problem.N+problem.M:  # parcel pickup
                            j = node - problem.N
                            load += problem.q[j-1]
                            if load > problem.Q[min(t_idx, len(problem.Q)-1)]: valid=False; break
                        elif 2*problem.N + problem.M + 1 <= node <= 2*problem.N + 2*problem.M:
                            j = node - (2*problem.N + problem.M)
                            load -= problem.q[j-1]
                            if load < 0: valid=False; break
                    if not valid: break
                    # route length
                    L = route_length_from_sequence(seq, problem.D)
                    route_lists.append([0] + seq + [0])
                    route_lengths.append(L)
                if not valid: continue
                sol = Solution(route_lists, route_lengths)
                if sol.max_length < best_cost:
                    best_cost = sol.max_length
                    best_solution = sol
                    if verbose:
                        print("New best cost:", best_cost, "time:", time.time()-start)
                    # if best_cost reaches 0 or very low, can early stop
                    if best_cost <= 0:
                        return "found"
            return None

        # else branch on assignment[index] ∈ [0..K-1]
        stats["nodes"] += 1
        # try assigning current pair to each taxi
        for k in range(K):
            assignment[index] = k
            # compute simple LB for partial assignment: for each taxi, collect nodes assigned so far and estimate minimal route length
            taxi_nodes_partial = [[] for _ in range(K)]
            for p_idx in range(index+1):  # only assigned ones so far
                taxi_id = assignment[p_idx]
                it = pairs[p_idx]
                if it[0] == "P":
                    taxi_nodes_partial[taxi_id].append(problem.ppick(it[1])); taxi_nodes_partial[taxi_id].append(problem.pdrop(it[1]))
                else:
                    taxi_nodes_partial[taxi_id].append(problem.parc_pick(it[1])); taxi_nodes_partial[taxi_id].append(problem.parc_drop(it[1]))
            # compute LB = max over taxis of NN heuristic on assigned nodes
            lb = 0
            for t_idx in range(K):
                lb_t = _nn_lower_bound_for_nodes(taxi_nodes_partial[t_idx], problem.D)
                if lb_t > lb:
                    lb = lb_t
            # if lb already >= best_cost, prune
            if best_solution is not None and lb >= best_cost:
                stats["pruned"] += 1
                continue
            # else recurse
            res = dfs_pair(index+1)
            if res == "timeout" or res == "found":
                return res
        assignment[index] = -1
        return None

    res = dfs_pair(0)
    elapsed = time.time() - start
    info = {"time": elapsed, "nodes": stats["nodes"], "pruned": stats["pruned"], "evaluated_assignments": stats["evaluated_assignments"], "status": res or "done"}
    # if no solution found, fallback to greedy
    if best_solution is None:
        greedy = greedy_balanced_solver(problem)
        return greedy, info
    return best_solution, info


# ---------------------- Test instance generators --------------------------------
def random_distance_matrix(n:int, low:int=3, high:int=9, asymmetric:bool=False, seed:Optional[int]=None):
    rng = random.Random(seed)
    D = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i][j] = 0
            else:
                if asymmetric:
                    D[i][j] = rng.randint(low, high)
                else:
                    if j < i:
                        D[i][j] = D[j][i]
                    else:
                        D[i][j] = rng.randint(low, high)
    return D

def euclidean_distance_matrix(coords:List[Tuple[float,float]]) -> List[List[int]]:
    n = len(coords)
    D = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i][j] = 0
            else:
                D[i][j] = int(round(math.hypot(coords[i][0]-coords[j][0], coords[i][1]-coords[j][1])))
    return D

def generate_instance_lazy(N:int, M:int, K:int,
                           low:int=3, high:int=9,
                           qlow:int=1, qhigh:int=4,
                           Qlow:int=6, Qhigh:int=12,
                           use_poisson:bool=False, poisson_lambda:float=2.0,
                           seed:Optional[int]=None) -> ShareARideProblem:
    rng = random.Random(seed)
    if use_poisson:
        # approximate Poisson by sampling; Python's random doesn't have poisson; use simple method:
        def sample_poisson(lmbd):
            # Knuth's algorithm
            L = math.exp(-lmbd)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= rng.random()
            return max(1, k-1)
        q = [sample_poisson(poisson_lambda) for _ in range(M)]
    else:
        q = [rng.randint(qlow, qhigh) for _ in range(M)]
    Q = [rng.randint(Qlow, Qhigh) for _ in range(K)]
    n_nodes = 2*N + 2*M + 1
    D = random_distance_matrix(n_nodes, low=low, high=high, asymmetric=False, seed=seed)
    return ShareARideProblem(N, M, K, q, Q, D)

def generate_instance_coords(N:int, M:int, K:int, area:float=20.0,
                             qlow:int=1, qhigh:int=4, Qlow:int=6, Qhigh:int=15,
                             seed:Optional[int]=None, visualize:bool=False):
    rng = random.Random(seed)
    total_points = 1 + 2*N + 2*M
    coords = []
    coords.append((area/2.0, area/2.0))  # depot center
    for _ in range(total_points-1):
        coords.append((rng.random()*area, rng.random()*area))
    D = euclidean_distance_matrix(coords)
    q = [rng.randint(qlow, qhigh) for _ in range(M)]
    Q = [rng.randint(Qlow, Qhigh) for _ in range(K)]
    prob = ShareARideProblem(N, M, K, q, Q, D)
    if visualize:
        try:
            import matplotlib.pyplot as plt
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            plt.figure(figsize=(6,6))
            plt.scatter(xs[1:], ys[1:], label="Nodes")
            plt.scatter(xs[0], ys[0], c='red', label="Depot")
            for idx,(x,y) in enumerate(coords):
                plt.text(x, y, str(idx), fontsize=8)
            plt.title(f"Instance coords N={N}, M={M}, K={K}")
            plt.legend()
            plt.show()
        except Exception as e:
            print("Visualization failed (matplotlib not available):", e)
    return prob

# ---------------------- Demo / main --------------------------------------------
def demo_generate_and_solve():
    random.seed(12345)
    typeI = []
    typeII = []
    params_I = [(2,2,2),(2,3,2),(3,2,2),(3,3,2),(2,2,3)]
    for idx,(N,M,K) in enumerate(params_I):
        prob = generate_instance_coords(N,M,K, area=15, seed=100+idx)
        typeI.append(prob)
    params_II = [(4,4,3),(4,5,3),(5,4,3),(5,5,3),(6,4,3)]
    for idx,(N,M,K) in enumerate(params_II):
        prob = generate_instance_coords(N,M,K, area=40, seed=200+idx)
        typeII.append(prob)

    print("Solving Type I tests with brute-force enumerator (exhaustive pair assignment + perms):\n")
    typeI_results = []
    for i,prob in enumerate(typeI):
        print(f"=== Type I test {i+1}: N={prob.N}, M={prob.M}, K={prob.K} ===")
        prob.pretty_print(verbose=0)
        sols, info = brute_force_enumerate(prob, max_solutions=2000, verbose=False, time_limit=15.0)
        print("Enumeration info:", info)
        if sols:
            best = sols[0]
            best.pretty_print()
        else:
            print("No solution enumerated within limits/time.")
        print("\n")
        typeI_results.append((sols, info))

    print("Type II tests: show greedy baseline (no exhaustive solve)\n")
    typeII_results = []
    for i,prob in enumerate(typeII):
        print(f"=== Type II test {i+1}: N={prob.N}, M={prob.M}, K={prob.K} ===")
        prob.pretty_print(verbose=0)
        greedy = greedy_balanced_solver(prob)
        greedy.pretty_print()
        typeII_results.append(greedy)
        print("\n")
    return typeI_results, typeII_results

if __name__ == "__main__":
    demo_generate_and_solve()