from datetime import time
from typing import List, Optional, Tuple, Dict, Any

from share_a_ride.problem import ShareARideProblem
from share_a_ride.solution import Solution
from share_a_ride.solvers.greedy import greedy_balanced_solver
from utils.helper import route_length_from_sequence

# ---------------------- Branch-and-bound (DFS on pair->taxi with LB) -------------
def _nn_lower_bound(nodes: List[int], D: List[List[int]]) -> int:
    """Nearest-neighbor heuristic: short route covering nodes + depot."""
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


def _gen_routes_for_taxi(
    prob: ShareARideProblem,
    taxi_nodes: List[Tuple[str, int]],
    taxi_idx: int,
    start_time: float,
    time_limit: float,
):
    """
    DFS route generation for a single taxi.
    Similar to CBUS: ensure pickup before drop,
    passenger ≤1, parcels ≤ Q[taxi_idx].
    """
    types = [t for t, _ in taxi_nodes]
    ids = [x for _, x in taxi_nodes]
    pick_nodes = [
        prob.ppick(i) if t == "P" else prob.parc_pick(i)
        for t, i in taxi_nodes
    ]
    drop_nodes = [
        prob.pdrop(i) if t == "P" else prob.parc_drop(i)
        for t, i in taxi_nodes
    ]

    n = len(taxi_nodes)
    picked, dropped = [False] * n, [False] * n
    seq: List[int] = []

    def recompute_state():
        load, passenger = 0, 0
        for node in seq:
            for i in range(n):
                if node == pick_nodes[i]:
                    if types[i] == "P":
                        passenger += 1
                    else:
                        load += prob.q[ids[i] - 1]
                elif node == drop_nodes[i]:
                    if types[i] == "P":
                        passenger = max(0, passenger - 1)
                    else:
                        load -= prob.q[ids[i] - 1]
        return load, passenger

    def dfs():
        if time.time() - start_time > time_limit:
            return
        if all(dropped):
            yield list(seq)
            return

        # try pickup
        for i in range(n):
            if picked[i]:
                continue
            load, passenger = recompute_state()
            if types[i] == "P" and passenger >= 1:
                continue
            if types[i] == "L":
                qj = prob.q[ids[i] - 1]
                if load + qj > prob.Q[taxi_idx]:
                    continue
            picked[i] = True
            seq.append(pick_nodes[i])
            yield from dfs()
            seq.pop()
            picked[i] = False

        # try drop
        for i in range(n):
            if not picked[i] or dropped[i]:
                continue
            seq.append(drop_nodes[i])
            load, passenger = recompute_state()
            if load <= prob.Q[taxi_idx] and passenger <= 1:
                dropped[i] = True
                yield from dfs()
                dropped[i] = False
            seq.pop()

    yield from dfs()


def branch_and_bound(
    problem: ShareARideProblem,
    time_limit: float = 30.0,
    verbose: bool = False,
    target_cost: Optional[int] = None
) -> Tuple[Solution, Dict[str, Any]]:
    """
    Branch-and-bound with:
      - symmetry breaking in pair assignment,
      - two-phase DFS: assign pairs then build routes,
      - LB pruning at both phases.
    """
    start = time.time()
    N, M, K = problem.N, problem.M, problem.K
    pairs = [("P", i) for i in range(1, N + 1)]
    pairs += [("L", j) for j in range(1, M + 1)]
    num_pairs = len(pairs)

    best_sol, best_cost = None, (
        float("inf") if target_cost is None else target_cost
    )
    stats = {"nodes": 0, "pruned": 0, "routes": 0}

    assignment = [-1] * num_pairs

    def dfs_assign(idx: int):
        nonlocal best_sol, best_cost

        if time.time() - start > time_limit:
            return "timeout"
        if idx >= num_pairs:
            return dfs_routes()
        stats["nodes"] += 1

        used_taxis = max(assignment[:idx]) + 1 if idx > 0 else 0
        for t in range(used_taxis + 1 if used_taxis < K else used_taxis):
            assignment[idx] = t

            # lower bound: estimate cost so far
            taxi_nodes = [[] for _ in range(K)]
            for j in range(idx + 1):
                kind, pid = pairs[j]
                if kind == "P":
                    taxi_nodes[assignment[j]].extend(
                        [problem.ppick(pid), problem.pdrop(pid)]
                    )
                else:
                    taxi_nodes[assignment[j]].extend(
                        [problem.parc_pick(pid), problem.parc_drop(pid)]
                    )
            lb = max(
                _nn_lower_bound(nodes, problem.D) for nodes in taxi_nodes
            )
            if lb >= best_cost:
                stats["pruned"] += 1
                continue

            res = dfs_assign(idx + 1)
            if res == "timeout":
                return "timeout"
        assignment[idx] = -1
        return None

    def dfs_routes():
        nonlocal best_sol, best_cost
        taxi_items = [[] for _ in range(K)]
        for i, t in enumerate(assignment):
            taxi_items[t].append(pairs[i])

        route_lists, lengths = [], []
        for t in range(K):
            items = taxi_items[t]
            seqs = _gen_routes_for_taxi(
                problem, items, t, start, time_limit
            )
            best_local = None
            for seq in seqs:
                stats["routes"] += 1
                L = route_length_from_sequence(seq, problem.D)
                if best_local is None or L < best_local[0]:
                    best_local = (L, [0] + seq + [0])
            if best_local is None:
                return None
            lengths.append(best_local[0])
            route_lists.append(best_local[1])

        sol = Solution(problem, route_lists, lengths)
        if sol.max_length < best_cost:
            best_cost = sol.max_length
            best_sol = sol
            if verbose:
                print("New best:", best_cost, "time", time.time() - start)
        return None

    res = dfs_assign(0)
    elapsed = time.time() - start
    info = {
        "time": elapsed,
        "nodes": stats["nodes"],
        "pruned": stats["pruned"],
        "routes": stats["routes"],
        "status": res or "done",
    }
    if best_sol is None:
        return greedy_balanced_solver(problem), info
    return best_sol, info