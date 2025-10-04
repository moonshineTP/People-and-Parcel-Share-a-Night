import itertools
from datetime import time
from typing import List, Tuple, Any, Dict

from share_a_ride.problem import ShareARideProblem
from share_a_ride.solution import Solution
from utils.helper import route_length_from_sequence


def _assign_pairs_canonical(num_pairs: int, K: int):
    """
    Yield assignments with symmetry breaking.
    First used taxi gets id=0, next new taxi id=1, etc.
    """
    def dfs(idx: int, used: int, cur: List[int]):
        if idx == num_pairs:
            yield tuple(cur)
            return

        # assign to existing taxis
        for t in range(used):
            cur.append(t)
            yield from dfs(idx + 1, used, cur)
            cur.pop()

        # open a new taxi if available
        if used < K:
            cur.append(used)
            yield from dfs(idx + 1, used + 1, cur)
            cur.pop()

    yield from dfs(0, 0, [])


def _gen_sequences_for_taxi(
    prob: ShareARideProblem,
    taxi_pairs: List[Tuple[str, int]],
    taxi_idx: int,
    start_time: float,
    time_limit: float
):
    """
    DFS generator yielding feasible sequences for one taxi.
    Ensures:
      - pickup before drop
      - capacity respected
      - at most 1 passenger onboard
    """
    k_pairs = len(taxi_pairs)
    if k_pairs == 0:
        yield []
        return

    types = [p[0] for p in taxi_pairs]
    ids = [p[1] for p in taxi_pairs]
    pick_nodes = [prob.ppick(x) if t == "P" else prob.parc_pick(x)
                  for t, x in taxi_pairs]
    drop_nodes = [prob.pdrop(x) if t == "P" else prob.parc_drop(x)
                  for t, x in taxi_pairs]

    picked = [False] * k_pairs
    dropped = [False] * k_pairs
    seq: List[int] = []

    def recompute_load_passenger():
        """Recompute parcel load and passenger count from current seq."""
        load, passenger = 0, 0
        for node in seq:
            for i in range(k_pairs):
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

        # try pickups
        for i in range(k_pairs):
            if picked[i]:
                continue
            # passenger check
            if types[i] == "P" and any(
                node in seq for node in pick_nodes if types[i] == "P"
            ):
                continue
            # parcel check
            if types[i] == "L":
                new_load = prob.q[ids[i] - 1]
                if new_load > prob.Q[taxi_idx]:
                    continue

            picked[i] = True
            seq.append(pick_nodes[i])
            yield from dfs()
            seq.pop()
            picked[i] = False

        # try drops
        for i in range(k_pairs):
            if not picked[i] or dropped[i]:
                continue
            seq.append(drop_nodes[i])
            load, passenger = recompute_load_passenger()
            if load <= prob.Q[taxi_idx] and 0 <= passenger <= 1:
                dropped[i] = True
                yield from dfs()
                dropped[i] = False
            seq.pop()

    yield from dfs()

def exhaustive_enumerate(
    problem: ShareARideProblem,
    max_solutions: int = 100000,
    time_limit: float = 30.0
) -> Tuple[List[Solution], Dict[str, Any]]:
    """
    Exhaustive solver:
      - Assign pairs to taxis canonically (symmetry breaking).
      - For each taxi, generate valid sequences directly (pickup before drop).
      - Combine taxi sequences into full solutions.
    """
    start = time.time()
    N, M, K = problem.N, problem.M, problem.K
    num_pairs = N + M
    pairs = [("P", i) for i in range(1, N + 1)]
    pairs += [("L", j) for j in range(1, M + 1)]

    solutions: List[Solution] = []
    assignments_done = 0
    seq_generated, combos = 0, 0
    timeout = False

    for assignment in _assign_pairs_canonical(num_pairs, K):
        if time.time() - start > time_limit:
            timeout = True
            break
        assignments_done += 1

        # map assignment to taxi->pairs
        taxi_pairs = [[] for _ in range(K)]
        for idx, t_id in enumerate(assignment):
            taxi_pairs[t_id].append(pairs[idx])

        used_taxis = max(assignment) + 1 if assignment else 0
        seqs_all: List[List[List[int]]] = []
        skip = False

        for t in range(used_taxis):
            gen = _gen_sequences_for_taxi(
                problem, taxi_pairs[t], t,
                start, time_limit, max_solutions
            )
            seqs = list(gen)
            if not seqs:
                skip = True
                break
            seqs_all.append(seqs)
            seq_generated += len(seqs)

        if skip:
            continue

        for combo in itertools.product(*seqs_all) if seqs_all else [()]:
            if time.time() - start > time_limit:
                timeout = True
                break
            combos += 1

            routes, lengths = [], []
            for t in range(K):
                if t < used_taxis and combo[t]:
                    seq = combo[t]
                    L = route_length_from_sequence(seq, problem.D)
                    routes.append([0] + seq + [0])
                    lengths.append(L)
                else:
                    routes.append([0, 0])
                    lengths.append(0)

            sol = Solution(problem, routes, lengths)
            if not sol.is_valid():
                continue

            solutions.append(sol)
            solutions.sort(key=lambda s: (s.max_length, sum(s.route_lengths)))
            if len(solutions) >= max_solutions:
                break

        if timeout or len(solutions) >= max_solutions:
            break

    elapsed = time.time() - start
    info = {
        "assignments": assignments_done,
        "sequences": seq_generated,
        "combos": combos,
        "time": elapsed,
        "found": len(solutions),
        "status": "timeout" if timeout else "done"
    }
    return solutions, info
