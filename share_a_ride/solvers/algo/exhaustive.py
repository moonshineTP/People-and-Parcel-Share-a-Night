import time
from typing import List, Tuple, Any, Dict, Iterator

from share_a_ride.problem import ShareARideProblem
from share_a_ride.solution import Solution
from share_a_ride.utils.helper import route_cost_from_sequence
from share_a_ride.solvers.utils.helper import gen_routes_for_taxi


def _assign_pairs_canonical(num_pairs: int, K: int) -> Iterator[List[int]]:
    """
    Yield assignments with symmetry breaking.
    First used taxi gets id=0, next new taxi id=1, etc.
    """
    def dfs_assign_pairs(idx: int, used: int, cur: List[int]):
        # Base case
        if idx == num_pairs:
            yield tuple(cur)
            return

        # assign to existing taxis
        for t in range(used):
            cur.append(t)
            yield from dfs_assign_pairs(idx + 1, used, cur)
            cur.pop()

        # open a new taxi if available
        if used < K:
            cur.append(used)
            yield from dfs_assign_pairs(idx + 1, used + 1, cur)
            cur.pop()

    yield from dfs_assign_pairs(0, 0, [])


def exhaustive_enumerate(
        problem: ShareARideProblem,
        max_solutions: int = 100000,
        time_limit: float = 30.0,
        verbose: bool = False
    ) -> Tuple[List[Solution], Dict[str, Any]]:
    """
    Exhaustive solver:
        - Assign pairs to taxis canonically (symmetry breaking).
        - For each taxi, generate valid sequences directly (pickup before drop).
        - Combine taxi sequences into full solutions without reintroducing symmetry.
    Returns:
        (solutions, info): tuple where
        - solutions: list of valid Solution objects found (up to max_solutions)
        - info: dictionary with statistics:
            + assignments: number of pair-to-taxi assignments considered
            + num_sequences: total number of taxi routes generated
            + num_solutions: number of valid solutions found
            + time: total time taken
            + status: "done" or "timeout"
    """
    start = time.time()

    # problem parameters
    N, M, K = problem.N, problem.M, problem.K
    num_pairs = N + M
    pairs = [("P", i) for i in range(1, N + 1)]
    pairs += [("L", j) for j in range(1, M + 1)]

    # results
    solutions: List[Solution] = []
    assignments_done = 0
    seq_generated = 0
    timeout = False


    # main dfs function
    def dfs_assign_routes(
            t_idx: int,                         # current taxi index
            seqs_all: List[List[List[int]]],    # all sequences for each taxi
            cur_routes: List[List[int]],        # current routes considered from t_idx
            cur_costs: List[float]              # current cost considered from t_idx
        ):      

        nonlocal assignments_done, seq_generated, timeout

        if time.time() - start > time_limit or timeout:
            timeout = True
            return

        # base case: all taxis assigned
        if t_idx == len(seqs_all):
            # Fill up remaining taxis with dummy routes
            cur_routes = cur_routes + [[0, 0]] * (K - len(seqs_all))
            cur_costs = cur_costs + [0] * (K - len(seqs_all))

            # build solution and check validity
            sol = Solution(problem, cur_routes, cur_costs)

            if sol.is_valid():
                solutions.append(sol)
                if len(solutions) >= max_solutions:
                    timeout = True

            return

        # Incrementally build routes
        for seq in seqs_all[t_idx]:
            # check time limit
            if time.time() - start > time_limit or timeout:
                timeout = True
                return

            # recurse to next taxi
            dfs_assign_routes(
                t_idx + 1,
                seqs_all,
                cur_routes + [seq],
                cur_costs + [route_cost_from_sequence(seq, problem.D)]
            )

            # check time limit
            if timeout:
                return



    # main loop
    assignments = _assign_pairs_canonical(num_pairs, K)

    # Check if assignment
    for assignment in assignments:
        # check time limit
        if time.time() - start > time_limit:
            timeout = True
            break


        assignments_done += 1
        # build pairs by taxi from assignment
        taxi_pairs = [[] for _ in range(K)]
        for idx, t_id in enumerate(assignment):
            taxi_pairs[t_id].append(pairs[idx])
        
        # parameters
        used_taxis = max(assignment) + 1 if assignment else 0
        seqs_all: List[List[List[int]]] = []
        skip = False

        # generate all feasible sequences for each taxi
        for t in range(used_taxis):
            # generate
            gen = gen_routes_for_taxi(problem, taxi_pairs[t], t, start, time_limit)
            seqs = list(gen)

            # check empty sequences
            if not seqs:
                skip = True
                break

            # Add sequences
            seqs_all.append(seqs)
            seq_generated += len(seqs)

        # skip if any taxi has no feasible routes
        if skip:
            continue

        # combine taxi routes into full solutions and append valid ones
        dfs_assign_routes(0, seqs_all, [], [])

        # check time limit and solution count
        if timeout or len(solutions) >= max_solutions:
            break

    solutions.sort(key=lambda s: s.max_cost)

    # final stats
    elapsed = time.time() - start
    info = {
        "assignments": assignments_done,
        "num_sequences": seq_generated,
        "num_solutions": len(solutions),
        "time": elapsed,
        "status": "timeout" if timeout else "done"
    }

    # return results
    return solutions, info