import time
from itertools import permutations
from typing import List, Tuple, Any, Dict, Iterator, Optional, Callable

from share_a_ride.problem import ShareARideProblem
from share_a_ride.solution import Solution
from share_a_ride.utils.helper import route_cost_from_sequence



def _assign_pairs_canonical(num_pairs: int, K: int) -> Iterator[List[int]]:
    """
    Assign pickup/delivery pairs into routes canonically with symmetry breaking.
    (First used taxi gets id=0, next new taxi id=1, etc.)
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



def _assign_routes_sequential(
        problem: ShareARideProblem, 
        seqs_all: List[List[List[int]]],
        tle: Optional[Callable[[], bool]] = None,
    ) -> Iterator[Solution]:
    """
    Combine taxi routes into full solutions using DFS.
    Each taxi must have at least one route.
    """

    def dfs_assign_routes(
            t_idx: int,                         # current taxi index
            cur_routes: List[List[int]],        # current routes considered from t_idx
            cur_costs: List[float]              # current costs considered from t_idx
        ):

        if tle and tle():
            return

        # Base case: all taxis assigned
        if t_idx == len(seqs_all):
            # build solution, check validity and yield
            sol = Solution(problem, cur_routes, cur_costs)
            if sol.is_valid():
                yield sol
            return

        # Develop case: incrementally build routes
        for seq in seqs_all[t_idx]:
            if tle and tle():
                return
            # recurse to next taxi
            yield from dfs_assign_routes(
                t_idx + 1,
                cur_routes + [seq],
                cur_costs + [route_cost_from_sequence(seq, problem.D)]
            )

    yield from dfs_assign_routes(0, [], [])



def _gen_routes_for_taxi(
        prob: ShareARideProblem,            # problem instance
        taxi_pairs: List[Tuple[str, int]],  # list of ("P", pass_id) or ("L", parc_id)
        taxi_idx: int,                      # taxi index
        start_time: float,                  # start time for break detection
        time_limit: float                   # time limit in seconds
    ) -> Iterator[List[int]]:
    """
    Generate all feasible routes for a given taxi, given the pickup/delivery pairs assignment. 
    This is done by:
      1) Enumerating all Regular Bracket Sequences (RBS) of length 2n, 
      representing pickup/drop order
      2) Assigning each bracket-pair to a pickup/drop pair via permutations
      3) Validating capacity and passenger constraints and yield only feasible routes
    
    Assumptions/constraints need to be met:
      - pickup before drop
      - at most 1 passenger onboard
      - parcel capacity respected
    
    Yields:
      routes as list of node indices, starting/ending at depot 0
    """

    # ========== Helper functions ==========

    def tle() -> bool:
        return time_limit is not None and (time.time() - start_time) > time_limit

    def _next_balanced_sequence(s: List[str], n: int) -> Optional[List[str]]:
        """
        Get the next balanced parenthesis sequence in lexicographic order.
        s is a list of '(' and ')', length n=2*r, currently balanced.
        Returns a new list or None if s is the last sequence.
        """
        s = s[:]
        depth = 0
        for i in range(n - 1, -1, -1):
            if s[i] == '(':
                depth -= 1
            else:
                depth += 1

            if s[i] == '(' and depth > 0:
                rem = n - i - 1
                depth -= 1
                open_cnt = (rem - depth) // 2
                close_cnt = rem - open_cnt

                return s[:i] + [')'] + ['('] * open_cnt + [')'] * close_cnt

        return None


    def _enumerate_all_rbs(r: int) -> Iterator[List[str]]:
        """
        Lexicographic enumeration of balanced sequences of length 2*r.
        r here should be the number of pairs (or requests in this context) assigned to the taxi.
        """
        # Edge case: r = 0
        if r == 0:
            yield []
            return

        # Start from the smallest sequence: r '(' followed by r ')'
        s = ['('] * r + [')'] * r
        n = 2 * r

        # Main loop
        while True:
            if tle():
                return

            yield s
            nxt = _next_balanced_sequence(s, n)
            if nxt is None:
                break

            s = nxt


    def _assign_request_from_rbs(
            rbs: List[str],
            pickup_order: List[int],
        ) -> Iterator[List[int]]:
        """
        Given an RBS (sequence of '(' for pickups and ')' for drops) and a pickup order,
        enumerate all feasible drop choices that respect:
          - pickup before drop (precedence)
          - at most 1 passenger onboard
          - parcel capacity
        Returns all feasible routes found as a list of node-index lists. Returns [] if none.
        """
        n = len(rbs)
        results: List[List[int]] = []

        route: List[int] = [0]          # start at depot
        open_reqs: List[int] = []       # indices of picked-up requests not yet dropped

        # ========== DFS helper function ==========
        def dfs(
                pos: int,               # current position in rbs
                next_pick_idx: int,     # next index in pickup_order to pick
                passenger_on: bool,     # whether a passenger is currently onboard
                load: int               # current parcel load
            ) -> Iterator[List[int]]:
            """
            Helper DFS to assign requests to the RBS positions.
            """
            if tle():
                return

            # End: all symbols processed
            if pos == n:
                yield route + [0]  # return to depot
                return

            # Else: process current symbol
            ch = rbs[pos]
            if ch == '(':
                # Take next pickup from pickup_order
                req_idx = pickup_order[next_pick_idx]
                pu, dr, qty = reqs[req_idx]

                if taxi_pairs[req_idx][0] == "P":   # passenger pickup
                    if passenger_on:
                        return
                    passenger_on2 = True
                    load2 = load
                else:                               # parcel pickup
                    passenger_on2 = passenger_on
                    load2 = load + qty
                    if load2 > cap:
                        return

                # apply
                open_reqs.append(req_idx)
                route.append(pu)

                # recurse
                yield from dfs(pos + 1, next_pick_idx + 1, passenger_on2, load2)

                # backtrack
                route.pop()
                open_reqs.pop()

            else:
                # Drop: try all currently open requests (not only LIFO)
                if not open_reqs:
                    return

                # Try each candidate to drop at this ')'
                for i, req_idx in enumerate(open_reqs):
                    pu, dr, qty = reqs[req_idx]

                    if taxi_pairs[req_idx][0] == "P":
                        if not passenger_on:
                            continue
                        passenger_on2 = False
                        load2 = load
                    else:
                        passenger_on2 = passenger_on
                        load2 = load - qty
                        if load2 < 0:
                            continue

                    # apply: remove this open request and append its drop node
                    removed = open_reqs.pop(i)
                    route.append(dr)

                    yield from dfs(pos + 1, next_pick_idx, passenger_on2, load2)

                    # backtrack
                    route.pop()
                    open_reqs.insert(i, removed)


        # ========== Main DFS call ==========
        yield from dfs(0, 0, False, 0)



    # ========== Main function body ==========
    n_requests = len(taxi_pairs)
    if n_requests == 0:
        yield []
        return

    # Precompute node indices and parcel quantities for each request
    # reqs[k] corresponds to taxi_pairs[k]
    reqs: List[Tuple[int, int, int]] = []  # (pickup_node, drop_node, parcel_qty)
    for type, idx in taxi_pairs:
        if type == "P":     # "P" passenger
            pu = prob.ppick(idx)
            dr = prob.pdrop(idx)
            qty = 0
        else:               # "L" parcel
            pu = prob.parc_pick(idx)
            dr = prob.parc_drop(idx)
            qty = prob.q[idx - 1]
        reqs.append((pu, dr, qty))

    cap = prob.Q[taxi_idx]

    # Enumerate all RBS and assign requests
    for rbs in _enumerate_all_rbs(n_requests):
        if tle():
            return
        
        for pickup_order in permutations(range(n_requests)):
            if tle():
                return

            # Enumerate all feasible routes compatible with this RBS and pickup order
            for route in _assign_request_from_rbs(rbs, pickup_order):
                yield route



def exhaustive_enumerator(
        problem: ShareARideProblem,
        max_solutions: int = 100000,
        time_limit: float = 30.0,
        verbose: bool = False
    ) -> Tuple[List[Solution], Dict[str, Any]]:
    """
    Exhaustive enumeration of all feasible solutions.
    - Use DFS to assign pickup/delivery pairs to taxis. Use canonical order as
    to break symmetry.
    - For each taxi, generate all feasible routes using gen_routes_for_taxi.
    - Combine taxi routes into full solutions using dfs again
    - Check validity and sort by max cost.

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

    def tle() -> bool:
        return time_limit is not None and (time.time() - start) > time_limit

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

    # main loop
    assignments = _assign_pairs_canonical(num_pairs, K)

    # Check if assignment
    for assignment in assignments:
        if tle():
            timeout = True
            break

        assignments_done += 1

        # Build pairs by taxi from assignment
        taxi_pairs = [[] for _ in range(K)]
        for idx, t_id in enumerate(assignment):
            taxi_pairs[t_id].append(pairs[idx])

        # Retrieve all feasible sequences for each taxi with assigned pairs
        used_taxis = max(assignment) + 1 if assignment else 0   # number of taxis used
        seqs_all: List[List[List[int]]] = []    # List of list of routes for each taxi
        skip = False   
        for t in range(used_taxis):
            if tle():
                timeout = True
                break
            gen = _gen_routes_for_taxi(problem, taxi_pairs[t], t, start, time_limit)
            seqs = list(gen)

            if tle():
                timeout = True
                break

            if not seqs:
                skip = True
                break

            seqs_all.append(seqs)
            seq_generated += len(seqs)
        if timeout:
            break
        if skip:
            continue

        # pad remaining taxis with a single dummy route
        for _ in range(K - used_taxis):
            seqs_all.append([[0, 0]])

        # combine taxi routes into full solutions and append valid ones
        for sol in _assign_routes_sequential(problem, seqs_all, tle=tle):
            if tle():
                timeout = True
                break
            solutions.append(sol)
            if len(solutions) >= max_solutions:
                timeout = True
                break
        if timeout:
            break

        # check time limit and solution count
        if timeout or len(solutions) >= max_solutions:
            break

    # sort solutions by max cost
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



def exhaustive_solver(
        problem: ShareARideProblem,
        max_solutions: int = 100000,
        time_limit: float = 30.0,
        verbose: bool = False
    ) -> Tuple[Solution, Dict[str, Any]]:
    """
    Exhaustive search solver that returns only the best solution found.
    Uses exhaustive_enumerate internally.
    """
    solutions, info = exhaustive_enumerator(
        problem,
        max_solutions=max_solutions,
        time_limit=time_limit,
        verbose=verbose
    )

    best_solution = solutions[0] if solutions else None

    return best_solution, info

