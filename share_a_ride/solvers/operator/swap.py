"""
Module for the Swap operator in ride-sharing optimization.
Swaps two requests between the same vehicle route.
"""
import random
import heapq
from typing import List, Optional, Tuple


from share_a_ride.core.solution import PartialSolution
from share_a_ride.solvers.operator.utils import TreeSegment



def cost_decrement_intra_swap(
        partial: PartialSolution,
        route_idx: int,
        a_idx: int,
        b_idx: int
    ) -> int:
    """
    Compute the change in cost of swapping two requests a_idx and b_idx in the
    same vehicle route identified by route_idx as Cost_before - Cost_After

    Parameters:
    - partial: PartialSolution object representing the current solution.
    - route_idx: Index of the vehicle route in which to perform the swap.
    - a_idx: Index of the first request to swap.
    - b_idx: Index of the second request to swap.

    Returns:
    - A float representing the change in cost resulting from the swap,
    not performing constraint check.
    """
    assert a_idx != b_idx, "Indices to swap must be different."
    if a_idx > b_idx:
        a_idx, b_idx = b_idx, a_idx

    route = partial.routes[route_idx]
    assert route[a_idx] != 0 and route[b_idx] != 0, "Cannot swap depot nodes."

    D = partial.problem.D

    # Helper to get successor cost, handling end-of-route case
    def is_node(idx: int) -> Optional[int]:
        # Return node at route[idx] if exists, else None
        # Actions count excludes depot at end, so idx should be <= actions
        return route[idx] \
            if 0 <= idx <= partial.route_states[route_idx]["actions"] else None

    def successor_cost(from_node: int, to_node: Optional[int]) -> int:
        if to_node is None:
            return 0
        return D[from_node][to_node]


    # Calculate cost difference 
    if a_idx < b_idx - 1:
        delta = (
            D[route[a_idx - 1]][route[a_idx]]
            + successor_cost(route[a_idx], is_node(a_idx + 1))
            + D[route[b_idx - 1]][route[b_idx]]
            + successor_cost(route[b_idx], is_node(b_idx + 1))
            - D[route[a_idx - 1]][route[b_idx]]
            - successor_cost(route[b_idx], is_node(a_idx + 1))
            - D[route[b_idx - 1]][route[a_idx]]
            - successor_cost(route[a_idx], is_node(b_idx + 1))
        )
    else:
        delta = (
            D[route[a_idx - 1]][route[a_idx]] + D[route[a_idx]][route[b_idx]]
            + successor_cost(route[b_idx], is_node(b_idx + 1))
            - D[route[a_idx - 1]][route[b_idx]] - D[route[b_idx]][route[a_idx]]
            - successor_cost(route[a_idx], is_node(b_idx + 1))
        )

    return delta




def intra_swap_one_route_operator(
        partial: PartialSolution,               # Current partial solution
        route_idx: int,                     # Index of the modified route
        steps: Optional[int] = None,        # Number of steps to consider
        mode: str = 'first',                # Mode of operation
        uplift: int = 1,                    # Minimum improvement required
        seed: int = 42,                     # Random seed for stochastic mode
        verbose: bool = False               # Verbose output
    ) -> Tuple[PartialSolution, List[bool], int]:
    """
    Find the swap of two requests in one vehicle route that improves the solution
    up to a certain extent controlled by ``uplift``. Perform up to ``steps`` swaps
    based on the specified ``mode``.

    Use in post-processing or local search.

    Parameters:
    - partial: PartialSolution object representing the current solution.
    - route_idx: Index of the vehicle route in which to perform the swap.
    - steps: Number of swap steps that the operation should perform, 
    until no improvement is found or the limit is reached. If steps = None, 
    the operation will continue till convergence.
    - mode: Mode of operation, can be 'best', 'first', or 'stochastic'.
    - uplift: Minimum improvement required to consider a swap.
    - seed: Random seed for stochastic mode.

    Returns:
    - A tuple signature containing:
        - A new PartialSolution object with the specified requests swapped.
        - A list of booleans indicating which routes were modified.
        - An integer count of the number of swaps performed.
    """

    # Set random seed for stochastic mode
    rng = random.Random(seed)

    # Extract current route information
    current_par = partial.copy()
    prob = current_par.problem
    K = prob.K
    route = current_par.routes[route_idx]
    n = len(route)


    # Don't perform if route being too short
    if n < 5:
        return current_par, [False] * K, 0
    if steps is None:
        steps = n ** 2

    # Build a position map for O(1) precedence checks
    pos = {node: idx for idx, node in enumerate(route)}
    # Build a prefix array for the current load of the taxi.
    pass_load: list[int] = [0] * n           # passenger onboard (0/1)
    pass_delta: list[int] = [0] * n          # passenger event delta: +1 pickup, -1 drop
    parc_load: list[int] = [0] * n           # parcel quantities onboard (0 <= q <= Q)
    parc_delta: list[int] = [0] * n          # parcel event delta: +q pickup, -q drop

    onboard_pass = 0
    onboard_parcels = 0
    for i in range(n):
        node = route[i]

        delta_pass = 0
        delta_parc = 0
        if prob.is_ppick(node):
            delta_pass = 1
        elif prob.is_pdrop(node):
            delta_pass = -1
        elif prob.is_parc_pick(node):
            jid = prob.rev_parc_pick(node)
            delta_parc = prob.q[jid - 1]
        elif prob.is_parc_drop(node):
            jid = prob.rev_parc_drop(node)
            delta_parc = -prob.q[jid - 1]

        onboard_pass += delta_pass
        onboard_parcels += delta_parc
        pass_load[i] = onboard_pass
        pass_delta[i] = delta_pass
        parc_load[i] = onboard_parcels
        parc_delta[i] = delta_parc


    # Build lazy segment trees for range min/max queries on load
    min_pass_segment = TreeSegment(
        data=pass_load,
        op=min,
        identity=float('inf'),
        sum_like=False
    )
    max_pass_segment = TreeSegment(
        data=pass_load,
        op=max,
        identity=0,
        sum_like=False
    )
    min_parc_segment = TreeSegment(
        data=parc_load,
        op=min,
        identity=float('inf'),
        sum_like=False
    )
    max_parc_segment = TreeSegment(
        data=parc_load,
        op=max,
        identity=0,
        sum_like=False
    )


    # Precedence check for swap of positions a and b
    # Only check pickup-drop precedence of the two swapped nodes
    def check_precedence(a: int, b: int) -> bool:
        assert a != b

        def new_idx(idx: int) -> int:
            if idx == a:
                return b
            if idx == b:
                return a
            return idx

        def check_node(idx_old: int) -> bool:
            node = route[idx_old]
            # Passengers
            if prob.is_ppick(node):
                pid = prob.rev_ppick(node)
                pair = prob.pdrop(pid)
                drop_idx = pos.get(pair)    # Take drop position

                if drop_idx is None:                    # drop not in the route yet
                    return True
                return new_idx(idx_old) < new_idx(drop_idx)

            if prob.is_pdrop(node):
                pid = prob.rev_pdrop(node)
                pair = prob.ppick(pid)
                pickup_idx = pos.get(pair)  # Take pickup position

                if pickup_idx is None:                  # pickup not in the route yet
                    return False
                return new_idx(pickup_idx) < new_idx(idx_old)

            # Parcels (the same logic)
            if prob.is_parc_pick(node):
                jid = prob.rev_parc_pick(node)
                pair = prob.parc_drop(jid)
                drop_idx = pos.get(pair)
                if drop_idx is None:
                    return True
                return new_idx(idx_old) < new_idx(drop_idx)
            if prob.is_parc_drop(node):
                jid = prob.rev_parc_drop(node)
                pair = prob.parc_pick(jid)
                pickup_idx = pos.get(pair)
                if pickup_idx is None:
                    return False
                return new_idx(pickup_idx) < new_idx(idx_old)

            # depot or unknown
            return False

        return check_node(a) and check_node(b)


    def check_passenger(a: int, b: int) -> bool:
        assert a != b

        if a > b:
            a, b = b, a

        swap_delta_pass = pass_delta[b] - pass_delta[a]
        if swap_delta_pass > 0:
            return max_pass_segment.query(a, b) + swap_delta_pass <= 1
        elif swap_delta_pass < 0:
            return min_pass_segment.query(a, b) + swap_delta_pass >= 0
        else:
            return True


    def check_parcel(a: int, b: int) -> bool:
        assert a != b

        if a > b:
            a, b = b, a

        swap_delta_parc = parc_delta[b] - parc_delta[a]
        if swap_delta_parc > 0:
            return max_parc_segment.query(a, b) <= prob.Q[route_idx] - swap_delta_parc
        elif swap_delta_parc < 0:
            return min_parc_segment.query(a, b) >= -swap_delta_parc
        else:
            return True


    # Aggregate feasibility and improvement check
    def check_swap(a: int, b: int) -> tuple[bool, int]:
        if not (check_precedence(a, b)
            and check_passenger(a, b)
            and check_parcel(a, b)
        ):
            return False, 0

        # If all checks pass, return True and the potential improvement
        dec = cost_decrement_intra_swap(
            current_par, route_idx, a, b
        )
        return True, dec

    # Find all candidate pairs (a, b) feasible for swap
    def find_candidates():
        for a in range(1, n - 1):
            for b in range(a + 1, n - 1):
                feasible, dec = check_swap(a, b)
                if not feasible or dec < uplift:
                    continue

                if mode == 'first':     # early exit if there exists one
                    yield (a, b, dec)
                    return
                else:
                    yield (a, b, dec)


    # Select the move from the candidates based on mode
    def select_candidate():
        cand_list = list(find_candidates())
        if not cand_list:
            return
        if mode == 'stochastic':
            return rng.choice(cand_list)
        elif mode == 'best':
            return max(cand_list, key=lambda x: x[2])
        else:
            return cand_list[0]


    # ================ Main loop ================
    # Data to track progress
    swaps_done = 0
    best_improvement = 0
    modified = [False] * K


    def update_partial_solution(action: tuple[int, int, int]) :
        nonlocal route
        nonlocal best_improvement
        nonlocal modified
        nonlocal current_par

        a, b, dec = action
        route[a], route[b] = route[b], route[a]
        current_par.route_costs[route_idx] -= dec
        current_par.max_cost = max(current_par.max_cost, current_par.route_costs[route_idx])


    def update_precalc(action: tuple[int, int, int]) :
        nonlocal route
        nonlocal pos
        nonlocal pass_delta
        nonlocal parc_delta
        nonlocal min_pass_segment
        nonlocal min_parc_segment
        nonlocal max_pass_segment
        nonlocal max_parc_segment

        a, b, _ = action
        if a > b:
            a, b = b, a

        # Update segment trees: swapping deltas at a and b shifts prefix sums
        # on the half-open interval [a, b) by (delta[b] - delta[a]).
        dpass = pass_delta[b] - pass_delta[a]
        if dpass != 0:
            min_pass_segment.update(a, b, dpass)
            max_pass_segment.update(a, b, dpass)
        dparc = parc_delta[b] - parc_delta[a]
        if dparc != 0:
            min_parc_segment.update(a, b, dparc)
            max_parc_segment.update(a, b, dparc)

        # Swap the per-position deltas in O(1)
        pass_delta[a], pass_delta[b] = pass_delta[b], pass_delta[a]
        parc_delta[a], parc_delta[b] = parc_delta[b], parc_delta[a]

        # Update positions map in O(1) using the current route (already swapped)
        pos[route[a]] = a
        pos[route[b]] = b



    def swap_until_convergence():
        nonlocal swaps_done
        while True:
            if steps is not None and swaps_done >= steps:
                break

            action = select_candidate()
            if action is None:
                break

            update_partial_solution(action)
            update_precalc(action)

            swaps_done += 1
            modified[route_idx] = True

            if verbose:
                a, b, dec = action
                print(
                    f"[Route {route_idx}] Swapped positions {a} and {b} "
                    + f"(nodes {route[b]} and {route[a]}). Cost decrease: {dec}."
                )

    swap_until_convergence()
    return current_par, modified, swaps_done



def intra_swap_operator(
        partial: PartialSolution,               # Current partial solution
        steps: Optional[int] = None,        # Number of steps to consider
        mode: str = 'first',                # Mode of operation
        uplift: int = 1,                    # Minimum improvement required
        seed: int = 42,                     # Random seed for stochastic mode
        verbose: bool = False
    ) -> Tuple[PartialSolution, list[bool], int]:
    """
    Intra-route swap operator that finds and performs swaps of two requests
    within the same vehicle route that improves the solution up to a certain extent
    until no further improvement is possible or the step limit is reached.

    Args:
    - partial: PartialSolution instance
    - steps: Number of swaps to perform
    - mode: Mode of operation
    - uplift: Minimum improvement required
    - seed: Random seed for stochastic mode
 
    Returns:
    - A tuple containing:
        - updated PartialSolution
        - boolean indicating if any route was modified
        - number of swaps performed
    """
    if steps is None:
        steps = 10**9

    total_swaps = 0
    modified: list[bool] = [False] * partial.problem.K
    current_par: PartialSolution = partial.copy()
    for route_it in range(partial.problem.K):
        tmp_par, modified_one, n_swaps_one = intra_swap_one_route_operator(
            current_par,
            route_idx=route_it,
            steps=(steps - total_swaps),
            mode=mode,
            uplift=uplift,
            seed=seed,
            verbose=verbose
        )

        current_par = tmp_par
        total_swaps += n_swaps_one
        if modified_one[route_it]:
            modified[route_it] = True

        if verbose:
            print(f"Route {route_it}: performed {n_swaps_one} intra-route swaps.")

    return current_par, modified, total_swaps



def cost_decrement_inter_swap(
        partial: PartialSolution,
        route_a_idx: int, route_b_idx: int,
        p_idx_a: int, d_idx_a: int,
        p_idx_b: int, d_idx_b: int
    ) -> tuple[int, int, int]:
    """
    Compute the decrement in max_cost of swapping two requests (including both pickup and drop)
    between two vehicle routes identified by route_a_idx and route_b_idx.

    Parameters:
    - partial: PartialSolution object representing the current solution.
    - route_a_idx: Index of the first vehicle route.
    - route_b_idx: Index of the second vehicle route.
    - a_idx: Index of the request in the first route to swap.
    - b_idx: Index of the request in the second route to swap.

    Returns:
    - A float representing the change in cost resulting from the swap,
    not performing constraint check.
    """
    route_a = partial.routes[route_a_idx]
    route_b = partial.routes[route_b_idx]
    assert route_a[p_idx_a] != 0 and route_b[p_idx_b] != 0, "Cannot swap depot nodes."

    route_cost_a = partial.route_costs[route_a_idx]
    route_cost_b = partial.route_costs[route_b_idx]
    max_cost_before = partial.max_cost


    # Compute current cost contribution of the two requests
    D = partial.problem.D

    # Helper: Handle the case when the node has no successor.
    # It is the last node currently and thus has no outgoing cost.
    # However, every node has a predecessor since depot is always present.
    def _succ_cost(from_node: int, routechar: str, idx: int) -> int:
        """Return D[from_node][next_node] if successor exists, otherwise 0."""
        if routechar == 'a':
            route = route_a
            route_idx = route_a_idx
        else:
            route = route_b
            route_idx = route_b_idx

        if idx >= partial.route_states[route_idx]["actions"]:
            return 0
        return D[from_node][route[idx + 1]]

    # Compute in-out costs before and after swap for both routes
    if p_idx_a + 1 == d_idx_a:
        in_out_cost_a_before = (
            D[route_a[p_idx_a - 1]][route_a[p_idx_a]]
            + D[route_a[p_idx_a]][route_a[d_idx_a]]
            + _succ_cost(route_a[d_idx_a], 'a', d_idx_a)
        )
        in_out_cost_a_after = (
            D[route_a[p_idx_a - 1]][route_b[p_idx_b]]
            + D[route_b[p_idx_b]][route_b[d_idx_b]]
            + _succ_cost(route_b[d_idx_b], 'a', d_idx_a)
        )
    else:
        in_out_cost_a_before = (
            D[route_a[p_idx_a - 1]][route_a[p_idx_a]]
            + D[route_a[p_idx_a]][route_a[p_idx_a + 1]]
            + D[route_a[d_idx_a - 1]][route_a[d_idx_a]]
            + _succ_cost(route_a[d_idx_a], 'a', d_idx_a)
        )
        in_out_cost_a_after = (
            D[route_a[p_idx_a - 1]][route_b[p_idx_b]]
            + D[route_b[p_idx_b]][route_a[p_idx_a + 1]]
            + D[route_a[d_idx_a - 1]][route_b[d_idx_b]]
            + _succ_cost(route_b[d_idx_b], 'a', d_idx_a)
        )
    if p_idx_b + 1 == d_idx_b:
        in_out_cost_b_before = (
            D[route_b[p_idx_b - 1]][route_b[p_idx_b]]
            + D[route_b[p_idx_b]][route_b[d_idx_b]]
            + _succ_cost(route_b[d_idx_b], 'b', d_idx_b)
        )
        in_out_cost_b_after = (
            D[route_b[p_idx_b - 1]][route_a[p_idx_a]]
            + D[route_a[p_idx_a]][route_a[d_idx_a]]
            + _succ_cost(route_a[d_idx_a], 'b', d_idx_b)
        )
    else:
        in_out_cost_b_before = (
            D[route_b[p_idx_b - 1]][route_b[p_idx_b]] 
            + D[route_b[p_idx_b]][route_b[p_idx_b + 1]]
            + D[route_b[d_idx_b - 1]][route_b[d_idx_b]]
            + _succ_cost(route_b[d_idx_b], 'b', d_idx_b)
        )
        in_out_cost_b_after = (
            D[route_b[p_idx_b - 1]][route_a[p_idx_a]]
            + D[route_a[p_idx_a]][route_b[p_idx_b + 1]]
            + D[route_b[d_idx_b - 1]][route_a[d_idx_a]]
            + _succ_cost(route_a[d_idx_a], 'b', d_idx_b)
        )

    # Compute new route costs and max cost after swap
    route_a_next_cost = route_cost_a - in_out_cost_a_before + in_out_cost_a_after
    route_b_next_cost = route_cost_b - in_out_cost_b_before + in_out_cost_b_after
    max_cost_after = max(
        route_a_next_cost,
        route_b_next_cost,
        *(
            partial.route_costs[i] for i in range(partial.problem.K)
            if i != route_a_idx and i != route_b_idx
        )
    )

    return route_a_next_cost, route_b_next_cost, max_cost_before - max_cost_after



def inter_swap_route_pair_operator (
        partial: PartialSolution,
        route_a_idx: int,
        route_b_idx: int,
        steps: Optional[int] = None,
        mode: str = 'first',
        uplift: int = 1,
        seed: int = 42,
        verbose: bool = False
    ) -> Tuple[PartialSolution, list[bool], int]:

    """
    Find and perform one swap of two requests between two specified vehicle routes 
    that improves the solution up to a certain extent controlled by ``uplift``.

    Parameters:
    - partial: PartialSolution object representing the current solution.
    - route_a_idx: Index of the first vehicle route.
    - route_b_idx: Index of the second vehicle route.
    - mode: Mode of operation, can be 'first', 'best', or 'stochastic'.
    - uplift: Minimum improvement required to consider a swap.
    - seed: Random seed for stochastic mode.

    Returns:
    - A tuple containing:
        - A new PartialSolution object with the specified requests swapped.
        - A boolean indicating whether the swap was successful.
        - An integer count of the number of swaps performed.
    """

    # Set random seed for stochastic mode
    rng = random.Random(seed)

    # Extract current route information
    current_par = partial.copy()
    prob = current_par.problem
    # K not needed here
    route_a = current_par.routes[route_a_idx]
    route_b = current_par.routes[route_b_idx]
    n_a = len(route_a)
    n_b = len(route_b)

    # Don't perform if any route being too short
    if n_a < 5 or n_b < 5:
        return current_par, [False] * prob.K, 0

    # Helpers to build loads, deltas, and segment trees for a route
    def build_loads_and_trees(route: list[int]):
        n = len(route)
        pass_load = [0] * n
        pass_delta = [0] * n
        parc_load = [0] * n
        parc_delta = [0] * n
        onboard_pass = 0
        onboard_parc = 0
        for i, node in enumerate(route):
            dp = 0
            dq = 0
            if prob.is_ppick(node):
                dp = 1
            elif prob.is_pdrop(node):
                dp = -1
            elif prob.is_parc_pick(node):
                jid = prob.rev_parc_pick(node)
                dq = prob.q[jid - 1]
            elif prob.is_parc_drop(node):
                jid = prob.rev_parc_drop(node)
                dq = -prob.q[jid - 1]
            onboard_pass += dp
            onboard_parc += dq
            pass_load[i] = onboard_pass
            pass_delta[i] = dp
            parc_load[i] = onboard_parc
            parc_delta[i] = dq

        # Segment trees for min/max prefix loads
        min_pass_segment = TreeSegment(
            data=pass_load, op=min, identity=float('inf'), sum_like=False
        )
        max_pass_segment = TreeSegment(
            data=pass_load, op=max, identity=0, sum_like=False
        )
        min_parc_segment = TreeSegment(
            data=parc_load, op=min, identity=float('inf'), sum_like=False
        )
        max_parc_segment = TreeSegment(
            data=parc_load, op=max, identity=0, sum_like=False
        )
        pos = {node: i for i, node in enumerate(route)}
        return pos, pass_delta, parc_delta, (
            min_pass_segment, max_pass_segment, min_parc_segment, max_parc_segment
        )

    # Extract loads, deltas, and segment trees for both routes
    pos_a, pass_delta_a, parc_delta_a, trees_a = build_loads_and_trees(route_a)
    pos_b, pass_delta_b, parc_delta_b, trees_b = build_loads_and_trees(route_b)
    min_pass_seg_a, max_pass_seg_a, min_parc_seg_a, max_parc_seg_a = trees_a
    min_pass_seg_b, max_pass_seg_b, min_parc_seg_b, max_parc_seg_b = trees_b

    cap_a = prob.Q[route_a_idx]
    cap_b = prob.Q[route_b_idx]



    # ============== Checker helpers ==============
    def check_passenger(p_idx_a: int, q_idx_a: int, p_idx_b: int, q_idx_b: int) -> bool:
        """
        Load feasibility check for passengers when swapping two requests
        Note that swapping full pairs only induces a constant shift on the inner 
        interval (pA, qA) and (pB, qB).
        """
        # Route A inner-interval shift
        delta_a = pass_delta_b[p_idx_b] - pass_delta_a[p_idx_a]
        if delta_a != 0:
            min_a = min_pass_seg_a.query(p_idx_a, q_idx_a)
            max_a = max_pass_seg_a.query(p_idx_a, q_idx_a)
            if min_a + delta_a < 0 or max_a + delta_a > 1:
                return False

        # Route B inner-interval shift
        delta_b = pass_delta_a[p_idx_a] - pass_delta_b[p_idx_b]
        if delta_b != 0:
            min_b = min_pass_seg_b.query(p_idx_b, q_idx_b)
            max_b = max_pass_seg_b.query(p_idx_b, q_idx_b)
            if min_b + delta_b < 0 or max_b + delta_b > 1:
                return False

        return True


    def check_parcel(p_idx_a: int, q_idx_a: int, p_idx_b: int, q_idx_b: int) -> bool:
        """
        Parcel feasibility mirrors passenger: only inner intervals are shifted,
        by a constant equal to the difference of deltas at the pickup positions.
        """
        # Route A inner-interval shift
        delta_a = parc_delta_b[p_idx_b] - parc_delta_a[p_idx_a]
        if delta_a != 0:
            min_a = min_parc_seg_a.query(p_idx_a, q_idx_a)
            max_a = max_parc_seg_a.query(p_idx_a, q_idx_a)
            if min_a + delta_a < 0 or max_a + delta_a > cap_a:
                return False
        # Route B inner-interval shift
        delta_b = parc_delta_a[p_idx_a] - parc_delta_b[p_idx_b]
        if delta_b != 0:
            min_b = min_parc_seg_b.query(p_idx_b, q_idx_b)
            max_b = max_parc_seg_b.query(p_idx_b, q_idx_b)
            if min_b + delta_b < 0 or max_b + delta_b > cap_b:
                return False
        return True


    def check_swap(
            p_idx_a: int, q_idx_a: int, p_idx_b: int, q_idx_b: int
        ) -> tuple[bool, int, int, int]:
        if not check_passenger(p_idx_a, q_idx_a, p_idx_b, q_idx_b):
            return False, 0, 0, 0
        if not check_parcel(p_idx_a, q_idx_a, p_idx_b, q_idx_b):
            return False, 0, 0, 0

        # Compute total decrement for swapping both pickup and drop pairs simultaneously
        after_cost_a, after_cost_b, dec = cost_decrement_inter_swap(
            current_par,
            route_a_idx, route_b_idx,
            p_idx_a, q_idx_a,
            p_idx_b, q_idx_b,
        )
        return True, after_cost_a, after_cost_b, dec



    # ============== Candidate search and selection ==============
    def find_candidates():
        pickup_indices_a = [
            i for i in range(n_a)
            if prob.is_ppick(route_a[i]) or prob.is_parc_pick(route_a[i])
        ]
        pickup_indices_b = [
            j for j in range(n_b)
            if prob.is_ppick(route_b[j]) or prob.is_parc_pick(route_b[j])
        ]

        for p_idx_a in pickup_indices_a:
            # Locate (pA, qA) in route A
            p_node_a = route_a[p_idx_a]
            if prob.is_ppick(p_node_a):
                pass_id = prob.rev_ppick(p_node_a)
                q_node_a = prob.pdrop(pass_id)
            else:
                parc_id = prob.rev_parc_pick(p_node_a)
                q_node_a = prob.parc_drop(parc_id)
            q_idx_a = pos_a.get(q_node_a)
            if q_idx_a is None:
                continue

            for p_idx_b in pickup_indices_b:
                # Locate (pB, qB) in route B
                p_node_b = route_b[p_idx_b]
                if prob.is_ppick(p_node_b):
                    pass_idx_b = prob.rev_ppick(p_node_b)
                    q_node_b = prob.pdrop(pass_idx_b)
                else:
                    parc_id_b = prob.rev_parc_pick(p_node_b)
                    q_node_b = prob.parc_drop(parc_id_b)
                q_idx_b = pos_b.get(q_node_b)
                if q_idx_b is None:
                    continue

                # # Patch for depot at the end
                # if q_idx_a == n_a - 1 or q_idx_b == n_b - 1:
                #     continue

                feasible, after_cost_a, after_cost_b, dec = check_swap(
                    p_idx_a, q_idx_a, p_idx_b, q_idx_b
                )
                if not feasible or dec < uplift:
                    continue
                if mode == 'first':
                    yield (p_idx_a, q_idx_a, p_idx_b, q_idx_b, after_cost_a, after_cost_b, dec)
                    return
                else:
                    yield (p_idx_a, q_idx_a, p_idx_b, q_idx_b, after_cost_a, after_cost_b, dec)


    def select_candidate():
        cand_list = list(find_candidates())
        if not cand_list:
            return None
        if mode == 'stochastic':
            return rng.choice(cand_list)
        elif mode == 'best':
            return max(cand_list, key=lambda x: x[4])
        else:
            return cand_list[0]


    # ================ Modular commit and loop ================
    # Data to track progress
    swaps_done = 0
    best_improvement = 0
    modified = [False] * prob.K

    # Optional upper bound on iterations to avoid infinite loops
    if steps is None:
        steps = (n_a + n_b) ** 2


    def update_partial_solution(action: tuple[int, int, int, int, int, int, int]):
        nonlocal route_a, route_b, current_par
        pA, qA, pB, qB, new_cost_a, new_cost_b, decrement = action

        # Swap the two nodes (pickup and drop) across the two routes
        a1, a2 = route_a[pA], route_a[qA]
        b1, b2 = route_b[pB], route_b[qB]
        route_a[pA], route_a[qA] = b1, b2
        route_b[pB], route_b[qB] = a1, a2

        # Recompute route costs and global max
        current_par.route_costs[route_a_idx] = new_cost_a
        current_par.route_costs[route_b_idx] = new_cost_b
        current_par.max_cost -= decrement


    def update_precalc(action: tuple[int, int, int, int, int, int, int]):
        nonlocal pos_a, pos_b
        nonlocal pass_delta_a, pass_delta_b
        nonlocal parc_delta_a, parc_delta_b
        nonlocal min_pass_seg_a, max_pass_seg_a, min_parc_seg_a, max_parc_seg_a
        nonlocal min_pass_seg_b, max_pass_seg_b, min_parc_seg_b, max_parc_seg_b

        pA, qA, pB, qB, _, __, ___ = action

        # Compute inner-interval constant shifts using original deltas
        dpass_a = pass_delta_b[pB] - pass_delta_a[pA]
        dparc_a = parc_delta_b[pB] - parc_delta_a[pA]
        if dpass_a != 0:
            min_pass_seg_a.update(pA, qA, dpass_a)
            max_pass_seg_a.update(pA, qA, dpass_a)
        if dparc_a != 0:
            min_parc_seg_a.update(pA, qA, dparc_a)
            max_parc_seg_a.update(pA, qA, dparc_a)

        dpass_b = pass_delta_a[pA] - pass_delta_b[pB]
        dparc_b = parc_delta_a[pA] - parc_delta_b[pB]
        if dpass_b != 0:
            min_pass_seg_b.update(pB, qB, dpass_b)
            max_pass_seg_b.update(pB, qB, dpass_b)
        if dparc_b != 0:
            min_parc_seg_b.update(pB, qB, dparc_b)
            max_parc_seg_b.update(pB, qB, dparc_b)

        # Swap per-position deltas at pickup and drop indices for both routes
        pass_delta_a[pA], pass_delta_b[pB] = pass_delta_b[pB], pass_delta_a[pA]
        pass_delta_a[qA], pass_delta_b[qB] = pass_delta_b[qB], pass_delta_a[qA]
        parc_delta_a[pA], parc_delta_b[pB] = parc_delta_b[pB], parc_delta_a[pA]
        parc_delta_a[qA], parc_delta_b[qB] = parc_delta_b[qB], parc_delta_a[qA]

        # Update positions for moved nodes using current routes
        pos_a[route_a[pA]] = pA
        pos_a[route_a[qA]] = qA
        pos_b[route_b[pB]] = pB
        pos_b[route_b[qB]] = qB


    def swap_until_convergence():
        nonlocal swaps_done, modified, best_improvement
        while swaps_done < steps:
            action = select_candidate()
            if action is None:
                break

            update_partial_solution(action)
            update_precalc(action)

            best_improvement += action[6]
            modified[route_a_idx] = True
            modified[route_b_idx] = True
            swaps_done += 1

            if verbose:
                pA, qA, pB, qB, _, __, dec = action
                print(
                    f"[Routes {route_a_idx} & {route_b_idx}] "
                    + f"Swapped nodes at positions ({pA}, {qA}) and ({pB}, {qB}). "
                    + f"Cost decrease: {dec}."
                )

    swap_until_convergence()
    return current_par, modified, swaps_done



def inter_swap_operator(
        partial: PartialSolution,               # Current partial solution
        steps: Optional[int] = None,        # Number of steps to consider
        mode: str = 'first',                # Mode of operation
        uplift: int = 1,                    # Minimum improvement required
        seed: int = 42,                     # Random seed for stochastic mode
        verbose: bool = False               # Verbose output
    ) -> Tuple[PartialSolution, List[bool], int]:
    """
    Find and swap of two requests between different vehicle routes that improves the solution
    up to a certain extent controlled by ``uplift``. Perform up to ``steps`` swaps
    based on the specified ``mode``.

    Use in post-processing or local search.

    Parameters:
    - partial: PartialSolution object representing the current solution.
    - steps: Number of swap steps that the operation should perform, until no 
    improvement is found or the limit is reached. If steps = None, the operation will
    continue till convergence
    - mode: Mode of operation, can be 'best', 'first', or 'stochastic'.
    - uplift: Minimum improvement required to consider a swap.

    Returns:
    - A tuple signature containing:
        - A new PartialSolution object with the specified requests swapped.
        - A list of booleans indicating which routes were modified.
        - An integer count of the number of swaps performed.
    """
    rng = random.Random(seed)

    K = partial.problem.K
    if K < 2:
        return partial.copy(), [False] * K, 0

    current_par: PartialSolution = partial.copy()
    modified: List[bool] = [False] * K
    total_swaps = 0

    # Treat None as effectively infinite steps
    max_steps = steps if steps is not None else 10**9

    # Priority queues for max and min costs (lazy updates)
    max_heap: list[tuple[int, int]] = [(-c, i) for i, c in enumerate(current_par.route_costs)]
    min_heap: list[tuple[int, int]] = [(c, i) for i, c in enumerate(current_par.route_costs)]
    heapq.heapify(max_heap)
    heapq.heapify(min_heap)

    def pop_valid_max() -> Optional[tuple[int, int]]:
        while max_heap:
            negc, idx = heapq.heappop(max_heap)
            if -negc == current_par.route_costs[idx]:
                return -negc, idx
        return None

    def pop_valid_min(exclude_idx: Optional[int] = None) -> Optional[tuple[int, int]]:
        while min_heap:
            c, idx = heapq.heappop(min_heap)
            if idx == exclude_idx:
                continue
            if c == current_par.route_costs[idx]:
                return c, idx
        return None

    def push_idx(idx: int):
        c = current_par.route_costs[idx]
        heapq.heappush(max_heap, (-c, idx))
        heapq.heappush(min_heap, (c, idx))


    # Main loop: greedily try to improve the worst route by swapping with the best routes
    while True:
        if steps is not None and total_swaps >= max_steps:
            break
        top = pop_valid_max()
        if top is None:
            break
        _, max_idx = top

        # Try pairing max_idx with the smallest-cost routes until a successful swap
        popped_mins: list[tuple[int, int]] = []
        improved = False
        while True:
            mn = pop_valid_min(exclude_idx=max_idx)
            if mn is None:
                break
            _, min_idx = mn
            popped_mins.append(mn)

            next_par, modified_pair, n_swaps_pair = inter_swap_route_pair_operator(
                current_par,
                route_a_idx=max_idx,
                route_b_idx=min_idx,
                steps=(max_steps - total_swaps),
                mode=mode,
                uplift=uplift,
                seed=rng.randint(10, 10**9),
                verbose=verbose
            )
            if n_swaps_pair > 0:
                # Commit improvement
                current_par = next_par
                push_idx(max_idx)
                push_idx(min_idx)

                total_swaps += n_swaps_pair
                if modified_pair[max_idx]:
                    modified[max_idx] = True
                if modified_pair[min_idx]:
                    modified[min_idx] = True
                improved = True

                if verbose:
                    print(
                        f"Inter-route swap between routes {max_idx} and {min_idx} "
                        + f"performed {n_swaps_pair} swaps."
                    )

                break

        # Push back any popped min candidates (their entries may be stale,
        # but lazy check handles it)
        for c, idx in popped_mins:
            heapq.heappush(min_heap, (c, idx))

        if not improved:
            # No improving swap found for current max route with any min route
            # -> convergence
            # Reinsert the max route (optional due to break) to keep heaps
            # consistent if loop continues
            push_idx(max_idx)
            break

    return current_par, modified, total_swaps
