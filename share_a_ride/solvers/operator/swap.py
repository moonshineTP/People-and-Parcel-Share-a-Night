"""
Module for the Swap operator in ride-sharing optimization.
"""
import random
import heapq
from typing import List, Optional, Tuple, Dict

from share_a_ride.core.solution import PartialSolution
from share_a_ride.solvers.operator.utils import TreeSegment

Request = Tuple[int, int, str]




def cost_decrement_intra_swap(
        partial: PartialSolution,
        route_idx: int,
        a_idx: int, len_a: int,
        b_idx: int, len_b: int
    ) -> int:
    """
    Compute the change in cost of swapping two request blocks A and B in the
    same vehicle route. A starts at a_idx with length len_a. B starts at b_idx
    with length len_b.
    """
    # Ensure a_idx < b_idx
    if a_idx > b_idx:
        a_idx, b_idx = b_idx, a_idx
        len_a, len_b = len_b, len_a


    # //// Extract data
    # Distance matrix
    D = partial.problem.D   # pylint: disable=invalid-name

    # Route
    route = partial.routes[route_idx]
    route_len = len(route)

    # //// Compute cost delta
    prev_a = route[a_idx - 1]
    start_a = route[a_idx]
    end_a = route[a_idx + len_a - 1]
    next_a = route[a_idx + len_a] if a_idx + len_a < route_len else None

    prev_b = route[b_idx - 1]
    start_b = route[b_idx]
    end_b = route[b_idx + len_b - 1]
    next_b = route[b_idx + len_b] if b_idx + len_b < route_len else None

    # Calculate local cost
    cost_before = 0
    cost_after = 0
    if a_idx + len_a == b_idx:
        # Adjacent: [Pre] [A] [B] [Post]
        cost_before += D[prev_a][start_a]
        cost_before += D[end_a][start_b]
        if next_b is not None:
            cost_before += D[end_b][next_b]

        # New: [Pre] [B] [A] [Post]
        cost_after += D[prev_a][start_b]
        cost_after += D[end_b][start_a]
        if next_b is not None:
            cost_after += D[end_a][next_b]
    else:
        # Separated: [Pre] [A] [Mid] [B] [Post]
        cost_before += D[prev_a][start_a]
        if next_a is not None:
            cost_before += D[end_a][next_a]
        cost_before += D[prev_b][start_b]
        if next_b is not None:
            cost_before += D[end_b][next_b]

        # New: [Pre] [B] [Mid] [A] [Post]
        cost_after += D[prev_a][start_b]
        if next_a is not None:
            cost_after += D[end_b][next_a]
        cost_after += D[prev_b][start_a]
        if next_b is not None:
            cost_after += D[end_a][next_b]

    return cost_before - cost_after




def intra_swap_one_route_operator(
        partial: PartialSolution,           # Current partial solution
        route_idx: int,                     # Index of the modified route
        steps: Optional[int] = None,        # Number of steps to consider
        mode: str = 'first',                # Mode of operation
        uplift: int = 1,                    # Minimum improvement required
        seed: Optional[int] = None,         # Random seed for stochastic mode
        verbose: bool = False               # Verbose output
    ) -> Tuple[PartialSolution, List[bool], int]:
    """
    Find the swap of two requests in one vehicle route that improves the solution
    up to a certain extent controlled by ``uplift``. Perform up to ``steps`` swaps
    based on the specified ``mode``.

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
    partial = partial.copy()
    prob = partial.problem
    route = partial.routes[route_idx]
    k_vehicles = prob.K                      # pylint: disable=invalid-name
    n_nodes = len(route)

    # Don't perform if route being too short
    if n_nodes < 5:
        return partial, [False] * k_vehicles, 0
    if steps is None:
        steps = n_nodes ** 2


    # //// Build data structures ////
    # Node/index position map
    pos = {}
    actions: List[List[int]] = []        # List of [node_idx, length]
    action_deltas: List[int] = []
    node_to_action = {}


    def build_structures():
        nonlocal pos, actions, action_deltas, node_to_action

        # Build position map
        pos = {node: idx for idx, node in enumerate(route)}

        # Build action data structures
        cur_idx = 1
        action_idx = 0
        while cur_idx < n_nodes - 1:
            node = route[cur_idx]
            length = 0
            delta = 0

            if prob.is_ppick(node):
                length = 2
                delta = 0
            elif prob.is_lpick(node):
                length = 1
                lid = prob.rev_lpick(node)
                delta = prob.q[lid - 1]
            elif prob.is_ldrop(node):
                length = 1
                lid = prob.rev_ldrop(node)
                delta = -prob.q[lid - 1]
            else:
                cur_idx += 1
                continue

            actions.append([cur_idx, length])
            action_deltas.append(delta)
            for k in range(length):
                node_to_action[cur_idx + k] = action_idx

            cur_idx += length
            action_idx += 1

    # Main builder
    build_structures()

    # Number of actions
    n_actions = len(actions)

    # Build partners map (action_idx -> partner_action_idx)
    partners = [-1] * n_actions
    for i in range(n_actions):
        node_idx, _length = actions[i]
        node = route[node_idx]
        if prob.is_lpick(node):
            lid = prob.rev_lpick(node)
            drop_node = prob.ldrop(lid)
            if drop_node in pos:
                drop_idx = pos[drop_node]
                if drop_idx in node_to_action:
                    p_idx = node_to_action[drop_idx]
                    partners[i] = p_idx
                    partners[p_idx] = i

    # Build segment tree for fast check
    action_loads = [0] * n_actions
    curr = 0
    for i in range(n_actions):
        curr += action_deltas[i]
        action_loads[i] = curr

    min_load_segment = TreeSegment(action_loads, min, 10**18, False)
    max_load_segment = TreeSegment(action_loads, max, 0, False)


    # //// Checker helpers ////
    # Precedence check for swap of actions i and j
    def check_precedence(i: int, j: int) -> bool:
        if action_deltas[i] > 0 and partners[i] <= j:
            return False

        if action_deltas[j] < 0 and partners[j] >= i:
            return False
        return True

    # Load check for swap of actions i and j
    def check_load(i: int, j: int) -> bool:
        delta_i = action_deltas[i]
        delta_j = action_deltas[j]
        diff = delta_j - delta_i

        if diff > 0:
            if max_load_segment.query(i, j) + diff > prob.Q[route_idx]:
                return False
        elif diff < 0:
            if min_load_segment.query(i, j) + diff < 0:
                return False

        return True

    # Aggregate check
    def check_swap(i: int, j: int) -> Tuple[bool, int]:
        if not (check_precedence(i, j) and check_load(i, j)):
            return False, 0

        idx_a, len_a = actions[i]
        idx_b, len_b = actions[j]
        dec = cost_decrement_intra_swap(partial, route_idx, idx_a, len_a, idx_b, len_b)
        return True, dec


    # //// Candidate search and selection ////
    def find_candidates():
        for i in range(n_actions):
            for j in range(i + 1, n_actions):
                feasible, dec = check_swap(i, j)
                if not feasible or dec < uplift:
                    continue

                yield (i, j, dec)
                if mode == 'first':
                    return

    def select_candidate():
        cand_list = list(find_candidates())
        if not cand_list:
            return None
        if mode == 'first':
            return cand_list[0]
        elif mode == 'best':
            return max(cand_list, key=lambda x: x[2])
        elif mode == 'stochastic':
            return rng.choice(cand_list)


    # ================ Main loop ================
    swaps_done = 0
    modified = [False] * k_vehicles

    # Update the route and cost
    def update_partial(action):
        nonlocal route, partial
        i, j, dec = action
        idx_a, len_a = actions[i]
        idx_b, len_b = actions[j]

        # Construct new route
        new_route = (
            route[:idx_a]
            + route[idx_b : idx_b + len_b]
            + route[idx_a + len_a : idx_b]
            + route[idx_a : idx_a + len_a]
            + route[idx_b + len_b:]
        )

        # Update route and cost
        route[:] = new_route
        partial.decrease_cost(route_idx, dec)


    # Update related data structures
    def update_ds(action):
        nonlocal action_deltas, partners, actions, pos, node_to_action
        i, j, _ = action

        # 1. Update segment trees & deltas
        diff = action_deltas[j] - action_deltas[i]
        if diff:
            min_load_segment.update(i, j, diff)
            max_load_segment.update(i, j, diff)
        action_deltas[i], action_deltas[j] = action_deltas[j], action_deltas[i]

        # 2. Update partners
        if partners[i] != -1:
            partners[partners[i]] = j
        if partners[j] != -1:
            partners[partners[j]] = i
        partners[i], partners[j] = partners[j], partners[i]

        # 3. Update actions list
        idx_a, len_a = actions[i]
        idx_b, len_b = actions[j]
        shift = len_b - len_a

        actions[i], actions[j] = actions[j], actions[i]
        actions[i][0], actions[j][0] = idx_a, idx_b + shift

        if shift:
            for k in range(i + 1, j):
                actions[k][0] += shift

        # 4. Rebuild maps
        pos = {node: idx for idx, node in enumerate(route)}
        node_to_action = {
            actions[k][0] + l: k
            for k in range(n_actions)
            for l in range(actions[k][1])
        }


    # //// Execution loop ////
    while True:
        if steps is not None and swaps_done >= steps:
            break

        action = select_candidate()
        if action is None:
            break

        update_partial(action)
        update_ds(action)

        swaps_done += 1
        modified[route_idx] = True

        if verbose:
            i, j, dec = action
            print(f"[IntraSwap] [Route {route_idx}] Swapped actions {i + 1}<->{j + 1}. Dec={dec}")
            if not partial.is_valid(verbose=verbose):
                raise ValueError("Intra-swap operator produced invalid solution during execution.")

    return partial, modified, swaps_done




def intra_swap_operator(
        partial: PartialSolution,           # Current partial solution
        steps: Optional[int] = None,        # Number of steps to consider
        mode: str = 'first',                # Mode of operation
        uplift: int = 1,                    # Minimum improvement required
        seed: Optional[int] = None,         # Random seed for stochastic mode
        verbose: bool = False
    ) -> Tuple[PartialSolution, List[bool], int]:
    """
    Intra-route swap operator that finds and performs swaps of two requests
    within the same vehicle route that improves the solution up to a certain extent
    until no further improvement is possible or the step limit is reached.

    Params:
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
    K = partial.problem.K       # pylint: disable=invalid-name
    modified: List[bool] = [False] * K
    current_par: PartialSolution = partial.copy()
    for route_idx in range(K):
        tmp_par, modified_one, n_swaps_one = intra_swap_one_route_operator(
            current_par,
            route_idx=route_idx,
            steps=(steps - total_swaps),
            mode=mode,
            uplift=uplift,
            seed=seed,
            verbose=verbose
        )

        current_par = tmp_par
        total_swaps += n_swaps_one
        if modified_one[route_idx]:
            modified[route_idx] = True

        if verbose:
            print(f"[IntraSwap] [Route {route_idx}]: performed {n_swaps_one} intra-route swaps.")

    # Logging
    if verbose:
        print()
        print("[IntraSwap] Operator completed.")
        print(f"[IntraSwap] Total swaps performed: {total_swaps};")
        print(f"[IntraSwap] Decrement: {partial.max_cost - current_par.max_cost};")
        print(f"[IntraSwap] New max cost: {current_par.max_cost}.")
        print("------------------------------")
        print()

    if current_par.is_valid(verbose=verbose) is False:
        current_par.stdin_print()
        raise ValueError("Intra-swap operator produced invalid solution.")

    return current_par, modified, total_swaps




def cost_decrement_inter_swap(
        partial: PartialSolution,
        raidx: int,
        rbidx: int,
        paidx: int,
        qaidx: int,
        pbidx: int,
        qbidx: int,
    ) -> Tuple[int, int, int]:
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

    # //// Extract data
    # Distance matrix
    D = partial.problem.D   # pylint: disable=invalid-name

    # Routes
    route_a = partial.routes[raidx]
    route_b = partial.routes[rbidx]
    route_a_len = len(route_a)
    route_b_len = len(route_b)
    assert route_a[paidx] != 0 and route_b[pbidx] != 0, "Cannot swap depot nodes."

    # Costs
    route_a_cost = partial.route_costs[raidx]
    route_b_cost = partial.route_costs[rbidx]
    max_cost_before = partial.max_cost


    #//// Compute cost deltas
    # Nodes involved
    paprev = route_a[paidx - 1]
    pa = route_a[paidx]
    pasucc = route_a[paidx + 1]
    qaprev = route_a[qaidx - 1]
    qa = route_a[qaidx]
    qasucc = None
    if qaidx + 1 < route_a_len:
        qasucc = route_a[qaidx + 1]

    pbprev = route_b[pbidx - 1]
    pb = route_b[pbidx]
    pbsucc = route_b[pbidx + 1]
    qbprev = route_b[qbidx - 1]
    qb = route_b[qbidx]
    qbsucc = None
    if qbidx + 1 < route_b_len:
        qbsucc = route_b[qbidx + 1]

    # Route A delta
    cost_delta_a = 0
    if paidx + 1 == qaidx:      # Adjacent: [Prev] [P] [Q] [Succ]
        cost_delta_a -= D[paprev][pa] + D[pa][qa]
        if qasucc is not None:
            cost_delta_a -= D[qa][qasucc]

        cost_delta_a += D[paprev][pb] + D[pb][qb]
        if qasucc is not None:
            cost_delta_a += D[qb][qasucc]
    else:                       # Separated: [Prev] [P] [Mid] [Q] [Succ]
        cost_delta_a -= D[paprev][pa] + D[pa][pasucc] + D[qaprev][qa]
        if qasucc is not None:
            cost_delta_a -= D[qa][qasucc]

        cost_delta_a += D[paprev][pb] + D[pb][pasucc] + D[qaprev][qb]
        if qasucc is not None:
            cost_delta_a += D[qb][qasucc]

    # Route B delta
    cost_delta_b = 0
    if pbidx + 1 == qbidx:      # Adjacent: [Prev] [P] [Q] [Succ]
        cost_delta_b -= D[pbprev][pb] + D[pb][qb]
        if qbsucc is not None:
            cost_delta_b -= D[qb][qbsucc]

        cost_delta_b += D[pbprev][pa] + D[pa][qa]
        if qbsucc is not None:
            cost_delta_b += D[qa][qbsucc]
    else:                       # Separated: [Prev] [P] [Mid] [Q] [Succ]
        cost_delta_b -= D[pbprev][pb] + D[pb][pbsucc] + D[qbprev][qb]
        if qbsucc is not None:
            cost_delta_b -= D[qb][qbsucc]

        cost_delta_b += D[pbprev][pa] + D[pa][pbsucc] + D[qbprev][qa]
        if qbsucc is not None:
            cost_delta_b += D[qa][qbsucc]


    # //// Compute new route costs
    racost_after = route_a_cost + cost_delta_a
    rbcost_after = route_b_cost + cost_delta_b
    remaining_costs = [
        partial.route_costs[i] for i in range(partial.problem.K)
        if i != raidx and i != rbidx
    ]

    max_cost_after = max(racost_after, rbcost_after, *remaining_costs)
    cost_dec = max_cost_before - max_cost_after

    return racost_after, rbcost_after, cost_dec




def inter_swap_route_pair_operator (
        partial: PartialSolution,           # Current partial solution
        route_a_idx: int,                   # Index of the first vehicle route
        route_b_idx: int,                   # Index of the second vehicle route
        steps: Optional[int] = None,        # Number of steps to consider
        mode: str = 'first',                # Mode of operation
        uplift: int = 1,                    # Minimum improvement required
        seed: Optional[int] = None,         # Random seed for stochastic mode
        verbose: bool = False               # Verbose output flag
    ) -> Tuple[PartialSolution, List[bool], int]:

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
    def build_ds(route: List[int]):
        n_nodes = len(route)
        cum_loads = [0] * n_nodes
        load_delta = [0] * n_nodes
        load = 0
        for i, node in enumerate(route):
            dq = 0

            if prob.is_ppick(node) or prob.is_pdrop(node):
                pass
            elif prob.is_lpick(node):
                lid = prob.rev_lpick(node)
                dq = prob.q[lid - 1]
            elif prob.is_ldrop(node):
                lid = prob.rev_ldrop(node)
                dq = -prob.q[lid - 1]

            load += dq
            cum_loads[i] = load
            load_delta[i] = dq

        # Segment trees for min/max prefix loads
        min_load_segment = TreeSegment(
            data=cum_loads, op=min, identity=10**18, sum_like=False
        )
        max_load_segment = TreeSegment(
            data=cum_loads, op=max, identity=0, sum_like=False
        )
        pos = {node: i for i, node in enumerate(route)}

        return pos, load_delta, min_load_segment, max_load_segment

    # Extract loads, deltas, and segment trees for both routes
    pos_a, load_delta_a, min_load_segment_a, max_load_segment_a = build_ds(route_a)
    pos_b, load_delta_b, min_load_segment_b, max_load_segment_b = build_ds(route_b)

    cap_a = prob.Q[route_a_idx]
    cap_b = prob.Q[route_b_idx]


    # ============== Checker helpers ==============
    # Precedence is already ensured by pickup-drop structure
    # so we only need to check load feasibility and consecutivity
    def check_load(req_a: Tuple[int, int, str], req_b: Tuple[int, int, str]) -> bool:
        """
        Verify that swapping the two requests does not violate vehicle capacity
        """
        paidx, qaidx, _kind_a = req_a
        pbidx, qbidx, _kind_b = req_b
        # Change in parcel load applied over the inner interval of route A
        delta_a = load_delta_b[pbidx] - load_delta_a[paidx]
        # Symmetric change for route B
        delta_b = load_delta_a[paidx] - load_delta_b[pbidx]

        # Route A inner-interval shift
        if delta_a != 0:
            min_a = min_load_segment_a.query(paidx, qaidx)
            max_a = max_load_segment_a.query(paidx, qaidx)
            if min_a + delta_a < 0 or max_a + delta_a > cap_a:
                return False

        # Route B inner-interval shift
        if delta_b != 0:
            min_b = min_load_segment_b.query(pbidx, qbidx)
            max_b = max_load_segment_b.query(pbidx, qbidx)
            if min_b + delta_b < 0 or max_b + delta_b > cap_b:
                return False

        return True


    def check_consecutivity(req_a: Tuple[int, int, str], req_b: Tuple[int, int, str]) -> bool:
        """
        Ensure that if a passenger pickup is swapped, its drop-off remains
        consecutive after swap.
        """
        paidx, qaidx, kind_a = req_a
        pbidx, qbidx, kind_b = req_b

        if kind_a == 'serveP':
            if qbidx != pbidx + 1:
                return False
        if kind_b == 'serveP':
            if qaidx != paidx + 1:
                return False

        return True


    def check_swap(
            req_a: Request,
            req_b: Request
        ) -> Tuple[bool, int, int, int]:
        paidx, qaidx, _ = req_a
        pbidx, qbidx, _ = req_b

        if not check_consecutivity(req_a, req_b):
            return False, 0, 0, 0
        if not check_load(req_a, req_b):
            return False, 0, 0, 0

        # Compute total decrement for swapping both pickup and drop pairs simultaneously
        after_cost_a, after_cost_b, dec = cost_decrement_inter_swap(
            current_par,
            route_a_idx, route_b_idx,
            paidx, qaidx,
            pbidx, qbidx,
        )
        return True, after_cost_a, after_cost_b, dec


    # ============== Candidate search and selection ==============
    def find_candidates():
        pickup_indices_a = [
            actid for actid in range(n_a)
            if prob.is_ppick(route_a[actid]) or prob.is_lpick(route_a[actid])
        ]
        pickup_indices_b = [
            actid for actid in range(n_b)
            if prob.is_ppick(route_b[actid]) or prob.is_lpick(route_b[actid])
        ]

        def form_request(p_idx: int, route: List[int], pos: Dict[int, int]) -> Optional[Request]:
            p_node = route[p_idx]
            kind = ''
            if prob.is_ppick(p_node):
                q_idx = p_idx + 1
                kind = 'serveP'
            else:
                lid = prob.rev_lpick(p_node)
                q_node = prob.ldrop(lid)
                if q_node not in pos:
                    return None

                q_idx = pos[q_node]
                kind = 'serveL'

            return (p_idx, q_idx, kind)


        for paidx in pickup_indices_a:
            req_a = form_request(paidx, route_a, pos_a)
            if req_a is None:
                continue

            for pbidx in pickup_indices_b:
                req_b = form_request(pbidx, route_b, pos_b)
                if req_b is None:
                    continue

                feasible, after_cost_a, after_cost_b, dec = check_swap(
                    req_a, req_b
                )
                if not feasible or dec < uplift:
                    continue

                yield (req_a, req_b, after_cost_a, after_cost_b, dec)
                if mode == 'first':
                    return


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
    if steps is None:
        steps = 10**9   # Effectively unlimited

    # Helper to update the partial solution
    def update_partial(action: Tuple[Request, Request, int, int, int]):
        nonlocal route_a, route_b, current_par
        nonlocal pos_a, pos_b

        # Unpack action
        req_a, req_b, new_cost_a, new_cost_b, decrement = action
        paidx, qaidx, _ = req_a
        pbidx, qbidx, _ = req_b
        pa, qa = route_a[paidx], route_a[qaidx]
        pb, qb = route_b[pbidx], route_b[qbidx]

        # Update position maps
        del pos_a[pa]
        del pos_a[qa]
        pos_a[pb] = paidx
        pos_a[qb] = qaidx

        del pos_b[pb]
        del pos_b[qb]
        pos_b[pa] = pbidx
        pos_b[qa] = qbidx

        # Update routes
        route_a[paidx], route_a[qaidx] = pb, qb
        route_b[pbidx], route_b[qbidx] = pa, qa

        # Update node assignments
        current_par.node_assignment[pa] = route_b_idx
        current_par.node_assignment[qa] = route_b_idx
        current_par.node_assignment[pb] = route_a_idx
        current_par.node_assignment[qb] = route_a_idx

        # Update costs
        current_par.route_costs[route_a_idx] = new_cost_a
        current_par.route_costs[route_b_idx] = new_cost_b
        current_par.max_cost -= decrement


    # Helper to update precalculated data structures
    def update_segment(action: Tuple[Request, Request, int, int, int]):
        nonlocal pos_a, pos_b
        nonlocal load_delta_a, load_delta_b
        nonlocal min_load_segment_a, max_load_segment_a
        nonlocal min_load_segment_b, max_load_segment_b

        req_a, req_b, _, __, ___ = action
        pa, qa, _ = req_a
        pb, qb, _ = req_b


        # //// Update load containers
        # Update segment trees
        dparc_a = load_delta_b[pb] - load_delta_a[pa]
        if dparc_a != 0:
            min_load_segment_a.update(pa, qa, dparc_a)
            max_load_segment_a.update(pa, qa, dparc_a)
        dparc_b = load_delta_a[pa] - load_delta_b[pb]
        if dparc_b != 0:
            min_load_segment_b.update(pb, qb, dparc_b)
            max_load_segment_b.update(pb, qb, dparc_b)

        # Update load deltas
        load_delta_a[pa], load_delta_b[pb] = load_delta_b[pb], load_delta_a[pa]
        load_delta_a[qa], load_delta_b[qb] = load_delta_b[qb], load_delta_a[qa]


    # Main execution loop
    def swap_until_convergence():
        nonlocal swaps_done, modified, best_improvement
        while swaps_done < steps:
            action = select_candidate()
            if action is None:
                break

            update_segment(action)
            update_partial(action)

            best_improvement += action[4]
            modified[route_a_idx] = True
            modified[route_b_idx] = True
            swaps_done += 1

            if verbose:
                req_a, req_b, _, __, dec = action
                pa, qa, _ = req_a
                pb, qb, _ = req_b
                print(
                    f"[InterSwap] [Routes {route_a_idx} & {route_b_idx}] "
                    + f"Swapped nodes at positions ({pa}, {qa}) and ({pb}, {qb}). "
                    + f"Cost decrease: {dec}."
                )

    swap_until_convergence()
    return current_par, modified, swaps_done




def inter_swap_operator(
        partial: PartialSolution,           # Current partial solution
        steps: Optional[int] = None,        # Number of steps to consider
        mode: str = 'first',                # Mode of operation
        uplift: int = 1,                    # Minimum improvement required
        seed: Optional[int] = None,         # Random seed for stochastic mode
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
    K = partial.problem.K       # pylint: disable=invalid-name
    if K < 2:
        return partial.copy(), [False] * K, 0

    current_par: PartialSolution = partial.copy()
    modified: List[bool] = [False] * K
    total_swaps = 0

    # Treat None as effectively unlimited steps
    max_steps = steps if steps is not None else 10**9

    # Priority queues for max and min costs (lazy updates)
    max_heap: List[Tuple[int, int]] = [(-c, i) for i, c in enumerate(current_par.route_costs)]
    min_heap: List[Tuple[int, int]] = [(c, i) for i, c in enumerate(current_par.route_costs)]
    heapq.heapify(max_heap)
    heapq.heapify(min_heap)


    # //// Helper functions for heaps with lazy updates
    # Pop the valid maximum cost route
    def pop_valid_max() -> Optional[Tuple[int, int]]:
        while max_heap:
            negc, idx = heapq.heappop(max_heap)
            if -negc == current_par.route_costs[idx]:
                return -negc, idx
        return None

    # Pop the valid minimum cost route, excluding a specific index
    def pop_valid_min(exclude_idx: Optional[int] = None) -> Optional[Tuple[int, int]]:
        while min_heap:
            c, idx = heapq.heappop(min_heap)
            if idx == exclude_idx:
                continue
            if c == current_par.route_costs[idx]:
                return c, idx
        return None

    # Push a route index back into both heaps
    def push_idx(idx: int):
        c = current_par.route_costs[idx]
        heapq.heappush(max_heap, (-c, idx))
        heapq.heappush(min_heap, (c, idx))


    # //// Main loop
    # The heuristic here is greedily try to improve the worst route
    # by swapping with the best routes
    while True:
        if steps is not None and total_swaps >= max_steps:
            break

        top = pop_valid_max()
        if top is None:
            break
        _, max_idx = top

        # Try pairing max_idx with the smallest-cost routes until a successful swap
        popped_mins: List[Tuple[int, int]] = []
        improved = False
        while True:
            min_tuple = pop_valid_min(exclude_idx=max_idx)
            if min_tuple is None:
                break
            _, min_idx = min_tuple
            popped_mins.append(min_tuple)

            # Proceed to attempt inter-route swap
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

            # Analyze result
            if n_swaps_pair > 0:    # Commit improvement
                # Update heaps with modified routes
                push_idx(max_idx)
                push_idx(min_idx)

                # Update current solution
                current_par = next_par

                # Update total swaps
                total_swaps += n_swaps_pair

                # Mark modified routes
                if modified_pair[max_idx]:
                    modified[max_idx] = True
                if modified_pair[min_idx]:
                    modified[min_idx] = True

                # Logging
                if verbose:
                    print(
                        f"[InterSwap] Perform {n_swaps_pair} swaps "
                        f"between routes {max_idx} and {min_idx}."
                    )

                improved = True
                break

        # Push back any popped min candidates
        for c, idx in popped_mins:
            heapq.heappush(min_heap, (c, idx))

        # If no improvement found, convergence reached
        if not improved:
            push_idx(max_idx)
            break

    # Logging
    if verbose:
        print()
        print("[InterSwap] Operator completed.")
        print(f"[InterSwap] Total swaps performed: {total_swaps};")
        print(f"[InterSwap] Decrement: {partial.max_cost - current_par.max_cost};")
        print(f"[InterSwap] New max cost: {current_par.max_cost}.")
        print("------------------------------")
        print()

    if current_par.is_valid(verbose=verbose) is False:
        raise ValueError("Inter-swap operator produced invalid solution.")

    return current_par, modified, total_swaps
