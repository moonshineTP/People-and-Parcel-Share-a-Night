"""
Relocate operator for Share-a-Ride problem.
Relocates a request from one vehicle route to another.
"""
import random
from typing import Iterator, List, Tuple, Optional

from share_a_ride.core.solution import PartialSolution
from share_a_ride.solvers.operator.utils import MinMaxPfsumArray

Request = Tuple[int, int, int, int, str]
CostChange = Tuple[int, int, int]
Action = Tuple[Request, CostChange]



def cost_decrement_relocate(
        partial: PartialSolution,
        rfidx: int,
        rtidx: int,
        pfidx: int,
        qfidx: int,
        ptidx: int,
        qtidx: int
    ) -> CostChange:
    """
    Compute the decrement in max_cost of relocating a full request (pickup and drop)
    from one route to another at specified insertion indices, as Cost_before - Cost_after.
    This assumes the pickup/drop indices are correct with some defensive assertions.

    Parameters:
    - partial: PartialSolution object representing the current solution.
    - from_route_idx: Index of the route from which the request is relocated.
    - to_route_idx: Index of the route to which the request is relocated.
    - pfidx: Index of the pickup node in the from_route.
    - qfidx: Index of the drop node in the from_route.
    - ptidx: Index in the to_route where the pickup node will be inserted.
    - qtidx: Index in the to_route where the drop node will be inserted.

    Returns: A tuple containing:
    - from_route_next_cost: Cost of the from_route after relocation.
    - to_route_next_cost: Cost of the to_route after relocation.
    - cost_decrement: Decrement in max_cost due to the relocation.
    """
    D = partial.problem.D       # pylint: disable=invalid-name
    cur_cost = partial.max_cost

    routef = partial.routes[rfidx]
    routet = partial.routes[rtidx]
    rfcost = partial.route_costs[rfidx]
    rtcost = partial.route_costs[rtidx]
    pf = routef[pfidx]
    qf = routef[qfidx]

    # Basic bound assertions
    assert 1 <= pfidx < qfidx, "Out-of-bounds pickup/drop indices in from_route"
    assert 1 <= ptidx < qtidx, "Out-of-bounds pickup/drop indices in to_route"


    # //// Cost change for from_route: remove both pickup and drop
    pprev = routef[pfidx - 1]
    pnext = routef[pfidx + 1]
    qprev = routef[qfidx - 1]
    qnext = routef[qfidx + 1]
    cost_delta_f = 0

    if pfidx + 1 == qfidx:      # Adjacent
        cost_delta_f -= D[pprev][pf] + D[pf][qf] + D[qf][qnext]
        cost_delta_f += D[pprev][qnext]
    else:                       # Non-adjacent
        cost_delta_f -= D[pprev][pf] + D[pf][pnext] + D[qprev][qf] + D[qf][qnext]
        cost_delta_f += D[pprev][pnext] + D[qprev][qnext]

    rfcost_after = rfcost + cost_delta_f


    # ---------- Cost change for to_route: insert pickup then drop ----------
    # Stitch neighbors for insertion (short, consistent names)
    pprev_to = routet[ptidx - 1]
    pnext_to = routet[ptidx]
    qprev_to = routet[qtidx - 2]    # -2 because shift after pickup insertion
    qnext_to = routet[qtidx - 1]    # -1 because shift after pickup insertion
    cost_delta_t = 0

    if qtidx == ptidx + 1:  # Adjacent insertion
        cost_delta_t -= D[pprev_to][qnext_to]
        cost_delta_t += D[pprev_to][pf] + D[pf][qf] + D[qf][qnext_to]
    else:  # Non-adjacent insertion
        cost_delta_t -= D[pprev_to][pnext_to] + D[qprev_to][qnext_to]
        cost_delta_t += D[pprev_to][pf] + D[pf][pnext_to] + D[qprev_to][qf] + D[qf][qnext_to]

    rtcost_after = rtcost + cost_delta_t


    # //// Compute the next max cost
    remain_costs = [
        partial.route_costs[i]
        for i in range(partial.problem.K) if i != rfidx and i != rtidx
    ]
    next_cost = max(rfcost_after, rtcost_after, *remain_costs)

    return rfcost_after, rtcost_after, cur_cost - next_cost




def relocate_from_to(
        partial: PartialSolution,   # Partial solution object
        route_from_idx: int,        # Donor route id
        route_to_idx: int,          # Receiver route id
        steps: int,                 # Number of steps to consider
        mode: str,                  # Mode of operation
        uplift: int = 1,            # Minimum improvement required
        seed: Optional[int] = None, # Seed for reproducibility
        verbose: bool = False       # Verbose output flag
    ) -> Tuple[PartialSolution, List[bool], int]:
    """
    Attempt to relocate requests from one predefined vehicle route to another that
    improves the solution as a helper to the main relocate_operator.

    Parameters:
    - partial: PartialSolution object representing the current solution.
    - from_route_idx: Index of the route from which to relocate requests.
    - to_route_idx: Index of the route to which to relocate requests.
    - steps: Number of steps to consider
    - mode: Mode of operation
    - uplift: Integer controlling the extent of improvement required.
    - seed: Random seed for stochastic modes.
    - verbose: If True, print detailed logs.
    """
    # RNG for stochastic behavior (if used later)
    rng = random.Random(seed)

    # Instance object data
    prob = partial.problem
    current_par = partial.copy()

    # Route data
    route_from = current_par.routes[route_from_idx]
    route_to = current_par.routes[route_to_idx]
    n_from = len(route_from)
    n_to = len(route_to)

    # Early exit if donor route too short
    if n_from < 5:
        return partial, [False] * prob.K, 0    # No requests to relocate


    # //// Build data structures
    def build_ds(route: List[int], n: int):
        load_deltas = [0] * n
        for i, node in enumerate(route):
            if prob.is_lpick(node):
                jid = prob.rev_lpick(node)
                dq = prob.q[jid - 1]
            elif prob.is_ldrop(node):
                jid = prob.rev_ldrop(node)
                dq = -prob.q[jid - 1]
            else:
                dq = 0

            load_deltas[i] = dq

        load_delta_manager = MinMaxPfsumArray(load_deltas)

        return load_delta_manager

    # Build load delta managers
    load_delta_from_manager = build_ds(route_from, n_from)
    load_delta_to_manager = build_ds(route_to, n_to)

    # Capacity of routes
    cap_from = prob.Q[route_from_idx]
    cap_to = prob.Q[route_to_idx]


    # //// Checker helpers
    def check_consecutive(req: Request) -> bool:
        """
        Ensure that pickup and drop indices are consecutive for 'serveP' requests.
        """
        pfidx, qfidx, ptidx, qtidx, kind = req

        if kind == "serveL":
            return True

        assert qfidx == pfidx + 1
        return qtidx == ptidx + 1


    def check_load(req: Request) -> bool:
        """
        Ensure load stays within [0,cap] after relocating a full request
        (pfidx, qfidx) to (ptidx, qtidx) for both routes.
        """
        pfidx, qfidx, ptidx, qtidx, kind = req
        if kind == "serveP":
            return True

        # Compute load delta
        pf = route_from[pfidx]
        if prob.is_lpick(pf):
            jid = prob.rev_lpick(pf)
            load_delta = prob.q[jid - 1]
        else:
            load_delta = 0

        # Check segment loads on from_route
        load_min_fr = load_delta_from_manager.query_min_prefix(pfidx, qfidx)
        load_max_fr = load_delta_from_manager.query_max_prefix(pfidx, qfidx)
        if load_min_fr - load_delta < 0:
            return False
        if load_max_fr - load_delta > cap_from:
            return False

        # Check segment loads on to_route
        load_min_to = load_delta_to_manager.query_min_prefix(ptidx - 1, qtidx - 1)
        load_max_to = load_delta_to_manager.query_max_prefix(ptidx - 1, qtidx - 1)
        if load_min_to + load_delta < 0:
            return False
        if load_max_to + load_delta > cap_to:
            return False

        return True


    def check_relocate(req: Request) -> Optional[CostChange]:
        """
        Check feasibility of relocating the request defined by (pfidx, qfidx)
        from route_from to route_to at insertion indices (ptidx, qtidx).
        
        Returns a tuple of:
            - feasibility (bool)
            - after_cost_a (int): cost of from_route after relocation
            - after_cost_b (int): cost of to_route after relocation
            - dec (int): total cost decrement if relocation is performed
        """
        # Check consecutiveness and load feasibility
        if not check_consecutive(req):
            return None
        if not check_load(req):
            return None

        # Compute cost change
        cost_change = cost_decrement_relocate(
            current_par, route_from_idx, route_to_idx,
            req[0], req[1], req[2], req[3],
        )
        return cost_change


    def find_candidates() -> Iterator[Tuple[Request, CostChange]]:
        """
        Find candidate relocation requests according to the specified mode.
        
        Yields tuples of the form (Request, CostChange).
        """
        # Build position map for route_from to locate corresponding drops
        pos_from = {node: i for i, node in enumerate(route_from)}

        # Enumerate all pickup positions in route_from
        delete_pairs = []
        for pfidx, pickup_node in enumerate(route_from[1:], start=1):
            if prob.is_ppick(pickup_node):
                qfidx = pfidx + 1
                delete_pairs.append((pfidx, qfidx, "serveP"))

            elif prob.is_lpick(pickup_node):
                jid = prob.rev_lpick(pickup_node)
                drop_node = prob.ldrop(jid)
                qfidx = pos_from.get(drop_node)

                if qfidx is not None and qfidx > pfidx:
                    delete_pairs.append((pfidx, qfidx, "serveL"))

        # Enumerate feasible insertion (pickup, drop) index pairs in route_to
        insert_pairs_pserve = [
            (ptidx, ptidx + 1)
            for ptidx in range(1, n_to)
            if not prob.is_ppick(route_to[ptidx - 1])  # Cannot insert after a pickup
        ]
        insert_pairs_lserve = [
            (ptidx, qtidx)
            for ptidx in range(1, n_to)
            if not prob.is_ppick(route_to[ptidx - 1])
            for qtidx in range(ptidx + 1, n_to + 1)
            if not prob.is_ppick(route_to[qtidx - 2])
        ]

        # Iterate over all delete pairs in route_from
        for (pfidx, qfidx, kind) in delete_pairs:
            insert_pairs = insert_pairs_pserve if kind == "serveP" else insert_pairs_lserve
            for (ptidx, qtidx) in insert_pairs:
                request = (pfidx, qfidx, ptidx, qtidx, kind)
                costchange = check_relocate(request)

                # If costchange is None, relocation is infeasible
                if costchange is None:
                    continue

                # Check uplift requirement
                after_cost_a, after_cost_b, dec = costchange
                if dec < uplift:
                    continue

                if mode == 'first':
                    yield (request, costchange)
                    return
                else:
                    yield (request, costchange)


    def select_candidate() -> Optional[Tuple[Request, CostChange]]:
        """
        Select a candidate relocation based on the specified mode.
        """
        cand_list = list(find_candidates())
        if not cand_list:
            return None
        if mode == 'stochastic':
            return rng.choice(cand_list)
        elif mode == 'best':
            # Choose by maximum decrement (index 2 of CostChange)
            return max(cand_list, key=lambda x: x[1][2])
        else:
            return cand_list[0]


    # //// Update helpers
    def update_partial_solution(action: Action):
        """
        Apply relocation to routes and update costs / max cost for the
        current partial solution object.
        """
        nonlocal route_from, route_to, current_par

        # Unpack action
        (p_from, q_from, p_to, q_to, _), (new_cost_from, new_cost_to, dec) = action

        # Extract nodes to move (pickup & drop) from route_from
        pf = route_from[p_from]
        qf = route_from[q_from]


        # 1. Remove from route_from (remember to remove drop first)
        del route_from[q_from]
        del route_from[p_from]

        # 2. Insert into route_to (remember to insert pickup first)
        route_to.insert(p_to, pf)
        route_to.insert(q_to, qf)

        # 3. Update partial routes
        current_par.routes[route_from_idx] = route_from
        current_par.routes[route_to_idx] = route_to

        # 4. Update partial costs
        current_par.route_costs[route_from_idx] = new_cost_from
        current_par.route_costs[route_to_idx] = new_cost_to
        current_par.max_cost -= dec

        # 5. Update node assignment
        current_par.node_assignment[pf] = route_to_idx
        current_par.node_assignment[qf] = route_to_idx


    def update_precalc(action: Action):
        """
        Incrementally update passenger & load delta managers after a relocation
        using MinMaxPfsumArray insert/delete operations (avoid full rebuild).

        action: (p_from, q_from, p_to, q_to, new_cost_from, new_cost_to, dec)
        Indices p_from,q_from,p_to,q_to refer to ORIGINAL pre-mutation routes.
        Relocation sequence applied earlier in update_partial_solution:
            1. Remove q_from, then p_from from donor route_from.
            2. Insert pickup at p_to in route_to.
            3. Insert drop at drop_insert_index = (q_to if q_to was final depot else q_to+1).
        Here we commit those operations on the delta managers.
        """
        nonlocal load_delta_from_manager, load_delta_to_manager, route_from, route_to

        # Unpack action
        (pfidx, qfidx, ptidx, qtidx, _), _costchange = action
        pf = route_from[pfidx]
        qf = route_from[qfidx]

        # Helper to map node -> load
        def node_load_delta(nodeid: int) -> int:
            if prob.is_lpick(nodeid):
                jid = prob.rev_lpick(nodeid)
                return prob.q[jid - 1]
            elif prob.is_ldrop(nodeid):
                jid = prob.rev_ldrop(nodeid)
                return -prob.q[jid - 1]
            else:
                return 0


        # 1. Donor route updates
        load_delta_from_manager.delete(qfidx)
        load_delta_from_manager.delete(pfidx)

        # 2. Receiver route updates
        load_delta_to_manager.insert(ptidx, node_load_delta(pf))
        load_delta_to_manager.insert(qtidx, node_load_delta(qf))


    def relocate_to_convergence() -> Tuple[List[bool], int]:
        """
        Perform relocation steps until no further improvement is possible
        or the specified number of steps is reached.

        Returns a tuple of:
        - modified_routes: List of booleans indicating which routes were modified.
        - reloc_done: Number of relocations performed.
        """
        nonlocal n_from, n_to, route_from, route_to

        reloc_done = 0
        modified_routes = [False] * prob.K
        while reloc_done < steps:
            cand = select_candidate()
            if cand is None:
                break

            # Apply the selected relocation
            update_precalc(cand)
            update_partial_solution(cand)

            # Update counters and flags
            reloc_done += 1
            modified_routes[route_from_idx] = True
            modified_routes[route_to_idx] = True

            # Update local route lengths
            n_from -= 2
            n_to += 2
            if n_from < 5:
                break

            # Verbose logging
            if verbose:
                pf, qf, pt, qt, _ = cand[0]
                _, _, dec = cand[1]
                print(
                    f"[Relocate] [Route {route_from_idx}->{route_to_idx}] Moved request "
                    f"(P:{pf},D:{qf}) to ({pt},{qt}). Decrement={dec}"
                )

        return modified_routes, reloc_done

    # Execute relocation to convergence
    modified_pair, reloc_done = relocate_to_convergence()

    return current_par, modified_pair, reloc_done




def relocate_operator(
        partial: PartialSolution,           # Current partial solution
        steps: Optional[int] = None,        # Number of steps to consider
        mode: str = 'first',                # Mode of operation
        uplift: int = 1,                    # Minimum improvement required
        seed: Optional[int] = None,         # Random seed for stochastic mode
        verbose: bool = False,              # Verbosity flag
    ) -> Tuple[PartialSolution, List[bool], int]:
    """
    Attempt to relocate the request from a vehicle route to another that
    improves the solution to different extents controlled by ``uplift``.
    Perform up to ``steps`` relocations based on the specified ``mode``.
    Use in post-processing or local search.

    The procedure attempts to relocate requests from the highest-cost route
    to the 1/3 lower-cost routes (traverse from the lowest to the highest) by
    iterating over all insertion pairs of the receiver routes, in ascending 
    order of their in-out contribution to the route cost.
    If not successful, it moves to the next donor route.

    Parameters:
    - partial: PartialSolution object representing the current solution.
    - steps: Number of relocation steps that the operation should perform.
    - mode: Mode of operation, can be 'best', 'first', or 'stochastic'.
    - uplift: Integer controlling the extent of improvement required.
    - seed: Random seed for stochastic modes.
    - verbose: If True, print detailed logs.

    Returns: a tuple signature containing:
    - A new PartialSolution object with the specified requests relocated.
    - A list of booleans indicating which routes were modified.
    - An integer count of the number of relocations performed.
    """
    k_vehicles = partial.problem.K
    if k_vehicles < 2:
        return partial.copy(), [False] * k_vehicles, 0

    if steps == None:
        steps = 10**9    # Effectively unlimited

    # RNG for stochastic behavior
    rng = random.Random(seed)

    # Initialize tracking variables
    current_par: PartialSolution = partial.copy()
    modified_total: List[bool] = [False] * k_vehicles
    total_moves = 0


    # //// Main relocation loop
    while total_moves < steps:
        # Sort candidate donor and receiver routes ascending by current costs
        taxi_cost: List[Tuple[int, int]] = list(enumerate(current_par.route_costs))
        donor_index = max(taxi_cost, key=lambda x: x[1])[0]
        receiver_indices = [
            idx for idx, _ in sorted(taxi_cost, key=lambda x: x[1])
        ]

        # Break if the donor is too short
        if len(current_par.routes[donor_index]) < 5:
            break

        # Iterate over receiver routes
        improved = False
        for r_idx in receiver_indices:
            # Skip self-relocation
            if r_idx == donor_index:
                continue

            # Skip too-short receivers (no place to insert between depots)
            if len(current_par.routes[r_idx]) < 2:
                continue


            # Attempt relocation from donor to receiver
            remain = steps - total_moves
            new_partial, modified_pair, moves_made = relocate_from_to(
                current_par,
                route_from_idx=donor_index,
                route_to_idx=r_idx,
                steps=remain,
                mode=mode,
                uplift=uplift,
                seed=rng.randint(10, 10**9),  # vary seed between attempts
                verbose=verbose,
            )


            # Analyze results
            if moves_made > 0:
                current_par = new_partial
                total_moves += moves_made
                for i in range(k_vehicles):
                    if modified_pair[i]:
                        modified_total[i] = True
                improved = True

                # Verbose logging
                if verbose:
                    print(f"{moves_made} relocation made from route {donor_index} to route {r_idx}")

                break   # break receivers loop, re-sort donors/receivers

        # Exit if no improvement found in this iteration (convergence)
        if not improved:
            break

    # Logging
    if verbose:
        print()
        print("[Relocate] Operator completed. ")
        print(f"Total relocations = {total_moves}; ")
        print(f"Decrement = {partial.max_cost - current_par.max_cost}; ")
        print(f"New max cost = {current_par.max_cost}.")
        print("------------------------------")
        print()

    return current_par, modified_total, total_moves




# ================== Playground ==================
if __name__ == "__main__":
    from share_a_ride.solvers.algo.utils import test_problem
    from share_a_ride.solvers.algo.greedy import greedy_solver

    # Run relocate operator
    solution, info = greedy_solver(test_problem, verbose=False)
    assert solution
    par = PartialSolution.from_solution(solution)

    new_par, modified_routes, n_relocs = relocate_operator(
        par,
        steps=5,
        mode='best',
        uplift=1,
        seed=42,
        verbose=True
    )

    new_solution = new_par.to_solution()
    assert new_solution
    print(f"Original cost: {solution.max_cost}, New cost: {new_solution.max_cost}")
