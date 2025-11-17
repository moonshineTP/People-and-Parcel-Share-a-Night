"""
Relocate operator for Share-a-Ride problem.
Relocates a request from one vehicle route to another.
"""
import random
from typing import Iterator, List, Tuple, Optional
from share_a_ride.core.solution import PartialSolution
from share_a_ride.solvers.operator.utils import MinMaxPfsumArray


def cost_decrement_relocate(
        partial: PartialSolution,
        from_route_idx: int,
        to_route_idx: int,
        p_idx_from: int,
        q_idx_from: int,
        p_idx_to: int,
        q_idx_to: int
    ) -> tuple[int, int, int]:
    """
    Compute the decrement in max_cost of relocating a full request (pickup and drop)
    from one route to another at specified insertion indices, as Cost_before - Cost_after.
    This assumes the pickup/drop indices are correct with some defensive assertions.

    Parameters:
    - partial: PartialSolution object representing the current solution.
    - from_route_idx: Index of the route from which the request is relocated.
    - to_route_idx: Index of the route to which the request is relocated.
    - p_idx_from: Index of the pickup node in the from_route.
    - q_idx_from: Index of the drop node in the from_route.
    - p_idx_to: Index in the to_route where the pickup node will be inserted.
    - q_idx_to: Index in the to_route where the drop node will be inserted.

    Returns:
    - A tuple containing:
        - from_route_next_cost: Cost of the from_route after relocation.
        - to_route_next_cost: Cost of the to_route after relocation.
        - cost_decrement: Decrement in max_cost due to the relocation.
    """
    from_route = partial.routes[from_route_idx]
    to_route = partial.routes[to_route_idx]

    # Basic assertions (should not move depot 0)
    assert from_route_idx != to_route_idx, \
        "from_route_idx and to_route_idx must be different for relocate."
    assert from_route[p_idx_from] != 0 and from_route[q_idx_from] != 0, \
        "Cannot relocate depot nodes."
    assert 1 <= p_idx_from < q_idx_from, "Invalid pickup/drop indices in from_route."
    assert 1 <= p_idx_to < q_idx_to, "Invalid pickup/drop indices in to_route"

    # Extract problem attributes
    D = partial.problem.D
    cur_cost = partial.max_cost

    # Identify the moving nodes
    p_from = from_route[p_idx_from]
    q_from = from_route[q_idx_from]


    # ---------- Cost change for from_route: remove both pickup and drop ----------
    # Stitch neighbors for deletion
    if p_idx_from + 1 == q_idx_from:    # Adjacent: remove p_from and q_from together
        prev_p_from = from_route[p_idx_from - 1]
        next_q_from = from_route[q_idx_from + 1]
        in_out_from_before = (
            D[prev_p_from][p_from] + D[p_from][q_from] + D[q_from][next_q_from]
        )
        in_out_from_after = (
            D[prev_p_from][next_q_from]
        )
    else:   # Non-adjacent: remove p_from and d_from separately
        prev_p_from = from_route[p_idx_from - 1]
        next_p_from = from_route[p_idx_from + 1]
        prev_q_from = from_route[q_idx_from - 1]
        next_q_from = from_route[q_idx_from + 1]
        in_out_from_before = (
            D[prev_p_from][p_from] + D[p_from][next_p_from]
            + D[prev_q_from][q_from] + D[q_from][next_q_from]
        )
        in_out_from_after = (
            D[prev_p_from][next_p_from] + D[prev_q_from][next_q_from]
        )

    from_route_next_cost = (
        partial.route_costs[from_route_idx]
        - in_out_from_before + in_out_from_after
    )


    # ---------- Cost change for to_route: insert pickup then drop ----------
    # Stitch neighbors for insertion
    if q_idx_to == p_idx_to + 1:
        prev_p_from = to_route[p_idx_to - 1]
        next_q_to = to_route[q_idx_to - 1]
        in_out_to_before = (
            D[prev_p_from][next_q_to]
        )
        in_out_to_after = (
            D[prev_p_from][p_from] + D[p_from][q_from] + D[q_from][next_q_to]
        )
    else:
        prev_p_from = to_route[p_idx_to - 1]
        next_p_from = to_route[p_idx_to]
        prev_q_to = to_route[q_idx_to - 2]
        next_q_to = to_route[q_idx_to - 1]
        in_out_to_before = (
            D[prev_p_from][next_p_from] + D[prev_q_to][next_q_to]
        )
        in_out_to_after = (
            D[prev_p_from][p_from] + D[p_from][next_p_from]
            + D[prev_q_to][q_from] + D[q_from][next_q_to]
        )

    # Update to_route cost
    to_route_next_cost = (
        partial.route_costs[to_route_idx]
        + in_out_to_after - in_out_to_before
    )

    # Compute the next max cost across all routes
    next_cost = max(
        from_route_next_cost,
        to_route_next_cost,
        *(
            partial.route_costs[i]
            for i in range(partial.problem.K)
            if i != from_route_idx and i != to_route_idx
        )
    )

    # Return the computed costs and decrement
    return from_route_next_cost, to_route_next_cost, cur_cost - next_cost


def relocate_from_to(
        partial: PartialSolution,   # Partial solution object
        from_route_idx: int,        # Donor route id
        to_route_idx: int,          # Receiver route id
        steps: int,                 # Number of steps to consider
        mode: str,                  # Mode of operation
        uplift: int = 1,            # Minimum improvement required
        seed: int = 42,             # Seed for reproducibility
        verbose: bool = False       # Verbose output flag
    ) -> Tuple[PartialSolution, list[bool], int]:
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

    # Local aliases
    prob = partial.problem
    current_par = partial.copy()
    # Work on the copied partial to avoid mutating the caller's object by aliasing
    route_from = current_par.routes[from_route_idx]
    route_to = current_par.routes[to_route_idx]
    n_from = len(route_from)
    n_to = len(route_to)

    if n_from < 5:
        return partial, [False] * prob.K, 0    # No requests to relocate


    # Build passenger and parcel deltas
    def build_segment_deltas(route: list[int], n: int):
        pass_delta = [0] * n
        parc_delta = [0] * n
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
            pass_delta[i] = dp
            parc_delta[i] = dq

        pass_delta_manager = MinMaxPfsumArray(pass_delta)
        parc_delta_manager = MinMaxPfsumArray(parc_delta)

        return pass_delta_manager, parc_delta_manager


    # Build deltas managers for both routes
    pass_delta_from, parc_delta_from = build_segment_deltas(route_from, n_from)
    pass_delta_to, parc_delta_to = build_segment_deltas(route_to, n_to)

    # Capacity caps
    cap_from = prob.Q[from_route_idx]
    cap_to = prob.Q[to_route_idx]


    # ------------- Check and selection helpers -------------
    def check_passenger(p_idx_a: int, q_idx_a: int, p_idx_b: int, q_idx_b: int) -> bool:
        """
        Ensure passenger load stays within [0,1] after relocating a full request
        (p_idx_a, q_idx_a) to (p_idx_b, q_idx_b) for both routes.
        """
        # Passenger pickup delta: only passenger pickup nodes contribute +1.
        # Parcel pickups do not affect passenger load.
        node_from_pick = route_from[p_idx_a]
        d_pass = 1 if prob.is_ppick(node_from_pick) else 0

        # If the moved request is not a passenger request, no passenger load shift occurs.
        if d_pass == 0:
            return True

        # Removing the pair shifts donor interval [p_idx_a, q_idx_a) by -d_pass.
        min_from = pass_delta_from.query_min_prefix(p_idx_a, q_idx_a)
        max_from = pass_delta_from.query_max_prefix(p_idx_a, q_idx_a)
        if min_from - d_pass < 0 or max_from - d_pass > 1:
            return False

        # Insertion applies +d_pass over [p_idx_b, q_idx_b - 1) on receiver.
        min_to = pass_delta_to.query_min_prefix(p_idx_b - 1, q_idx_b - 1)
        max_to = pass_delta_to.query_max_prefix(p_idx_b - 1, q_idx_b - 1)
        if min_to + d_pass < 0 or max_to + d_pass > 1:
            return False

        return True


    def check_parcel(p_idx_a: int, q_idx_a: int, p_idx_b: int, q_idx_b: int) -> bool:
        """
        Ensure parcel load stays within [0,cap] after relocating a full request
        (p_idx_a, q_idx_a) to (p_idx_b, q_idx_b) for both routes.
        """
        # Parcel pickup delta: quantity q_j if parcel pickup else 0.
        node_from_pick = route_from[p_idx_a]
        if prob.is_parc_pick(node_from_pick):
            jid = prob.rev_parc_pick(node_from_pick)
            d_parc = prob.q[jid - 1]
        else:
            d_parc = 0

        # If not a parcel request, no parcel capacity shift occurs.
        if d_parc == 0:
            return True

        # Removing pair shifts donor interval [p_idx_a, q_idx_a) by -d_parc.
        min_from = parc_delta_from.query_min_prefix(p_idx_a, q_idx_a)
        max_from = parc_delta_from.query_max_prefix(p_idx_a, q_idx_a)
        if min_from - d_parc < 0 or max_from - d_parc > cap_from:
            return False

        # Insertion applies +d_parc over [p_idx_b, q_idx_b) on receiver.
        min_to = parc_delta_to.query_min_prefix(p_idx_b - 1, q_idx_b - 1)
        max_to = parc_delta_to.query_max_prefix(p_idx_b - 1, q_idx_b - 1)
        if min_to + d_parc < 0 or max_to + d_parc > cap_to:
            return False

        return True


    def check_relocate(
            p_idx_a: int, q_idx_a: int, p_idx_b: int, q_idx_b: int
        ) -> tuple[bool, int, int, int]:
        """
        Check feasibility of relocating the request defined by (p_idx_a, q_idx_a)
        from route_from to route_to at insertion indices (p_idx_b, q_idx_b).
        
        Returns a tuple of:
            - feasibility (bool)
            - after_cost_a (int): cost of from_route after relocation
            - after_cost_b (int): cost of to_route after relocation
            - dec (int): total cost decrement if relocation is performed
        """
        if not check_passenger(p_idx_a, q_idx_a, p_idx_b, q_idx_b):
            return False, 0, 0, 0
        if not check_parcel(p_idx_a, q_idx_a, p_idx_b, q_idx_b):
            return False, 0, 0, 0

        # Compute total decrement for swapping both pickup and drop pairs simultaneously
        after_cost_a, after_cost_b, dec = cost_decrement_relocate(
            current_par,
            from_route_idx, to_route_idx,
            p_idx_a, q_idx_a,
            p_idx_b, q_idx_b,
        )
        return True, after_cost_a, after_cost_b, dec


    def find_candidates() -> Iterator[tuple[int, int, int, int, int, int, int]]:
        """
        Find candidate relocation indices (p_idx_from, d_idx_from, p_idx_to, d_idx_to)
        according to the specified mode ('best', 'first', 'stochastic').
        
        Yields tuples of the form:
            (p_idx_from, d_idx_from, p_idx_to, d_idx_to, route_from_next_cost,
             route_to_next_cost, max_cost_decrement)
        """
        # Build position map for route_from to locate corresponding drops
        pos_from = {node: i for i, node in enumerate(route_from)}

        # Enumerate all pickup positions in route_from
        pickup_indices_from = [
            i for i in range(1, n_from - 1)
            if prob.is_ppick(route_from[i]) or prob.is_parc_pick(route_from[i])
        ]

        # Enumerate feasible insertion (pickup, drop) index pairs in route_to
        insertion_pairs_to = [
            (p_to, q_to)
            for p_to in range(1, n_to)
            for q_to in range(p_to + 1, n_to + 1)
        ]

        for p_idx_a in pickup_indices_from:
            # Locate corresponding drop in route_from
            node_a = route_from[p_idx_a]
            if prob.is_ppick(node_a):
                pid = prob.rev_ppick(node_a)
                q_node_a = prob.pdrop(pid)
            else:
                jid = prob.rev_parc_pick(node_a)
                q_node_a = prob.parc_drop(jid)
            q_idx_a = pos_from.get(q_node_a)
            if q_idx_a is None or q_idx_a <= p_idx_a:
                continue

            # Try all insertion pairs in route_to
            for p_idx_b, q_idx_b in insertion_pairs_to:
                feasible, after_cost_a, after_cost_b, dec = check_relocate(
                    p_idx_a, q_idx_a, p_idx_b, q_idx_b
                )
                if not feasible or dec < uplift:
                    continue

                if mode == 'first':
                    yield (p_idx_a, q_idx_a, p_idx_b, q_idx_b, after_cost_a, after_cost_b, dec)
                    return
                else:
                    yield (p_idx_a, q_idx_a, p_idx_b, q_idx_b, after_cost_a, after_cost_b, dec)


    def select_candidate() -> Optional[tuple[int, int, int, int, int, int, int]]:
        """
        Select a candidate relocation based on the specified mode.
        """
        cand_list = list(find_candidates())
        if not cand_list:
            return None
        if mode == 'stochastic':
            return rng.choice(cand_list)
        elif mode == 'best':
            # Choose by maximum decrement (index 6)
            return max(cand_list, key=lambda x: x[6])
        else:
            return cand_list[0]


    # ------------- Update helpers -------------
    def update_partial_solution(action: tuple[int, int, int, int, int, int, int]):
        """ 
        Apply relocation to routes and update costs / max cost for the
        current partial solution object.
        """
        (p_from, q_from, p_to, q_to, new_cost_from, new_cost_to, dec) = action
        nonlocal route_from, route_to, current_par

        # Extract nodes to move (pickup & drop) from route_from
        node_p = route_from[p_from]
        node_q = route_from[q_from]

        # Remove in reverse order to keep indices stable
        del route_from[q_from]
        del route_from[p_from]

        # Insert into route_to
        route_to.insert(p_to, node_p)
        route_to.insert(q_to, node_q)

        # Update stored routes in current_par
        current_par.routes[from_route_idx] = route_from
        current_par.routes[to_route_idx] = route_to

        # Update route costs from precomputed candidate values
        current_par.route_costs[from_route_idx] = new_cost_from
        current_par.route_costs[to_route_idx] = new_cost_to
        current_par.max_cost -= dec

        return


    def update_precalc(action: tuple[int, int, int, int, int, int, int]):
        """
        Incrementally update passenger & parcel delta managers after a relocation
        using MinMaxPfsumArray insert/delete operations (avoid full rebuild).

        action: (p_from, q_from, p_to, q_to, new_cost_from, new_cost_to, dec)
        Indices p_from,q_from,p_to,q_to refer to ORIGINAL pre-mutation routes.
        Relocation sequence applied earlier in update_partial_solution:
            1. Remove q_from, then p_from from donor route_from.
            2. Insert pickup at p_to in route_to.
            3. Insert drop at drop_insert_index = (q_to if q_to was final depot else q_to+1).
        Here we commit those operations on the delta managers.
        """
        nonlocal pass_delta_from, parc_delta_from, pass_delta_to, parc_delta_to
        nonlocal route_from, route_to

        p_from, q_from, p_to, q_to, *_ = action

        # Helper to map node -> (passenger_delta, parcel_delta)
        def node_deltas(node: int) -> tuple[int, int]:
            if prob.is_ppick(node):
                return 1, 0
            if prob.is_pdrop(node):
                return -1, 0
            if prob.is_parc_pick(node):
                jid = prob.rev_parc_pick(node)
                return 0, prob.q[jid - 1]
            if prob.is_parc_drop(node):
                jid = prob.rev_parc_drop(node)
                return 0, -prob.q[jid - 1]
            return 0, 0

        # ---------- Donor route delta manager updates ----------
        # Delete in the reverse order (drop then pickup) to keep indices stable
        pass_delta_from.delete(q_from)
        parc_delta_from.delete(q_from)
        pass_delta_from.delete(p_from)
        parc_delta_from.delete(p_from)


        # ---------- Receiver route delta manager updates ----------
        pass_delta_to.insert(p_to, node_deltas(route_from[p_from])[0])
        parc_delta_to.insert(p_to, node_deltas(route_from[p_from])[1])
        pass_delta_to.insert(q_to, node_deltas(route_from[q_from])[0])
        parc_delta_to.insert(q_to, node_deltas(route_from[q_from])[1])

        return


    def relocate_to_convergence() -> tuple[list[bool], int]:
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
            modified_routes[from_route_idx] = True
            modified_routes[to_route_idx] = True

            # Update local route lengths
            n_from -= 2
            n_to += 2
            if n_from < 5:
                break

            # Verbose logging
            if verbose:
                pf, qf, pt, qt, _, __, dec = cand
                print(f"[Relocate {from_route_idx}->{to_route_idx}] moved request "
                      f"(P:{pf},D:{qf}) to ({pt},{qt}) dec={dec}"
                )

            # Stop if in 'first' mode
            if mode == 'first':
                break

        return modified_routes, reloc_done


    # Execute relocation to convergence
    modified_pair, reloc_done = relocate_to_convergence()
    return current_par, modified_pair, reloc_done



def relocate_operator(
        partial: PartialSolution,           # Current partial solution
        steps: Optional[int] = None,        # Number of steps to consider
        mode: str = 'first',                # Mode of operation
        uplift: int = 1,                    # Minimum improvement required
        seed: int = 42,                     # Random seed for stochastic mode
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

    Returns:
    - A tuple signature containing:
        - A new PartialSolution object with the specified requests relocated.
        - A list of booleans indicating which routes were modified.
        - An integer count of the number of relocations performed.
    """
    K = partial.problem.K
    if K < 2:
        return partial.copy(), [False] * K, 0

    # Treat None as effectively large
    max_steps = steps if steps is not None else 10**9

    # RNG for stochastic behavior
    rng = random.Random(seed)

    # Initialize current partial solution and tracking variables
    current_par: PartialSolution = partial.copy()
    modified_total: List[bool] = [False] * K
    total_moves = 0


    # Greedy outer loop: attempt donors from highest-cost to lowest-cost,
    # receivers from lowest-cost upwards, as per procedure description.
    while total_moves < max_steps:
        # Sort candidate donor and receiver routes by current costs
        # Note: we only pick top 5 from each to limit attempts
        costs: list[tuple[int, int]] = list(enumerate(current_par.route_costs))
        donor_idx = max(costs, key=lambda x: x[1])[0]
        receivers = [idx for idx, _ in sorted(costs, key=lambda x: x[1])][:max(4, K // 2)]

        # Break if the donor is too short
        # Note that if it is too short, all other routes must be too short as well
        if len(current_par.routes[donor_idx]) < 5:
            break


        # Attempt relocations from donor donor_idx to each receiver r_idx
        improved = False
        for r_idx in receivers:
            if r_idx == donor_idx:      # Skip same route
                continue
            # Skip too-short receivers (no place to insert between depots)
            if len(current_par.routes[r_idx]) < 2:
                continue

            # Attempt relocation from donor_idx to r_idx
            remain = max_steps - total_moves
            new_partial, modified_pair, moves_made = relocate_from_to(
                current_par,
                from_route_idx=donor_idx,
                to_route_idx=r_idx,
                steps=remain,
                mode=mode,
                uplift=uplift,
                seed=rng.randint(10, 10**9),  # vary seed between attempts
                verbose=verbose,
            )

            # If successful, update current solution and tracking variables
            if moves_made > 0:
                current_par = new_partial
                total_moves += moves_made
                for i in range(K):
                    if modified_pair[i]:
                        modified_total[i] = True
                improved = True

                # Verbose logging
                if verbose:
                    print(f"{moves_made} relocation made from route {donor_idx} to route {r_idx}")

                break   # break receivers loop, re-sort donors/receivers

        if not improved:
            break   # No improvement found, exit main loop

    return current_par, modified_total, total_moves



if __name__ == "__main__":
    import time
    from share_a_ride.core.utils.generator import generate_instance_coords
    from share_a_ride.solvers.algo.greedy import greedy_balanced_solver

    # Simple test case
    problem = generate_instance_coords(
        N=200, M=300, K=7, area=1000, seed=12345
    )
    # dataset = "H"
    # instance_name = "H-n30-m25-k10"
    # path = path_router(dataset, "readfile", instance_name)
    # problem = parse_sarp_to_problem(path)

    sol, msg = greedy_balanced_solver(problem)
    assert sol
    # print("Solution before relocation:")
    # sol.stdin_print(True)
    # print()

    st = time.time()
    par = PartialSolution.from_solution(sol)
    next_par, modified, n_moves = relocate_operator(
        par,
        steps=None,
        mode='first',
        seed=100,
        verbose=True
    )

    sol_after = next_par.to_solution()
    assert sol_after
    # print("Solution after relocation:")
    # sol_after.stdin_print(True)
    # print()

    print()
    print(f"Relocate operator performed {n_moves} moves, modified routes: {modified}")
    print(f"Solution cost before: {sol.max_cost}, after: {sol_after.max_cost}")
    print(f"Relocate operator time: {time.time() - st:.4f} seconds" )
