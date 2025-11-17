"""
Beam search module for the Share-a-Ride problem.
"""
import cProfile
import math
import random
import time
from typing import Any, List, Optional, Tuple, Dict

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, Solution, PartialSolutionSwarm
from share_a_ride.solvers.operator.swap import (
    intra_swap_operator,
    inter_swap_operator,
)


def parsol_scorer(
        parsol: PartialSolution,
        sample_size: int = 15,
        w_std: float = 0.15,
        seed: int = 42,
    ) -> float:
    """
    Score a PartialSolution using a stochastic surrogate of the route-cost spread.

    Route costs are sampled with replacement (default samples=15) to estimate the
    standard deviation, and the final score is ``max_cost + std_weight * std``.
    """
    rng = random.Random(seed)
    effective_size = max(1, sample_size)
    costs = parsol.route_costs
    if len(costs) == 1:
        return parsol.max_cost

    sampled = rng.choices(costs, k=effective_size)
    mean = math.fsum(sampled) / len(sampled)
    variance = math.fsum((value - mean) ** 2 for value in sampled) / len(sampled)
    std_dev = math.sqrt(max(0.0, variance))

    return parsol.max_cost + w_std * std_dev



def beam_search_swarm_solver(
        problem: ShareARideProblem,
        cost_function: Any = parsol_scorer,
        initial: Optional[PartialSolutionSwarm] = None,
        l_width: int = 10,
        r_intra: float = 0.75,
        r_inter: float = 0.90,
        f_intra: float = 0.05,
        f_inter: float = 0.10,
        verbose: bool = False
    ) -> Tuple[PartialSolutionSwarm, Dict[str, Any]]:
    """
    Beam search solver using a swarm of partial solutions.

    This should work as follows:
    - At each depth step, maintains a beam of the best partial solutions with the
    number of actions equal to the depth. Note that the first few steps may have
    less than beam_width solutions if not enough distinct partial solutions exist.
    - In the expansion phase, expands each partial solution by evaluating every
    possible next actions / next states and filtering to the best (step + 1) ones. 
    - Selects the top N partial solutions based on a cost function.

    Returns:
    (sol, info): tuple where
    - sol: Solution object with routes and costs.
    - Info dictionary contains:
        + iterations: number of main loop iterations
        + time: total time taken
    The solver also injects intra-route swaps in the last ``r_intra`` portion of
    the action horizon and inter-route swaps in the last ``r_inter`` portion, and
    it repeats these operators every ``f_intra`` / ``f_inter`` fraction of the
    horizon respectively, enabling adaptive late-stage refinement without
    dominating runtime.
    """
    # Initialize timing and parameters.
    start = time.time()
    total_actions = max(1, 2 * (problem.N + problem.M) + problem.K)

    # Clamp ratio and frequency parameters to [0, 1] range.
    def _clamp(value: float) -> float:
        return min(max(value, 0.0), 1.0)
    r_intra_clamped = _clamp(r_intra)
    r_inter_clamped = _clamp(r_inter)
    f_intra_clamped = _clamp(f_intra)
    f_inter_clamped = _clamp(f_inter)

    # Compute phase depths and frequencies.
    until_intra_depth = max(0, int(total_actions * r_intra_clamped))
    until_inter_depth = max(0, int(total_actions * r_inter_clamped))
    intra_frequency = max(1, int(total_actions * f_intra_clamped))
    inter_frequency = max(1, int(total_actions * f_inter_clamped))


    # Helper: expand one PartialSolution into next-depth candidates.
    def expand(parsol: PartialSolution) -> List[PartialSolution]:
        candidates: List[PartialSolution] = []

        # Pick the taxis with the smallest accumulated cost, mirroring greedy balancing.
        available_taxis = [
            idx for idx, state in enumerate(parsol.route_states)
            if not state["ended"]
        ]
        if not available_taxis:
            return candidates

        # Keep a small slice (2-3) of the least-cost taxis.
        taxi_order = sorted(
            available_taxis,
            key=lambda idx: parsol.route_states[idx]["cost"]
        )
        taxi_considered = min(
            2 if problem.K >= 50
            else 3 if problem.K >= 20
            else 4,
            len(taxi_order)
        )
        taxi_branches = taxi_order[:taxi_considered]

        # Determine closing depth to allow return-to-depot actions.
        # Allow some windows to close routes a little bit earlier.
        closing_depth = max(0, total_actions - 2 * parsol.problem.K)

        # Expand each selected taxi branch.
        for t_idx in taxi_branches:
            state = parsol.route_states[t_idx]
            actions = parsol.possible_actions(t_idx)

            # If no load and feasible to end route, add return-to-depot as an action branch.
            can_return = (
                state["passenger"] == 0 and
                not state["parcels"] and
                not state["ended"] and
                state["pos"] != 0
            )

            # Expand all feasible actions for this taxi.
            if actions:
                # Keep a few (2-3) best incremental moves per taxi.
                action_limit = min(
                    1 if problem.num_nodes >= 500   # Basically greedy with l branches
                    else 2 if problem.num_nodes >= 200  # Has more branching
                    else 4, # More branching for smaller problems
                    len(actions)
                )
                for kind, node_idx, inc in sorted(actions, key=lambda item: item[2])[:action_limit]:
                    parsol.apply_action(t_idx, kind, node_idx, inc)
                    candidates.append(parsol.copy())
                    parsol.reverse_action(t_idx)

                if can_return and depth >= closing_depth:
                    parsol.apply_return_to_depot(t_idx)
                    candidates.append(parsol.copy())
                    parsol.reverse_action(t_idx)

            # No feasible actions: only consider return to depot if possible.
            elif can_return:
                parsol.apply_return_to_depot(t_idx)
                candidates.append(parsol.copy())
                parsol.reverse_action(t_idx)

            # Else, no actions possible for this taxi at this time.
            # Print a warning if verbose.
            else:
                if verbose:
                    print(
                        f"[BeamSearch] Taxi {t_idx} has no feasible actions "
                        f"and cannot return to depot."
                    )

        # Sort candidates by cost function and return.
        candidates.sort(key=cost_function)
        return candidates


    # Helper: apply local refinements to a list of partial solutions.
    def apply_local_refinements(
            parsols: List[PartialSolution],
            use_intra: bool,
            use_inter: bool,
            seed_offset: int
        ) -> List[PartialSolution]:
        if not (use_intra or use_inter):
            return parsols

        # Apply refinements to each partial solution.
        refined: List[PartialSolution] = []
        for idx, base in enumerate(parsols):
            updated = base
            if use_intra:
                updated, _, _ = intra_swap_operator(
                    updated,
                    steps=None,
                    mode='first',
                    uplift=1,
                    seed=1337 + seed_offset + idx,
                    verbose=False
                )
            if use_inter:
                updated, _, _ = inter_swap_operator(
                    updated,
                    steps=None,
                    mode='first',
                    uplift=1,
                    seed=2671 + seed_offset + idx,
                    verbose=False
                )
            refined.append(updated)

        # Return refined partial solutions.
        return refined


    # Initialize beam with a single empty partial solution.
    if initial is None:
        initial = PartialSolutionSwarm(
            solutions=[PartialSolution(problem=problem, routes=[])]
        )
    beam = initial.parsol_list
    depth = initial.parsol_list[0].n_actions  # should be 0 initially
    iterations = 0

    assert all(ps.n_actions == depth for ps in beam), \
        "All initial partial solutions must have the same action count."


    # //////// Main loop ////////
    while beam:
        # Termination if beam consists only of complete solutions.
        if all(ps.is_complete() for ps in beam):
            break

        # Update iterations and target depth.
        iterations += 1

        # Check if intra / inter phases should be applied.
        use_intra_phase = (
            intra_frequency is not None and
            depth >= until_intra_depth and
            (depth - until_intra_depth) % intra_frequency == 0
        )
        use_inter_phase = (
            inter_frequency is not None and
            depth >= until_inter_depth and
            (depth - until_inter_depth) % inter_frequency == 0
        )

        # Apply local refinements if the turn comes.
        if use_intra_phase or use_inter_phase:
            beam = apply_local_refinements(
                beam,
                use_intra_phase,
                use_inter_phase,
                seed_offset=depth
            )

        # Basic setup for the next depth.
        diversity_relaxed = False
        beam.sort(key=cost_function)
        next_beam: List[Tuple[float, PartialSolution]] = []

        # Helper: insert candidate into next beam for insertion sort.
        # This also checks if the candidates exists or not.
        def _insert_candidate(cost: float, parsol: PartialSolution) -> None:
            """Insert candidate in ascending cost order, keeping at most l_width entries."""
            if any(parsol.is_identical(other) for _, other in next_beam):
                return

            insert_idx = len(next_beam)
            while insert_idx > 0 and cost < next_beam[insert_idx - 1][0]:
                insert_idx -= 1
            next_beam.insert(insert_idx, (cost, parsol))
            if len(next_beam) > l_width:
                next_beam.pop()


        # //// Iterate through partial solutions in the beam.
        for ps in beam:
            if ps.n_actions != depth:
                continue

            # //// Iterate through expanded candidates.
            for cand in expand(ps)[:min(5, l_width)]:
                candidate_cost = cost_function(cand)
                _insert_candidate(candidate_cost, cand)


        # If no candidates survived, terminate to highlight the stall.
        if not next_beam:
            raise RuntimeError("Beam search stalled: no candidates generated.")

        # Finalize next beam.
        beam = [item[1] for item in next_beam]
        depth += 1

        # //////// Holder for validity, prune later /////////
        # Check for validity of new beam.
        # if iterations > 20:
        #     assert len(beam) == l_width
        # for parsol in beam:
        #     assert parsol.is_valid(), "Generated invalid partial solution in beam expansion."
        #     print(parsol.route_costs)
        #     print(parsol.routes)

        # Verbose logging.
        if verbose:
            if diversity_relaxed:
                print(
                    f"[BeamSearch] Depth {depth}. Diversity relaxed due to empty beam."
                )
            max_costs = [ps.max_cost for ps in beam]
            print(
                f"[BeamSearch] Depth {depth}. Max_cost range: "
                f"{min(max_costs)} - {max(max_costs)}. "
                f"Avg max_cost: {sum(max_costs) / len(max_costs):.1f}"
            )

    # Sort final beam and return best solution swarm.
    beam.sort(key=cost_function)
    swarm = PartialSolutionSwarm(solutions=beam)
    search_info = {
        "iterations": iterations,
        "time": time.time() - start,
    }

    # Summary logging.
    if verbose:
        print(f"[BeamSearch] Completed. Final beam size {len(beam)}")
        print(f"[BeamSearch] Beam max_cost range: {swarm.min_cost} - {swarm.max_cost}")
        print(f"[BeamSearch] Avg max_cost: {swarm.avg_cost}")
        print(f"[BeamSearch] Time taken: {search_info['time']:.4f} seconds")

    return swarm, search_info



def beam_search_solver(
        problem: ShareARideProblem,
        cost_function: Any = parsol_scorer,
        initial: Optional[PartialSolutionSwarm] = None,
        l_width: int = 10,
        r_intra: float = 0.75,
        r_inter: float = 0.90,
        f_intra: float = 0.05,
        f_inter: float = 0.10,
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Beam search solver for the Share-a-Ride problem. This uses a beam search
    approach to explore a swarm of partial solutions, maintaining a balance
    between solution quality and diversity.

    The solver use the beam_search_swarm_solver to generate a swarm of partial
    solutions, then selects the best complete solution based on the ultimate
    max_cost minimization goal.

    Returns:
    (sol, info): tuple where
    - sol: Solution object with routes and costs.
    - Info dictionary contains:
        + iterations: number of main loop iterations
        + time: total time taken
    - r_intra / r_inter control when intra/inter operators start (ratio of depth)
    - f_intra / f_inter control operator repetition frequency as depth fractions
    """
    solswarm, msg = beam_search_swarm_solver(
        problem, cost_function, initial, l_width,
        r_intra, r_inter, f_intra, f_inter, verbose
    )

    best_sol = solswarm.extract_best_solution()
    if verbose and best_sol:
        print(f"[BeamSearch] Best solution max_cost: {best_sol.max_cost}")

    return best_sol, msg


if __name__ == "__main__":
    from share_a_ride.solvers.algo.greedy import greedy_balanced_solver
    from share_a_ride.core.utils.generator import generate_instance_coords

    # Example usage
    prob = generate_instance_coords(
        N = 300,
        M = 200,
        K = 27,
        area = 1000,
        seed = 60
    )

    def _run():
        return beam_search_solver(
            prob,
            cost_function=parsol_scorer,
            l_width=10,
            verbose=True
        )

    prof = cProfile.Profile()
    prof.enable()
    sol, info = _run()
    prof.disable()
    prof.dump_stats("beam_search_profile.prof")

    assert sol is not None
    sol.stdin_print(verbose=True)

    sol2, info2 = greedy_balanced_solver(
        prob,
        verbose=False
    )
    assert sol2 is not None
    sol2.stdin_print(verbose=True)
