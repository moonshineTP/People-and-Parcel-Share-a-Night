"""
Beam search module for the Share-a-Ride problem.
"""
import time
from typing import Any, List, Optional, Tuple, Dict, Callable, Concatenate, ParamSpec

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, Solution, PartialSolutionSwarm
from share_a_ride.solvers.operator.swap import intra_swap_operator, inter_swap_operator
from share_a_ride.solvers.operator.relocate import relocate_operator
from share_a_ride.solvers.algo.utils import (
    balanced_scorer, enumerate_actions_greedily, apply_general_action
)
from share_a_ride.solvers.algo.greedy import iterative_greedy_solver

# Type alias for defense policy function
Params = ParamSpec("Params")
DefensePolicy = Callable[Concatenate[PartialSolution, Params], Optional[Solution]]




# ================ Default Defense Policy ================
def _default_defense_policy(
        partial: PartialSolution,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Optional[Solution]:
    """
    Defense policy: complete a promising partial with beam search.
    This mirrors the default defense strategy used in ``mcts.py``.
    """

    best, _ = iterative_greedy_solver(
        partial.problem,
        partial,
        iterations=1000,
        seed=113*seed if seed is not None else None,
        verbose=verbose,
    )
    return best




# ================ Beam Search Solver =================
def beam_enumerator(
        problem: ShareARideProblem,
        swarm: Optional[PartialSolutionSwarm] = None,
        n_partials: int = 20,
        n_return: int = 10,
        width: int = 5,
        r_intra: float = 0.55,
        r_inter: float = 0.75,
        f_intra: float = 0.10,
        f_inter: float = 0.10,
        cost_function: Any = balanced_scorer,
        time_limit: float = 30.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[PartialSolutionSwarm, Dict[str, Any]]:
    """
    Beam search solver using a swarm of partial solutions.

    This should work as follows:
    - At each depth step, maintains a beam of the best partial solutions with the
    number of actions equal to the depth. Note that the first few steps may have
    less than ``n_partials`` solutions if not enough distinct partial solutions exist.
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
    start = time.time()
    total_actions = problem.num_actions
    if swarm is None:
        swarm = PartialSolutionSwarm(
            solutions=[PartialSolution(problem=problem, routes=[])]
        )

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


    # //// Helper functions ////
    # Partial list sort
    def beam_sort(beam: List[PartialSolution]) -> None:
        beam.sort(key=lambda ps: cost_function(ps, seed=seed))

    # Apply local refinements to a beam
    def apply_local_refinements(
            beam: List[PartialSolution],
            use_intra: bool,
            use_inter: bool,
            depth: int,
            base_seed: Optional[int]
        ) -> List[PartialSolution]:
        if not (use_intra or use_inter):
            return beam

        # Apply refinements to each partial solution.
        intra_seed, inter_seed = None, None
        refined: List[PartialSolution] = []
        intra_showed = False
        inter_showed = False
        for idx, base in enumerate(beam):
            updated = base
            if base_seed is not None:
                intra_seed = base_seed + 108 * depth + 11 * idx
                inter_seed = base_seed + 801 * depth + 31 * idx + 23

            if use_intra:
                if verbose and not intra_showed:
                    print(f"[BeamSearch] Depth {depth}: Applied intra-swap operator.")
                    intra_showed = True
                updated, _, _ = intra_swap_operator(
                    updated,
                    steps=None,
                    mode='first',
                    uplift=1,
                    seed=intra_seed,
                    verbose=False
                )

            if use_inter:
                if verbose and not inter_showed:
                    print(f"[BeamSearch] Depth {depth}: Applied inter-swap operator.")
                    inter_showed = True
                updated, _, _ = inter_swap_operator(
                    updated,
                    steps=None,
                    mode='first',
                    uplift=1,
                    seed=inter_seed,
                    verbose=False
                )

            refined.append(updated)

        # Return refined partial solutions.
        return refined

    # Expand a beam to the next depth
    def expand(beam: List[PartialSolution], aggressive=False) -> List[PartialSolution]:
        candidates: List[PartialSolution] = []
        actions_per = min(3, width) if aggressive else width

        for partial in beam:
            actions = enumerate_actions_greedily(
                partial,
                width=actions_per,
                asymmetric=True
            )

            for action in actions:
                child = partial.copy()
                apply_general_action(child, action)

                # Ensure uniqueness in insertion
                if not any(child.is_identical(other) for other in candidates):
                    candidates.append(child)

        # Sort candidates by cost function and return top width
        beam_sort(candidates)
        candidates = candidates[:n_partials]
        return candidates[:n_partials]


    # //////// Main loop ////////
    beam = swarm.partial_lists
    beam_sort(beam)
    depth = swarm.partial_lists[0].num_actions  # should be 0 initially
    iterations = 0
    status = "done"

    while beam:
        # //// Validation checks for termination.
        if time.time() - start >= time_limit:
            status = "overtime"
            if verbose:
                print(
                    f"[BeamSearch] Time limit ({time_limit:.2f}s) reached at depth {depth}. "
                    f"Terminating search."
                )
            break
        if not beam[0].is_pending():
            if verbose:
                print(
                    f"[BeamSearch] All solutions in beam completed at depth {depth}. "
                    f"Terminating search."
                )
            break


        # //// Apply local refinements if it's the turn.
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
        if use_intra_phase or use_inter_phase:
            beam = apply_local_refinements(
                beam,
                use_intra_phase,
                use_inter_phase,
                depth=depth,
                base_seed=seed
            )
            beam_sort(beam)


        # //// Expand each partial solution in the beam
        # Expand
        next_beam = expand(beam, aggressive=True)
        if not next_beam:
            raise RuntimeError("Beam search stalled: no candidates generated.")
        if len(next_beam) < n_partials:
            next_beam = expand(beam, aggressive=False)
        # Update
        beam = next_beam
        depth += 1
        iterations += 1


        # //// Logging
        if verbose:
            max_costs = [ps.max_cost for ps in beam]
            magnitude = 10 ** (len(str(depth)) - 1)
            if depth % magnitude == 0 or depth % 100 == 0:  # log at intervals
                print(
                    f"[BeamSearch] Depth {depth}. Max_cost range: "
                    f"{min(max_costs)} - {max(max_costs)}. "
                    f"Average max cost: {sum(max_costs) / len(max_costs):.2f}. "
                    f"Time elapsed: {time.time() - start:.2f}s."
                )


    # Sort final beam and return best solution swarm.
    beam_sort(beam)
    beam = beam[:n_return]

    # Info
    swarm = PartialSolutionSwarm(solutions=beam)
    search_info = {
        "iterations": iterations,
        "time": time.time() - start,
        "status": status,
    }

    # Logging.
    if verbose:
        print()
        print(f"[BeamSearch] Completed. Final beam size {len(beam)}")
        print(f"[BeamSearch] Beam depth reached: {depth}")
        print(f"[BeamSearch] Beam cost range: {swarm.min_cost} - {swarm.max_cost}")
        print(f"[BeamSearch] Average max cost: {swarm.avg_cost}")
        print(f"[BeamSearch] Time taken: {search_info['time']:.4f} seconds")
        print("------------------------------")
        print()

    return swarm, search_info




def beam_solver(
        problem: ShareARideProblem,
        swarm: Optional[PartialSolutionSwarm] = None,
        n_partials: int = 20,
        n_returns: int = 10,
        width: int = 5,
        r_intra: float = 0.55,
        r_inter: float = 0.75,
        f_intra: float = 0.05,
        f_inter: float = 0.10,
        cost_function: Any = balanced_scorer,
        defense_policy: DefensePolicy = _default_defense_policy,
        time_limit: float = 30.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Beam search solver for the Share-a-Ride problem. This uses a beam search
    approach to explore a swarm of partial solutions, maintaining a balance
    between solution quality.

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
    solswarm, stats = beam_enumerator(
        problem=problem,
        swarm=swarm,
        n_partials=n_partials,
        n_return=n_returns,
        r_intra=r_intra,
        r_inter=r_inter,
        f_intra=f_intra,
        f_inter=f_inter,
        cost_function=cost_function,
        time_limit=time_limit,
        seed=seed,
        verbose=verbose
    )

    # Apply defense policy to complete any pending partial solutions (e.g. if timed out).
    # This ensures we have complete solutions before refinement.
    if verbose:
        print("[BeamSearch] Applying defense policy to complete pending partial solutions...")
    for idx, partial in enumerate(solswarm.partial_lists):
        if partial.is_pending():
            completed_sol = defense_policy(
                partial,
                seed=seed,
                verbose=False
            )
            if completed_sol is not None:
                solswarm.partial_lists[idx] = PartialSolution.from_solution(completed_sol)

    # Apply relocate operator to final beam for refinement.
    if verbose:
        print("[BeamSearch] Applying relocate operator to refine final solutions...")
    for idx, partial in enumerate(solswarm.partial_lists):
        refined_partial, _, _ = relocate_operator(
            partial,
            steps=None,
            mode='first',
            uplift=1,
            seed=seed,
            verbose=False
        )
        solswarm.partial_lists[idx] = refined_partial
    if verbose:
        print("[BeamSearch] Relocate applied. Selecting best solution...")

    best_sol: Optional[Solution] = solswarm.opt()
    if best_sol is None:
        if verbose:
            print("[BeamSearch] No complete solution found in beam swarm.")
            print("[BeamSearch] Attempting defense policy to complete best partial...")

        # Apply defense policy to complete best partial solution.
        best_partial = min(solswarm.partial_lists, key=lambda ps: ps.max_cost)
        best_sol = defense_policy(
            best_partial,
            seed=seed,
            verbose=verbose
        )

    if verbose:
        if best_sol is not None:
            print(
                f"[BeamSearch] Best solution found with max_cost: {best_sol.max_cost}."
            )
            print("------------------------------")
            print()
        else:
            print("[BeamSearch] No valid solution could be constructed.")
            print("------------------------------")
            print()

    return best_sol, stats




# ================ Playground ================
if __name__ == "__main__":
    from share_a_ride.solvers.algo.utils import relay_problems

    test_problem = relay_problems[-1]
    sol, info = beam_solver(
        test_problem,
        time_limit=60.0,
        seed=42,
        verbose=True
    )
