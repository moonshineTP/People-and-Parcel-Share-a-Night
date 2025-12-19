"""
Repair operator that expands a partial route by applying feasible actions.
"""
import random
from typing import List, Tuple, Optional

from share_a_ride.core.solution import PartialSolution
from share_a_ride.solvers.utils.sampler import sample_from_weight
from share_a_ride.solvers.utils.weighter import weighted, softmax_weighter




def repair_one_route(
        partial: PartialSolution,
        route_idx: int,
        steps: int,
        T: float = 1.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[PartialSolution, int]:
    """
    Repair operator for a single route.
    Expands route ``route_idx`` on ``partial`` by applying up to ``steps`` feasible actions.
    Uses softmax weighting (temperature ``T``) over incremental costs to bias towards
    cheaper extensions. Returns the updated partial, per-route modification flags, and
    the number of actions applied.
    """

    assert steps > 0, "Number of steps must be positive."
    assert T > 1e-5, "Temperature T must be positive."


    # //// Repair loop
    rng = random.Random(seed)
    added_actions = 0
    for _ in range(steps):
        state = partial.states[route_idx]
        if state["ended"]:
            break

        # Expansion case
        actions = partial.possible_expand(route_idx)
        if not actions:
            if verbose:
                print(f"[Repair] Route {route_idx} has no feasible actions, return to depot.")
            partial.apply_return(route_idx)
            added_actions += 1
            break

        # Softmax sampling
        incs = [weighted(kind, inc) for kind, _, inc in actions]
        weights = softmax_weighter(incs, T)
        selected_idx = sample_from_weight(rng, weights)

        # Apply
        kind, node_idx, inc = actions[selected_idx]
        partial.apply_extend(route_idx, kind, node_idx, inc)
        added_actions += 1

        if verbose:
            print(f"[Repair] Route {route_idx} select action {actions[selected_idx]}")

    if verbose:
        print(f"[Repair] Route {route_idx} finished building, added {added_actions} actions.")

    return partial, added_actions




def repair_operator(
        partial: PartialSolution,
        repair_proba: Optional[float] = None,
        steps: Optional[int] = None,
        T: float = 1.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[PartialSolution, List[bool], int]:
    """
    Repair operator that expands multiple routes in the partial solution.
    Selects a subset of routes to repair based on the repair probability.
    Each selected route is expanded by applying up to ``steps`` feasible actions.
    Uses softmax weighting (temperature ``T``) over incremental costs to bias towards
    cheaper extensions.

    Returns the signature of `operator` including updated partial, per-route
    modification flags, and the total number of actions applied.
    """

    # Initialize parameters
    rng = random.Random(seed)
    if repair_proba is None:
        repair_proba = 1.0
    if steps is None:
        steps = 10**9  # Effectively unlimited

    # Initialize tracking variables
    routes = list(range(partial.problem.K))
    num_routes = partial.problem.K
    approx_repair_count = round(repair_proba * num_routes + 0.5)
    repair_count = min(num_routes, max(1, approx_repair_count))


    # //// Sample and repair routes ////
    selected_routes = rng.sample(routes, repair_count)
    total_added_actions = 0
    modified = [False] * num_routes

    for r_idx in selected_routes:
        partial, added_actions = repair_one_route(
            partial=partial,
            route_idx=r_idx,
            steps=steps,
            T=T,
            seed=rng.randint(0, 1_000_000),
            verbose=verbose
        )

        total_added_actions += added_actions
        modified[r_idx] = True

        if verbose:
            print(f"[Repair]: Repairing route {r_idx} with up to {steps} steps.")


    # Logging
    if verbose:
        print()
        print("[Repair] Operator completed.")
        print(f"Total routes repaired: {repair_count};")
        print(f"Total actions added: {total_added_actions}.")
        print("------------------------------")
        print()

    partial.stdin_print()

    return partial, modified, total_added_actions
