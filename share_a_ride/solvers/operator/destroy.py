"""
Destroy Operator Module.
Implements destroy operators for route modification in Share-a-Ride solutions.
These operators remove parts of vehicle routes to allow for solution improvement.
"""
import random
from typing import List, Tuple, Optional, Union

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import Solution, PartialSolution
from share_a_ride.solvers.utils.sampler import sample_from_weight
from share_a_ride.solvers.utils.weighter import softmax_weighter




def destroy_one_route(
        problem: ShareARideProblem,
        route: List[int],
        route_idx: int,
        steps: int = 10,
        verbose: bool = False
    ) -> Tuple[List[int], int]:
    """
    Remove a suffix of the given route (keeping the starting depot).
    The resulting route is still a valid partial route.

    Args:
        route: List of node indices representing the route.
        route_idx: Index of the route (for logging purposes).
        steps: Maximum number of nodes to remove from the suffix.
        verbose: If True, print detailed logs.

    Returns:
        The destroyed route with suffix removed.
    """
    # Define the route and remove length
    res_route = route[:]  # Make a copy
    actions_removed = 0

    # Remove up to `steps` actions from the end
    while actions_removed < steps and len(res_route) > 1:
        nodeid = res_route.pop()
        if problem.is_pdrop(nodeid):
            pid = problem.rev_pdrop(nodeid)
            pickid = problem.pserve(pid)[0]
            if res_route.pop() != pickid:
                raise RuntimeError(
                    "The destroyed route is likely invalid beforehand."
                )
        elif problem.is_lpick(nodeid) or problem.is_ldrop(nodeid) or nodeid == 0:
            pass    # for other node type, it is quite safe to remove directly
        else:
            raise RuntimeError(
                "The destroyed route is likely invalid beforehand."
            )

        actions_removed += 1

    # Logging
    if verbose:
        print(f"[Destroy] Route {route_idx}: removed {actions_removed} actions.")

    return res_route, actions_removed




def destroy_operator(
        sol: Union[PartialSolution, Solution],
        destroy_proba: float,
        destroy_steps: int,
        seed: Optional[int] = None,
        t: float = 1.0,
        verbose: bool = False
    ) -> Tuple[PartialSolution, List[bool], int]:
    """
    Destroy operator that selects routes to partially destroy based on their cost.

    The selection uses a temperature-based softmax heuristic:
    - t -> 0: greedily select the most expensive routes
    - t -> inf: uniform random selection
    - t = 1: balanced probabilistic selection

    Higher cost routes have higher probability of being selected, but with
    some randomness to allow exploration.

    Args:
        - sol: Current Solution instance.
        - destroy_proba: Fraction of routes to destroy (0 to 1).
        - destroy_steps: Maximum number of nodes to remove per route.
        - seed: Random seed for reproducibility.
        - t: Temperature for softmax selection (default 1.0).
        - verbose: If True, print detailed logs.

    Returns:
        - partial_sol: The modified PartialSolution after destruction.
        - flags: Boolean list indicating which routes were destroyed.
        - num_removed: Total number of nodes removed.
    """
    rng = random.Random(seed)

    # Initialize working copies
    K = sol.problem.K   # number of vehicles
    routes = [route[:] for route in sol.routes]
    costs = sol.route_costs

    # Compute number of routes to destroy
    approx_destroyed_count = round(destroy_proba * K + 0.5)
    destroyed_count = min(K, max(1, approx_destroyed_count))


    # //// Route sampling based on cost ////
    # Extract weights using softmax
    weights = softmax_weighter(costs, t=t)

    # Sampling loop
    selected_ids: List[int] = []
    available_ids = list(range(K))
    available_weights = weights[:]

    for _ in range(destroyed_count):
        if not available_ids:
            break

        # Sample one route based on weights
        selected_idx = sample_from_weight(rng, available_weights)
        selected_ids.append(available_ids[selected_idx])

        # Remove selected from available pool
        available_ids.pop(selected_idx)
        available_weights.pop(selected_idx)


    # //// Route destruction ////
    flags = [False] * K
    actions_removed = 0
    for idx in selected_ids:
        route = routes[idx]

        # Skip empty or minimal routes
        if len(route) <= 2:
            continue

        # Apply destruction
        reduced_route, num_removed = destroy_one_route(
            sol.problem, route, idx, steps=destroy_steps, verbose=verbose
        )

        # Update route and tracking variables
        if num_removed > 0:
            routes[idx] = reduced_route
            flags[idx] = True
            actions_removed += num_removed

    # Logging
    if verbose:
        print()
        print("[Destroy] Operation complete.")
        print(
            f"[Destroy] Destroyed {len(selected_ids)} routes, "
            f"removed {actions_removed} nodes total."
        )
        print("------------------------------")
        print()

    # Create new PartialSolution
    new_partial = PartialSolution(problem=sol.problem, routes=routes)
    return new_partial, flags, actions_removed
