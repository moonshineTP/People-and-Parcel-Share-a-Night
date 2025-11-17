"""
Module implementing destroy operators for Share-a-Ride solutions.
These operators remove parts of vehicle routes to allow for solution improvement.
"""

import random

from typing import List, Tuple

from share_a_ride.core.solution import Solution, PartialSolution
from share_a_ride.solvers.utils.sampler import sample_from_weight



def destroy_one_route(
        route: List[int],
        route_idx: int,
        steps: int = 10,
        verbose: bool = False
    ) -> List[int]:
    """
    Remove a suffix of the given route (keeping the starting depot).
    The resulting route is still a valid partial route.
    """

    res_route = route[:-1]                              # Exclude the ending depot
    remove_len = min(steps, max(0, len(res_route) - 1)) # Ensure at least the starting depot remains
    if remove_len <= 0:
        return route[:]

    suffix_start = len(res_route) - remove_len
    destroyed_route = res_route[:suffix_start]
    if not destroyed_route:
        destroyed_route = [0]

    if verbose:
        print(f"[Operator: Destroy]: last {remove_len} nodes from route {route_idx} removed.")

    return destroyed_route



def destroy_operator(
        sol: Solution,
        destroy_proba: float,
        destroy_steps: int,
        seed: int = 42,
        T: float = 1.0
    ) -> Tuple[PartialSolution, List[bool], int]:
    """
    Select a subset of routes to destroy based on their cost
    The selection use a temperature-based heuristic with probabilities    
    Temperature controls the selection bias:
    - temperature = 0: always select the most expensive routes (greedy)
    - temperature = inf: uniform random selection
    - temperature = 1: balanced probabilistic selection    
    Higher cost routes have higher probability of being selected, but with
    some randomness to allow exploration.
    
    Args:
        prob: ShareARideProblem instance
        sol: Current solution
        destroy_proba: Fraction of routes to destroy (0 to 1)
        destroy_steps: Maximum number of nodes to remove per route
        seed: for reproducibility
        temperature: Controls selection randomness (default 1.0)
    
    Returns:
        (partial_sol, flags, num_removed): 
            - partial_sol: The modified partial solution after destruction
            - flags: Boolean list indicating which routes were destroyed
            - num_removed: Total number of nodes removed
    """
    rng = random.Random(seed)

    routes = [route[:] for route in sol.routes]
    costs = sol.route_costs
    flags = [False] * len(routes)
    num_removed = 0

    if not routes:
        return PartialSolution(problem=sol.problem, routes=routes), flags, num_removed
    approx_destroyed_count = round(destroy_proba * len(routes) + 0.5)
    destroyed_count = min(sol.problem.K, max(1, approx_destroyed_count))

    # Normalize costs to probabilities using temperature
    min_cost = min(costs) if costs else 0.0
    max_cost = max(costs) if costs else 1.0
    cost_range = max_cost - min_cost
    temperature = max(T, 1e-6)

    if cost_range < 1e-6:
        # All routes have similar cost, select uniformly at random
        selected_ids = rng.sample(range(sol.problem.K), destroyed_count)
    else:
        # Collect weights based on normalized costs
        weights = []
        for cost in costs:
            normalized = (cost - min_cost) / cost_range
            weights.append((normalized + 0.1) ** (1.0 / temperature))

        # Select routes based on weights
        selected_ids = []
        available_ids = list(range(sol.problem.K))
        available_weights = weights
        for _ in range(destroyed_count):
            total_weight = sum(available_weights)

            if total_weight < 1e-10:    # All weights zero: pick any subset
                selected_ids.extend(
                    available_ids[:destroyed_count - len(selected_ids)]
                )
                break
            else:                       # Weighted random selection
                selected_idx = sample_from_weight(rng, available_weights)
                selected_ids.append(available_ids[selected_idx])

                # Remove selected index from available
                available_ids.pop(selected_idx)
                available_weights.pop(selected_idx)

                if not available_ids:
                    break


    # Destroy selected routes
    for idx in selected_ids:
        route = routes[idx]

        # Skip empty routes
        if len(route) <= 2:
            continue

        # Update if any nodes were removed
        reduced = destroy_one_route(route, idx, steps=destroy_steps, verbose=False)
        removed = max(0, len(route) - len(reduced))

        if removed > 0:
            routes[idx] = reduced
            flags[idx] = True
            num_removed += removed

    partial_sol = PartialSolution(problem=sol.problem, routes=routes)

    return partial_sol, flags, num_removed
