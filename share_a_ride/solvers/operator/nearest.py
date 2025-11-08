"""
Nearest Operator Module.
Implements the nearest operator for route modification.
"""
from typing import Tuple, List

from share_a_ride.core.solution import PartialSolution

def nearest_operator(
        par: PartialSolution,
        route_idx: int,
        step: int = 1
    ) -> Tuple[PartialSolution, List[bool], int]:
    """
    Nearest operator that adds the nearest feasible node to the route.
    Args:
        par: PartialSolution instance
        route_idx: Index of the route to modify
        step: Number of nodes to add
    Returns:
        - updated PartialSolution
        - list of booleans indicating which routes were modified (only route_idx here)
        - number of newly added nodes (actions applied)
    """
    added_nodes = 0
    for _ in range(step):
        state = par.route_states[route_idx]
        if state["ended"]:
            break

        actions = par.possible_actions(route_idx)

        # Depot case: end the route
        if not actions:
            par.apply_return_to_depot(route_idx)
            added_nodes += 1
            break

        # Search case: select nearest feasible action
        nearest_action = min(actions, key=lambda x: x[2])  # x[2] is incremental cost
        kind, node_idx, inc = nearest_action
        par.apply_action(route_idx, kind, node_idx, inc)
        added_nodes += 1

    # Return modified flags and count
    modified = [r_idx == route_idx for r_idx in range(par.problem.K)]
    return par, modified, added_nodes
