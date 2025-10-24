import random
from typing import Tuple, Optional

from share_a_ride.core.solution import PartialSolution
from share_a_ride.solvers.utils.sampler import sample_from_weight
from share_a_ride.solvers.utils.weighter import softmax_weighter


def build_operator(
        partial: PartialSolution,
        route_idx: int,
        steps: int = 5,
        T: float = 1.0,
        seed: Optional[int] = 42,
        verbose: bool = False
    ) -> Tuple[PartialSolution, int]:
    """
    Expand a partial route directly on the given PartialSolution by applying up to
    ``steps`` feasible actions. 
    Use softmax weighting controlled by ``T`` to favor lower incremental travel cost. 
    Returns the updated PartialSolution and the number of new nodes added to the route.
    """

    assert steps > 0, "Number of steps must be positive."
    assert T > 1e-5, "Temperature T must be positive."

    rng = random.Random(seed)
    added_nodes = 0  # count of newly added nodes (actions applied)

    # Main building loop
    for _ in range(steps):
        state = partial.route_states[route_idx]
        if state["ended"]:
            break

        actions = partial.possible_actions(route_idx)
        if verbose:
            print(f"[build] route {route_idx} available actions: {actions}")

        if not actions:
            if verbose:
                print(f"[build] route {route_idx} has no feasible actions, ending.")

                partial.apply_return_to_depot(route_idx)
                added_nodes += 1
                break

        incs = [action[2] for action in actions]
        weights = softmax_weighter(incs, T)
        selected_idx = sample_from_weight(rng, weights)

        kind, node_idx, inc = actions[selected_idx]

        if verbose:
            print(f"[build] route {route_idx} selected action: {actions[selected_idx]}")

        partial.apply_action(route_idx, kind, node_idx, inc)
        added_nodes += 1


    if verbose:
        print(f"[build] route {route_idx} finished building, added {added_nodes} nodes.")

    return partial, added_nodes
