"""
Utility functions to compute weights from some score lists.
"""
from typing import Sequence, Union, List, Tuple




def weighted(kind: str, inc: Union[int, float], pweight: float = 0.7) -> float:
    """
    Because pserve is a 2-node collapsed action, we have to reweight the
    increment of its cost to reflect the equality in contribution of the three
    action types.
    """
    if kind == "serveP":
        return pweight * inc
    return inc + 1


Action = Tuple[int, str, int, int]
def action_weight(action: Action, pweight: float = 0.7) -> float:
    """
    Weight action incremental cost by calling ``weighted``.
    """
    return weighted(action[1], action[3], pweight)


def softmax_weighter(incs: Sequence[Union[int, float]], t: float) -> List[float]:
    """
    Given a list of incremental costs, compute weights using a temperature-scaled softmax.
    Lower increments get higher weights. If all increments are similar, returns uniform weights.
    """
    min_inc, max_inc = min(incs), max(incs)
    inc_range = max_inc - min_inc

    if inc_range < 1e-6:     # All increments similar: uniform weights
        return [1.0] * len(incs)

    # Else, softmax weights
    weights: List[float] = []
    for inc in incs:
        normalized = (inc - min_inc) / inc_range
        weights.append((1.0 - normalized + 0.1) ** (1.0 / t))

    return weights
