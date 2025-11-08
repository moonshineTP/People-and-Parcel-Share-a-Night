"""
Utility functions to compute weights from some score lists.
"""

def softmax_weighter(incs: list[int] | list[float], T: float) -> list[float]:
    """
    Given a list of incremental costs, compute weights using a temperature-scaled softmax.
    Lower increments get higher weights. If all increments are similar, returns uniform weights.
    """
    min_inc, max_inc = min(incs), max(incs)
    inc_range = max_inc - min_inc

    if inc_range < 1e-6:     # All increments similar: uniform weights
        return [1.0] * len(incs)

    # Else, softmax weights
    weights = []
    for inc in incs:
        normalized = (inc - min_inc) / inc_range
        weights.append((1.0 - normalized + 0.1) ** (1.0 / T))

    return weights
