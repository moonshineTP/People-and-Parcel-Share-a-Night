"""
Helper functions for sampling in solvers/operators.
"""

import random
from typing import Union, Sequence

def sample_from_weight(rng: random.Random, weights: Sequence[Union[int, float]]) -> int:
    """
    Sample an index from a list of weights using Knuth's algorithm.
    Returns the selected index.
    """
    total_weight = sum(weights)

    if total_weight < 1e-10:    # All weights zero: pick uniformly
        res_idx = rng.randrange(len(weights))
    else:                       # Weighted random selection
        rand_val = rng.random() * total_weight
        cumsum = 0.0
        res_idx = 0
        for i, weight in enumerate(weights):
            cumsum += weight
            if rand_val <= cumsum:
                res_idx = i
                break

    return res_idx
