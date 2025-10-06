from typing import List

# ---------------------- Helper functions ----------------------------------------
def route_cost_from_sequence(seq: List[int], D: List[List[int]]) -> int:
    prev, total_cost = 0, 0

    for node in seq[1:]:
        total_cost += D[prev][node]
        prev = node

    return total_cost