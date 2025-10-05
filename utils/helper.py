from typing import List

# ---------------------- Helper functions ----------------------------------------
def route_length_from_sequence(
    seq: List[int],
    D: List[List[int]]
) -> int:
    pos = 0
    total = 0
    for node in seq:
        total += D[pos][node]
        pos = node
    total += D[pos][0]
    return total