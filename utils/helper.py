from typing import List

# ---------------------- Helper functions ----------------------------------------
def route_cost_from_sequence(
        seq: List[int], 
        D: List[List[int]], 
        verbose: bool = False
    ) -> int:

    prev, total_cost = 0, 0
    for node in seq[1:]:
        if verbose: 
            print(D[prev][node], end=" ")

        total_cost += D[prev][node]
        prev = node
        
    if verbose:
        print()

    return total_cost
