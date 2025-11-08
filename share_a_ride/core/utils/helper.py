"""
Helper functions for the core module.
"""
def route_cost_from_sequence(
        seq: list[int], 
        D: list[list[int]], 
        verbose: bool = False
    ) -> int:
    """
    Compute the total cost of a route given its sequence of nodes and the distance matrix D.
    Params:
    - seq: list of node indices representing the route (must start with depot 0)
    - D: distance matrix
    - verbose: if True, prints individual leg costs
    Returns: total cost of the route
    """

    assert seq and seq[0] == 0

    prev, total_cost = 0, 0
    for node in seq[1:]:
        if verbose: 
            print(D[prev][node], end=" ")

        total_cost += D[prev][node]
        prev = node
   
    if verbose:
        print()

    return total_cost
