from typing import List

from share_a_ride.problem import ShareARideProblem

def destroy(
        prob: ShareARideProblem,
        route: List[int],
        max_remove: int = 10,
        verbose: bool = False
    ) -> List[int]:
    """
    Remove a suffix of the given route (keeping the starting depot).
    The resulting route is still a valid partial route.
    """

    if not route or len(route) <= 2:
        return route[:]

    res_route = route[:-1] 
    remove_len = min(max_remove, max(0, len(res_route) - 1))
    if remove_len <= 0:
        return route[:]

    suffix_start = len(res_route) - remove_len

    if verbose:
        print(f"[Operator: Destroy]: removing last {remove_len} nodes from route")

    kept_prefix = res_route[:suffix_start]
    if not kept_prefix:
        kept_prefix = [0]

    return kept_prefix