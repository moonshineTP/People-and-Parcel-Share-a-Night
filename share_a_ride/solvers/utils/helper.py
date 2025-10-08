from typing import List, Tuple, Iterator
import time
from share_a_ride.problem import ShareARideProblem

def gen_routes_for_taxi(
        prob: ShareARideProblem,            # problem instance
        taxi_pairs: List[Tuple[str, int]],  # list of ("P"/"L", id)
        taxi_idx: int,                      # taxi index
        start_time: float,
        time_limit: float
    ) -> Iterator[List[int]]:

    """
    DFS generator yielding feasible routes for one taxi.
    Ensures:
        - pickup before drop
        - capacity respected
        - at most 1 passenger onboard
    Returns:
        Iterator over valid routes (list of node indices).
    """

    num_pairs = len(taxi_pairs)
    if num_pairs == 0:
        yield []
        return

    # Precomputed lists
    types = [type for type, _ in taxi_pairs]
    ids = [id for _, id in taxi_pairs]
    pick_nodes = [prob.ppick(x) if type == "P" else prob.parc_pick(x)
                    for type, x in taxi_pairs]
    drop_nodes = [prob.pdrop(x) if type == "P" else prob.parc_drop(x)
                    for type, x in taxi_pairs]

    # External containers and state.
    visited_pick = set()        # indices where pickup is done.
    visited_drop = set()        # indices where drop is done.
    current_passenger = None    # holds index of current passenger onboard, if any.
    current_parcels = set()     # indices for parcels currently onboard.
    current_load = 0            # current parcel load.
    seq: List[int] = [0]        # current route sequence (starts at depot)

    # main dfs function
    def dfs() -> Iterator[List[int]]:
        nonlocal current_passenger, current_parcels, current_load, seq

        # Check time limit
        if time.time() - start_time > time_limit:
            return

        # Base-case: all pairs fully processed (all drops done)
        if len(visited_drop) == num_pairs:
            yield list(seq + [0])
            return

        # Try pickup moves.
        for i in range(num_pairs):
            if i not in visited_pick:
                # For a passenger, ensure no passenger is onboard.
                if types[i] == "P" and (current_passenger is not None):
                    continue
                # For a parcel, ensure capacity is not exceeded.
                if types[i] == "L":
                    q = prob.q[ids[i] - 1]
                    if current_load + q > prob.Q[taxi_idx]:
                        continue

                # Do the pickup.
                visited_pick.add(i)
                seq.append(pick_nodes[i])
                old_passenger = current_passenger
                old_load = current_load
                old_parcels = set(current_parcels)
                if types[i] == "P":
                    current_passenger = i
                elif types[i] == "L":
                    current_load += prob.q[ids[i] - 1]
                    current_parcels.add(i)
                
                # Recurse
                yield from dfs()
                
                # Backtrack.
                seq.pop()
                visited_pick.remove(i)
                current_passenger = old_passenger
                current_load = old_load
                current_parcels = old_parcels

        # Try drop moves.
        for i in range(num_pairs):
            if i in visited_pick and i not in visited_drop:
                # For a passenger drop, ensure that the onboard passenger is the correct one.
                if types[i] == "P" and current_passenger != i:
                    continue
                # For a parcel drop, ensure the parcel is onboard.
                if types[i] == "L" and i not in current_parcels:
                    continue

                # Do the drop.
                visited_drop.add(i)
                seq.append(drop_nodes[i])
                # Save old state.
                old_passenger = current_passenger
                old_load = current_load
                old_parcels = set(current_parcels)
                if types[i] == "P":
                    current_passenger = None
                elif types[i] == "L":
                    current_load -= prob.q[ids[i] - 1]
                    assert current_load >= 0

                    current_parcels.remove(i)

                # Recurse
                yield from dfs()

                # Backtrack: remove drop, reset state.
                seq.pop()
                visited_drop.remove(i)
                current_passenger = old_passenger  # reset current passenger
                current_load = old_load            # reset load
                current_parcels = old_parcels

    yield from dfs()