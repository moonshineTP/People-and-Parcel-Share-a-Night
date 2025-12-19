"""
Utility for generating simple randomized Share-a-Ride instances.
Use for the early stages of algorithm development and testing.
"""

import math
import random

from typing import Optional, List, Tuple
from share_a_ride.core.problem import ShareARideProblem




# ================ Computation Utilities ================
def route_cost_from_sequence(
        seq: List[int],
        D: List[List[int]],     # pylint: disable=invalid-name
        verbose: bool = False
    ) -> int:
    """
    Compute the total cost of a (possibly prematured) route given its sequence of nodes
    and the cost matrix D.

    Params:
    - seq: list of node indices representing the route (must start with depot 0)
    - D: cost matrix
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



# ================ Sampling Utilities ================
def _sampler_cost_value(
        i: int, j: int, dmat: List[List[int]], rng: random.Random,
        low: int, high: int, lmbd: Optional[float], asymmetric: bool
    ) -> int:
    """
    Generate a single distance value, symmetric if needed.
    """
    if i == j:
        return 0
    if asymmetric:
        if lmbd is not None:
            return _sample_poisson(rng, low, high, lmbd)
        return rng.randint(low, high)
    if j < i:
        return dmat[j][i]
    if lmbd is not None:
        return _sample_poisson(rng, low, high, lmbd)
    return rng.randint(low, high)


def _sample_poisson(
        rng: random.Random,
        low: int,
        high: int,
        lmbd: float,
    ) -> int:
    """
    Sample from 1D Poisson distribution with parameter ``lmbd``
    and proceed until result is in [low, high].
    """

    while True:
        # Parameters
        exp_minus_l = math.exp(-lmbd)
        k = 0
        p = 1.0

        # Generate Poisson sample using Knuth's algorithm
        while p > exp_minus_l:
            k += 1
            p *= rng.random()
        value = k - 1

        # Check range and return
        if low <= value <= high:
            return value




# ================ Generator Utilities ================
def random_distance_matrix(
        n: int,
        low: int,
        high: int,
        lmbd: float,
        asymmetric: bool,
        seed: Optional[int],
    ) -> List[List[int]]:
    """
    Generate a random symmetric or asymmetric distance matrix.
    """
    rng = random.Random(seed)
    dmat = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            dmat[i][j] = _sampler_cost_value(i, j, dmat, rng, low, high, lmbd, asymmetric)

    return dmat


def euclidean_distance_matrix(
        coords: List[Tuple[int, int]]
    ) -> List[List[int]]:
    """
    Compute pairwise Euclidean distance matrix from coordinates,
    rounding distances to the nearest integer.
    """

    n = len(coords)
    dmat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = int(round(math.hypot(
                coords[i][0] - coords[j][0],
                coords[i][1] - coords[j][1]
            )))
            dmat[i][j] = dmat[j][i] = dist

    return dmat




# ================ Generator APIs ================
def generate_instance_lazy(
        N: int, M: int, K: int,     # pylint: disable=invalid-name
        low: int = 10, high: int = 50, lmbd: float = 20.0,
        qlow: int = 5, qhigh: int = 15, qlmbd: float = 10.0,
        Qlow: int = 15, Qhigh: int = 30, Qlmbd: float = 20.0,   # pylint: disable=invalid-name
        use_poisson: bool = False,
        seed: Optional[int] = None
    ) -> ShareARideProblem:

    """
    Generate random instance using lazy distance matrix.
    """

    rng = random.Random(seed)
    n_nodes = 2*N + 2*M + 1

    if use_poisson:
        qtys = [_sample_poisson(rng, qlow, qhigh, qlmbd) for _ in range(M)]
        caps = [_sample_poisson(rng, Qlow, Qhigh, Qlmbd) for _ in range(K)]
        dmat = random_distance_matrix(
            n_nodes, low=low, high=high, lmbd=lmbd, asymmetric=True, seed=seed
        )

    else:
        qtys = [rng.randint(qlow, qhigh) for _ in range(M)]
        caps = [rng.randint(Qlow, Qhigh) for _ in range(K)]
        dmat = random_distance_matrix(
            n_nodes, low=low, high=high, lmbd=lmbd, asymmetric=True, seed=seed
        )

    return ShareARideProblem(N, M, K, qtys, caps, dmat)


def generate_instance_coords(
        N: int, M: int, K: int,     # pylint: disable=invalid-name
        area: int = 100,
        qlow: int = 5, qhigh: int = 15, qlmbd: float = 10.0,
        Qlow: int = 20, Qhigh: int = 45, Qlmbd: float = 30.0,   # pylint: disable=invalid-name
        seed: Optional[int] = None,
    ) -> ShareARideProblem:

    """
    Generate instance with coordinates and optional visualization with matplotlib.
    """

    rng = random.Random(seed)
    total_points = 1 + 2 * N + 2 * M

    # Generate coordinates for depot and request points, ensuring no overlaps
    coords: List[Tuple[int, int]] = [(area // 2, area // 2)]
    used_coords: set[Tuple[int, int]] = {(area // 2, area // 2)}
    while len(coords) < total_points:
        new_coord = (
            round(rng.random() * area + 0.5),
            round(rng.random() * area + 0.5)
        )
        if new_coord not in used_coords:
            coords.append(new_coord)
            used_coords.add(new_coord)

    # Containers
    dmat = euclidean_distance_matrix(coords)
    qtys = [_sample_poisson(rng, qlow, qhigh, qlmbd) for _ in range(M)]
    caps = [_sample_poisson(rng, Qlow, Qhigh, Qlmbd) for _ in range(K)]

    return ShareARideProblem(N, M, K, qtys, caps, dmat, coords)




# ================ Playground ================
if __name__ == "__main__":
    # Example usage
    problem: ShareARideProblem = generate_instance_coords(N=3, M=7, K=2, seed=123)
    problem.stdin_print()
