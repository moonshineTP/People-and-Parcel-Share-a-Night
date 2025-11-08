"""
    Utility functions for generating simple, random Share-a-Ride instances.
    Use for the early stages of algorithm development and testing.
"""

import math
import random
from typing import Optional
from share_a_ride.core.problem import ShareARideProblem


def _generate_cost_value(
        i: int, j: int, D: list[list[int]], rng: random.Random,
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
        return D[j][i]
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
    Sample from Poisson(lmbd) until result is in [low, high].
    """

    while True:
        # Parameters
        L = math.exp(-lmbd)
        k = 0
        p = 1.0

        # Generate Poisson sample using Knuth's algorithm
        while p > L:
            k += 1
            p *= rng.random()
        value = k - 1

        # Check range and return
        if low <= value <= high:
            return value


# ---------------------- Test instance generators --------------------------------
def random_distance_matrix(
        n: int,
        low: int = 5,
        high: int = 20,
        lmbd: float = 10.0,
        asymmetric: bool = False,
        seed: int = 42,
    ) -> list[list[int]]:
    """
    Generate a random symmetric or asymmetric distance matrix.
    """
    rng = random.Random(seed)
    D = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            D[i][j] = _generate_cost_value(i, j, D, rng, low, high, lmbd, asymmetric)

    return D


def euclidean_distance_matrix(
        coords: list[tuple[int, int]]
    ) -> list[list[int]]:
    """
    Compute pairwise Euclidean distance matrix from coordinates, 
    rounding distances to the nearest integer.
    """
    n = len(coords)
    D = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            dist = int(round(math.hypot(
                coords[i][0] - coords[j][0],
                coords[i][1] - coords[j][1]
            )))
            D[i][j] = D[j][i] = dist

    return D


def generate_instance_lazy(
        N: int, M: int, K: int,
        low: int = 10, high: int = 50, lmbd: float = 20.0,
        qlow: int = 5, qhigh: int = 15, qlmbd: float = 10.0,
        Qlow: int = 15, Qhigh: int = 30, Qlmbd: float = 20.0,
        use_poisson: bool = False,
        seed: int = 42
    ) -> ShareARideProblem:
    """
    Generate random instance using lazy distance matrix.
    """
    rng = random.Random(seed)
    n_nodes = 2*N + 2*M + 1

    if use_poisson:
        q = [_sample_poisson(rng, qlow, qhigh, qlmbd) for _ in range(M)]
        Q = [_sample_poisson(rng, Qlow, Qhigh, Qlmbd) for _ in range(K)]
        D = random_distance_matrix(n_nodes, low=low, high=high, lmbd=lmbd,
                               asymmetric=True, seed=seed)

    else:
        q = [rng.randint(qlow, qhigh) for _ in range(M)]
        Q = [rng.randint(Qlow, Qhigh) for _ in range(K)]
        D = random_distance_matrix(n_nodes, low=low, high=high, lmbd=lmbd,
                               asymmetric=True, seed=seed)

    return ShareARideProblem(N, M, K, q, Q, D)


def generate_instance_coords(
        N: int, M: int, K: int,
        area: int = 100,
        qlow: int = 5, qhigh: int = 15, qlmbd: float = 10.0,
        Qlow: int = 20, Qhigh: int = 45, Qlmbd: float = 30.0,
        seed: int = 42,
    ) -> ShareARideProblem:

    """
    Generate instance with coordinates and optional visualization with matplotlib.
    """
    rng = random.Random(seed)
    total_points = 1 + 2 * N + 2 * M

    # Generate depot and random coordinates, ensuring no overlaps
    coords: list[tuple[int, int]] = [(area // 2, area // 2)]
    used_coords: set[tuple[int, int]] = {(area // 2, area // 2)}

    while len(coords) < total_points:
        new_coord = (
            round(rng.random() * area + 0.5), 
            round(rng.random() * area + 0.5)
        )
        if new_coord not in used_coords:
            coords.append(new_coord)
            used_coords.add(new_coord)

    D = euclidean_distance_matrix(coords)
    q = [rng.randint(qlow, qhigh) for _ in range(M)]
    Q = [rng.randint(Qlow, Qhigh) for _ in range(K)]
    prob = ShareARideProblem(N, M, K, q, Q, D, coords)

    return prob


if __name__ == "__main__":
    # Example usage
    problem: ShareARideProblem = generate_instance_coords(N=3, M=7, K=2, seed=123)
    problem.stdin_print()