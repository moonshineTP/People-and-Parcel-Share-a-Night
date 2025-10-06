import math
import random
from typing import Optional, List, Tuple

from share_a_ride.problem import ShareARideProblem
from utils.visualization import visualize_instance


def _generate_symmetric_value(
        i: int, j: int, D: List[List[int]], rng: random.Random,
        low: int, high: int, asymmetric: bool
    ) -> int:
    """Generate a single distance value, symmetric if needed."""
    if i == j:
        return 0
    if asymmetric:
        return rng.randint(low, high)
    if j < i:
        return D[j][i]
    return rng.randint(low, high)


def _sample_poisson(
        lmbd: float,
        rng: random.Random
    ) -> int:
    """Approximate Poisson distribution using Knuth's algorithm."""
    L = math.exp(-lmbd)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return max(1, k - 1)


# ---------------------- Test instance generators --------------------------------
def random_distance_matrix(
        n: int,
        low: int = 3,
        high: int = 9,
        asymmetric: bool = False,
        seed: int | None = None
    ) -> List[List[int]]:
    """
    Generate a random symmetric or asymmetric distance matrix.
    """
    rng = random.Random(seed)
    D = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            D[i][j] = _generate_symmetric_value(i, j, D, rng, low, high, asymmetric)

    return D


def euclidean_distance_matrix(
        coords: List[Tuple[float, float]]
    ) -> List[List[int]]:
    """Compute pairwise Euclidean distance matrix from coordinates."""
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
        low: int = 5, high: int = 20,
        qlow: int = 5, qhigh: int = 15,
        Qlow: int = 15, Qhigh: int = 30,
        use_poisson: bool = False, poisson_lambda: float = 10.0,
        seed: Optional[int] = None
    ) -> ShareARideProblem:
    """Generate random instance using lazy distance matrix."""
    rng = random.Random(seed)

    if use_poisson:
        q = [_sample_poisson(poisson_lambda, rng) for _ in range(M)]
    else:
        q = [rng.randint(qlow, qhigh) for _ in range(M)]

    Q = [rng.randint(Qlow, Qhigh) for _ in range(K)]
    n_nodes = 2 * N + 2 * M + 1
    D = random_distance_matrix(n_nodes, low=low, high=high,
                               asymmetric=False, seed=seed)
    return ShareARideProblem(N, M, K, q, Q, D)


def generate_instance_coords(
        N: int, M: int, K: int,
        area: float = 20.0,
        qlow: int = 5, qhigh: int = 15,
        Qlow: int = 15, Qhigh: int = 30,
        seed: Optional[int] = None,
        visualize: bool = False
    ) -> ShareARideProblem:
    """Generate instance with coordinates and optional visualization."""
    rng = random.Random(seed)
    total_points = 1 + 2 * N + 2 * M

    # Generate depot and random coordinates
    coords = [(area / 2.0, area / 2.0)] + [
        (rng.random() * area, rng.random() * area)
        for _ in range(total_points - 1)
    ]

    D = euclidean_distance_matrix(coords)
    q = [rng.randint(qlow, qhigh) for _ in range(M)]
    Q = [rng.randint(Qlow, Qhigh) for _ in range(K)]
    prob = ShareARideProblem(N, M, K, q, Q, D)

    if visualize:
        visualize_instance(coords, N, M, K)

    return prob