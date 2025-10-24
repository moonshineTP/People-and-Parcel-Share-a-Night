import random, math

def text2lines(text: str) -> list[str]:
    """
    Takes a string and returns a list of non-empty, stripped lines. 
    Also removes any comment lines from the given string.
    """
    return [
        stripped
        for line in text.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]


def infer_type(s: str) -> int | float | str:
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def parse_distances(
        edge_weight_type: str,
        node_coord: list[tuple[int | float, int | float]],
    ) -> list[list[int]]:
    """
    Computes the pairwise Euclidean distance matrix for the given coordinates.
    Distances are rounded to the nearest integer (TSPLIB EUC_2D convention).
    """
    if edge_weight_type != "EUC_2D":
        raise ValueError("Edge weight type or format not supported.")

    n = len(node_coord)
    dist = [[0] * n for _ in range(n)]

    for i in range(n):
        xi, yi = node_coord[i]
        for j in range(i + 1, n):
            xj, yj = node_coord[j]
            dx = xi - xj
            dy = yi - yj
            d = (dx * dx + dy * dy) ** 0.5
            dij = int(d + 0.5)  # round to nearest integer
            dist[i][j] = dij
            dist[j][i] = dij

    return dist


def sample_poisson(
        low: int,
        high: int,
        lmbd: float,
        seed: int = 42,
    ) -> int:
    """
    Sample from Poisson(lmbd) until result is in [low, high].
    """

    rng = random.Random(seed)
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
