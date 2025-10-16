import os
import math
import random

from typing import Any, Dict, Tuple, List
from share_a_ride.precurated.utils import text2lines, parse_distances


Instance = Dict[str, Any]



def _is_valid_solomon_instance(lines: list[str]):
    """
    Checks if the passed-in lines follow the Solomon format requirements.
    """

    try:
        assert lines[0]  # non-empty first line
        assert "VEHICLE" in lines[1]
        assert "NUMBER" in lines[2]
        assert "CAPACITY" in lines[2]
        assert "CUSTOMER" in lines[4]

        # Header names are separated on whitespace because the spacing of
        # some Solomon instances is off.
        headers = [
            "CUST",
            "NO.",
            "XCOORD.",
            "YCOORD.",
            "DEMAND",
            "READY",
            "DUE",
            "DATE",
            "SERVICE",
            "TIME",
        ]
        for header in headers:
            assert header in lines[5]

    except (IndexError, ValueError, AssertionError) as err:
        msg = "Instance does not conform to the Solomon format."
        raise RuntimeError(msg) from err



def _parse_solomon(text: str, compute_edge_weights: bool = True) -> Instance:
    """
    Parses the text of a Solomon VRPTW instance.
    Params:
    - text: the full text of a Solomon instance
    - compute_edge_weights: if True, computes pairwise Euclidean distance matrix
    Returns: a dict representing the instance
    """

    lines = text2lines(text)
    _is_valid_solomon_instance(lines)    # Validate format

    # Parse header
    instance: Instance = {"name": lines[0]}
    instance["vehicles"], instance["capacity"] = [
        int(num) for num in lines[3].split()
    ]

    # Parse data lines
    rows: list[list[int]] = []
    for line in lines[6:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        ints: list[int] = []
        for tok in parts:
            try:
                ints.append(int(tok))
            except ValueError:
                pass
        if len(ints) != 7:
            raise RuntimeError("Each data line must have exactly 7 integer values.")

        rows.append(ints)

    # Extract columns
    instance["node_coord"]      = [(r[1], r[2]) for r in rows]
    instance["demand"]          = [r[3] for r in rows]
    instance["time_window"]     = [(r[4], r[5]) for r in rows]
    instance["service_time"]    = [r[6] for r in rows]

    if compute_edge_weights:
        coords = instance["node_coord"]
        n = len(coords)
        edge_weight = [[0.0] * n for _ in range(n)]
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                d = math.hypot(xi - xj, yi - yj)
                edge_weight[i][j] = d
                edge_weight[j][i] = d
        instance["edge_weight"] = edge_weight

    return instance



def _curate_solomon_to_sarp_text(
        inst: Instance,
        request_ratio: float = 0.5,
        capacity_factor: float = 1.0,
        seed: int = 42,
    ) -> str:
    """
    Curate a parsed Solomon instance into SARP text
    Params:
    - inst: parsed Solomon instance dict (from _parse_solomon)
    - request_ratio: fraction of requests to be passengers (0.0 to 1.0)
    - capacity_factor: control the fluctuation of vehicle capacity (centered at 1.0)
    - seed: random seed for reproducibility
    Returns: 
    - text of the curated .sarp file
    """

    if (capacity_factor < 1e-5):
        raise ValueError("capacity_factor must be positive")
    if (request_ratio < 1e-5) or (request_ratio > 1 - 1e-5):
        raise ValueError("request_ratio must be between 0.0 and 1.0")


    include_coords = True  # keep coords for visualization
    rng = random.Random(seed)

    # Source data
    coords_src: List[Tuple[int | float, int | float]] = inst["node_coord"]
    num_nodes = len(coords_src)

    # Ensure odd number of nodes so D = 2N + 2M + 1 fits
    if num_nodes % 2 == 0:
        num_nodes -= 1

    # Split requests
    num_requests = (num_nodes - 1) // 2

    # Target parameters
    N = int(num_requests * request_ratio + 0.5)
    M = max(1, num_requests - N)
    K = int(inst.get("vehicles", 1))

    num_nodes = 2 * N + 2 * M + 1
    num_requests = N + M

    # Shuffle and select nodes
    coords_arr = [[coord, idx] for idx, coord in enumerate(coords_src[1:num_nodes])]
    rng.shuffle(coords_arr)
    coords = [coords_src[0]] + [coord for coord, _ in coords_arr]
    D = parse_distances("EUC_2D", coords)

    q = [inst["demand"][idx] for _, idx in coords_arr[N + 1: N + M + 1]]
    factor_low = min(capacity_factor, 1 / capacity_factor)
    factor_high = max(capacity_factor, 1 / capacity_factor)
    shrinked_capacity = int(inst["capacity"] * (M / num_requests) + 0.5)
    Qlow = int(factor_low * shrinked_capacity + 1)
    Qhigh = int(factor_high * shrinked_capacity + 1)
    Q = [rng.randint(Qlow, Qhigh) for _ in range(K)]


    # ========= NODE_TYPE_SECTION (1-based IDs) ==========
    node_types: List[Tuple[int, int, int]] = []
    # depot
    node_types.append((1, 1, 0))
    # passenger pickups
    for i in range(N):
        pid = i + 2
        node_types.append((pid, pid, 1))
    # parcel pickups
    for j in range(M):
        lid = N + 2 + j
        node_types.append((lid, lid, 2))
    # passenger drops
    for i in range(N):
        pid = N + M + 2 + i
        node_types.append((pid, pid, 3))
    # parcel drops
    for j in range(M):
        lid = 2 * N + M + 2 + j
        node_types.append((lid, lid, 4))


    # ============ PAIR_SECTION (1-based IDs) ============
    pairs: List[Tuple[int, int, str, int]] = []
    # passengers
    for j in range(N):
        pid = j + 1
        p_pick = j + 2
        p_drop = N + M + j + 2
        pairs.append((pid, p_pick, "P", p_drop))
    # parcels
    for j in range(M):
        lid = N + j + 1
        l_pick = N + j + 2
        l_drop = 2 * N + M + j + 2
        pairs.append((lid, l_pick, "L", l_drop))

    # Compose SARP text (sections 1-based; EDGE_WEIGHT and DEPOT 0-based)
    name = inst["name"].replace(" ", "_")
    lines: List[str] = []
    lines.append(f"NAME : {name}_SARP")
    lines.append("COMMENT : Curated from Solomon to SARP")
    lines.append("TYPE : SARP")
    lines.append(f"DIMENSION : {num_nodes}")
    lines.append("EDGE_WEIGHT_TYPE : EXPLICIT")
    lines.append("EDGE_WEIGHT_FORMAT : FULL_MATRIX")
    lines.append("EDGE_WEIGHT_SECTION")
    for row in D:
        lines.append(" ".join(str(int(c)) for c in row))
    lines.append("EOF_EDGE_WEIGHT_SECTION")
    lines.append("")

    if include_coords:
        lines.append("NODE_COORD_SECTION")
        for i, (x, y) in enumerate(coords, start=1):
            lines.append(f"{i} {float(x)} {float(y)}")
        lines.append("EOF_NODE_COORD_SECTION")
        lines.append("")

    lines.append("NODE_TYPE_SECTION")
    for id, node_id, type in node_types:
        lines.append(f"{id} {node_id} {type}")
    lines.append("EOF_NODE_TYPE_SECTION")
    lines.append("")

    lines.append("PAIR_SECTION")
    for id, pnode, cat, dnode in pairs:
        lines.append(f"{id} {pnode} {cat} {dnode}")
    lines.append("EOF_PAIR_SECTION")
    lines.append("")

    lines.append("VEHICLE_CAPACITY_SECTION")
    for i, cap in enumerate(Q, start=1):
        lines.append(f"{i} {i} {cap}")
    lines.append("EOF_VEHICLE_CAPACITY_SECTION")
    lines.append("")

    lines.append("PARCEL_QUANTITY_SECTION")
    for id, qty in enumerate(q, start=1):
        pick_node = N + 1 + id
        lines.append(f"{id} {pick_node} {qty}")
    lines.append("EOF_PARCEL_QUANTITY_SECTION")
    lines.append("")

    lines.append("DEPOT_SECTION")
    lines.append("1")
    lines.append("EOF_DEPOT_SECTION")
    lines.append("")
    lines.append("EOF")

    return "\n".join(lines)



def write_sarp_from_solomon(solomon_file: str, sarp_file: str, purpose: str) -> None:
    """
    Read a Solomon file and write a curated .sarp file.
    Params:
    - solomon_file: filename of the Solomon instance 
        (preferably in share_a_ride/precurated/Solomon/)
    - sarp_file: filename to write the curated SARP instance
        (preferably in share_a_ride/data/{purpose}/Solomon/)
    - purpose: name of the dataset folder under share_a_ride/data/
        (preferably "sanity", "train", "val", "test" or "benchmark")
    """
    solomon_path = "share_a_ride/precurated/Solomon/" + solomon_file
    sarp_path = "share_a_ride/data/" + purpose + "/Solomon/" + sarp_file

    with open(solomon_path, "r", encoding="utf-8") as f:
        solomon_text = f.read()

    inst = _parse_solomon(solomon_text)
    sarp_text = _curate_solomon_to_sarp_text(inst, request_ratio=0.5, capacity_factor=1.2, seed=42)

    os.makedirs(os.path.dirname(sarp_path), exist_ok=True)
    with open(sarp_path, "w", encoding="utf-8") as f:
        f.write(sarp_text)

    print(f"Wrote curated SARP instance to {sarp_path}")


def curate_solomon_dataset(purpose: str) -> None:
    """
    Curate all Solomon instances in the share_a_ride/precurated/Solomon directory to .sarp files.
    Params:
    - purpose: name of the dataset folder under share_a_ride/data/
        (e.g. "sanity", "train", "val", "test" or "benchmark")
    """
    for _, _, files in os.walk("share_a_ride/precurated/Solomon"):
        for solomon_file in files:
            # Ignore if the path containing "README"
            if "README" in solomon_file:
                continue

            if not solomon_file.endswith(".txt"):
                continue

            sarp_file = solomon_file.replace(".txt", ".sarp")
            write_sarp_from_solomon(solomon_file, sarp_file, purpose)


if __name__ == "__main__":
    curate_solomon_dataset("benchmark")