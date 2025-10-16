import random, os

from typing import Any, Dict, List, Tuple
from share_a_ride.precurated.utils import infer_type, text2lines, parse_distances

# Type alias for instance dict
Instance = Dict[str, Any]



# ...existing code...
def _is_valid_vrplib_instance(lines: List[str]) -> None:
    """
    Lightweight VRPLIB sanity check.
    Ensures required specs appear before the first section and that required
    section headers (with matching EOF markers) are present.
    Does NOT validate section contents.
    """
    if not lines:
        raise RuntimeError("Empty VRPLIB file.")

    i = 0
    n = len(lines)

    # Gather specs until the first *_SECTION or EOF
    specs: Dict[str, str] = {}
    while i < n:
        raw = lines[i].strip()
        if not raw or raw.startswith("#"):
            i += 1
            continue
        if raw == "EOF" or raw.endswith("_SECTION"):
            break
        if ":" not in raw:
            raise RuntimeError(f"Malformed specification line: '{raw}'")
        key, value = raw.split(":", 1)
        specs[key.strip().lower()] = value.strip()
        i += 1

    for key in ("name", "type", "dimension", "edge_weight_type", "capacity"):
        if key not in specs:
            raise RuntimeError(f"Missing '{key.upper()}' specification.")

    # Check for edge_weight_type validity
    ew_type = specs["edge_weight_type"].lower()
    if ew_type not in ("explicit", "euc_2d"):
        raise RuntimeError("EDGE_WEIGHT_TYPE must be EXPLICIT or EUC_2D.")
    if ew_type == "explicit":
        if "edge_weight_format" not in specs:
            raise RuntimeError("Missing 'EDGE_WEIGHT_FORMAT' specification.")
        if specs["edge_weight_format"].lower() != "full_matrix":
            raise RuntimeError("Only 'FULL_MATRIX' EDGE_WEIGHT_FORMAT is supported.")

    # Required sections
    required_sections = {"NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"}
    if ew_type == "explicit":
        required_sections.add("EDGE_WEIGHT_SECTION")

    # Scan remaining lines for section headers and their EOF markers
    seen_sections: Dict[str, bool] = {sec: False for sec in required_sections}
    while i < n:
        raw = lines[i].strip()

        # Skip empty/comment lines
        if not raw or raw.startswith("#"):
            i += 1
            continue

        if raw == "EOF":
            break
        if raw.endswith("_SECTION"):
            section = raw
            if section in seen_sections:
                seen_sections[section] = True

            # Walk through the section body until the next header / EOF
            i += 1
            while i < n:
                nxt = lines[i].strip()
                if not nxt:
                    i += 1
                    continue
                if nxt == "EOF":
                    break
                if nxt.endswith("_SECTION"):
                    break
                i += 1
            continue

        raise RuntimeError(f"Unexpected line outside sections: '{raw}'")


    # Ensure all required sections were seen
    missing = [sec for sec, seen in seen_sections.items() if not seen]
    if missing:
        raise RuntimeError(f"Missing required sections: {', '.join(missing)}")

    # Optional: ensure final EOF line exists
    if "EOF" not in (line.strip() for line in lines):
        raise RuntimeError("Missing EOF terminator.")



def _parse_vrplib(text: str) -> Instance:
    """
    Parser for VRPLIB instances (no numpy) to a dict representing an instance.
    This instance contains:
    - specs: name, type, dimension, edge_weight_type, capacity
    - sections: edge_weight (if present), node_coord, demand, depot
    """
    lines = text2lines(text)
    _is_valid_vrplib_instance(lines)

    inst: Instance = {}
    i = 0
    n = len(lines)

    # ================ Parse specs ================
    while i < n and "_SECTION" not in lines[i] and lines[i] != "EOF":
        line = lines[i].strip()
        if line and not line.startswith("#") and ":" in line:
            k, v = line.split(":", 1)
            inst[k.strip().lower()] = infer_type(v.strip())

        i += 1

    # ================ Parse sections ================
    def is_section_header(idx: int) -> bool:
        """
        Check if lines[idx] is a section header or EOF.
        """
        if idx >= n:
            return False
        val = lines[idx].strip()
        res = val and not val.startswith("#") \
            and (val == "EOF" or val.split()[0].endswith("_SECTION"))

        return res


    def read_block(start: int) -> Tuple[str, int, List[str]]:
        """
        Read a block starting at lines[start], which is a section header.
        Returns: (token, next_index, body_lines)
        """
        header_line = lines[start].strip()
        token = header_line.split()[0]
        body: List[str] = []
        idx = start + 1
        while idx < n and not is_section_header(idx):
            row = lines[idx].strip()
            if row and not row.startswith("#"):
                body.append(row)
            idx += 1
        return token, idx, body


    while i < n:
        header = lines[i].strip()
        if not header or header.startswith("#"):
            i += 1
            continue
        if header == "EOF":
            break

        token, next_i, body = read_block(i)
        if not token.endswith("_SECTION"):
            raise RuntimeError(f"Unexpected line outside sections: '{header}'")

        if token == "EDGE_WEIGHT_SECTION":
            if str(inst.get("edge_weight_type", "")).lower() != "explicit":
                raise ValueError(
                    "EDGE_WEIGHT_SECTION present " \
                    "but EDGE_WEIGHT_TYPE is not 'EXPLICIT'"
                )

            inst["edge_weight"] = [
                [int(infer_type(x)) for x in row.split()] for row in body
            ]

        elif token == "NODE_COORD_SECTION":
            coords: List[Tuple[float, float]] = []
            for row in body:
                parts = row.split()
                if len(parts) < 3:
                    raise ValueError(
                        "Each NODE_COORD_SECTION line " \
                        "must have at least 3 values"
                    )
                coords.append((float(parts[1]), float(parts[2])))

            inst["node_coord"] = coords

        elif token == "DEMAND_SECTION":
            inst["demand"] = [int(parts[1])
                for parts in (row.split() for row in body)
            ]

        elif token == "DEPOT_SECTION":
            depots = [int(row) for row in body if row != "-1"]
            if len(depots) != 1:
                raise ValueError("DEPOT_SECTION must contain exactly one depot index")
            inst["depot"] = depots[0]

        # Ignore other sections if present
        i = next_i


    # sanity
    for key in ("node_coord", "demand", "depot"):
        if key not in inst:
            raise RuntimeError(f"Missing '{key}' section in VRPLIB instance.")

    # compute distances when needed
    if str(inst["edge_weight_type"]).lower() == "euc_2d":
        inst["edge_weight"] = parse_distances("EUC_2D", inst["node_coord"])

    return inst



def _curate_vrplib_to_sarp_text(
        inst: Instance,
        num_vehicles: int,
        request_ratio: float = 0.5,
        capacity_factor: float = 1.0,
        seed: int = 42
    ) -> str:
    """
    Curate a parsed VRPLIB instance into SARP text
    Params:
    - inst: parsed VRPLIB instance dict (from _parse_vrplib)
    - request_ratio: fraction of requests to be passengers (0.0 to 1.0)
    - capacity_factor: control the fluctuation of vehicle capacity (centered at 1.0)
    - seed: random seed for reproducibility
    Returns: 
    - text of the curated .sarp file
    """


    # ============= Validate and extract source instance data =============
    # Check parameters
    if not (1e-5 < request_ratio < 1.0 - 1e-5):
        raise ValueError("request_ratio must be in (0, 1)")
    if capacity_factor < 1e-5:
        raise ValueError("capacity_factor must be positive")
    if num_vehicles < 1:
        raise ValueError("num_vehicles must be at least 1")

    # Random generator
    rng = random.Random(seed)

    # Extract nodes info
    num_nodes = int(inst["dimension"])
    coords_src = [(coords, idx) for (idx, (coords)) in enumerate(inst["node_coord"])]

    # ensure odd dimension so 2N+2M+1 fits
    if num_nodes % 2 == 0:
        num_nodes -= 1
        coords_src = coords_src[:num_nodes]

    # Split requests
    num_requests = (num_nodes - 1) // 2
    N = max(1, int(num_requests * request_ratio + 0.5))
    M = max(1, num_requests - N)
    K = num_vehicles

    # Update target parameters
    num_nodes = 2 * N + 2 * M + 1
    num_requests = N + M
    coords_src = coords_src[:num_nodes]

    # depot
    dep = inst["depot"] - 1  # 0-based depot index
    if dep < 0 or dep >= num_nodes:
        raise ValueError("Invalid depot index")

    # Select nodes (keep depot fixed, shuffle only the remainder)
    chosen_indices = [idx for _, idx in coords_src if idx != dep]
    rng.shuffle(chosen_indices)
    chosen_indices = [dep] + chosen_indices[:num_nodes - 1]
    coords = [inst["node_coord"][idx] for idx in chosen_indices]

    if inst["edge_weight_type"].lower() == "explicit":
        D_full = inst["edge_weight"]
        D = [[D_full[i][j] for j in chosen_indices] for i in chosen_indices]
    else:
        D = parse_distances("EUC_2D", coords)

    # Capacity and parcel quantity
    q = [inst["demand"][idx] for idx in chosen_indices[N + 1: N + M + 1]]
    factor_low = min(capacity_factor, 1 / capacity_factor)
    factor_high = max(capacity_factor, 1 / capacity_factor)
    shrinked_capacity = int(inst["capacity"] * (M / num_requests) + 0.5)
    Qlow = int(factor_low * shrinked_capacity + 1)
    Qhigh = int(factor_high * shrinked_capacity + 1)
    Q = [rng.randint(Qlow, Qhigh) for _ in range(K)]

    # Node types
    node_types: List[Tuple[int, int, int]] = []
    node_types.append((1, 1, 0))
    for idx in range(N):
        nid = idx + 2
        node_types.append((nid, nid, 1))
    for idx in range(M):
        nid = N + idx + 2
        node_types.append((nid, nid, 2))
    for idx in range(N):
        nid = N + M + idx + 2
        node_types.append((nid, nid, 3))
    for idx in range(M):
        nid = 2 * N + M + idx + 2
        node_types.append((nid, nid, 4))

    # Node pairs
    pairs: List[Tuple[int, int, str, int]] = []
    for pid in range(1, N + 1):
        pick = pid + 1
        drop = N + M + pid + 1
        pairs.append((pid, pick, "P", drop))
    for lid in range(1, M + 1):
        pickup_node = N + lid + 1
        drop_node = 2 * N + M + lid + 1
        pairs.append((N + lid, pickup_node, "L", drop_node))



    # ============= Write SARP text =============
    lines: List[str] = []

    # specs
    name = str(inst["name"]).replace(" ", "_")
    lines.append(f"NAME : {name}_SARP")
    lines.append("COMMENT : Curated from VRPLIB to SARP")
    lines.append("TYPE : SARP")
    lines.append(f"DIMENSION : {num_nodes}")
    lines.append("EDGE_WEIGHT_TYPE : EXPLICIT")
    lines.append("EDGE_WEIGHT_FORMAT : FULL_MATRIX")
    lines.append("EDGE_WEIGHT_SECTION")
    for row in D:
        lines.append(" ".join(str(int(v)) for v in row))
    lines.append("EOF_EDGE_WEIGHT_SECTION")
    lines.append("")

    # coords
    lines.append("NODE_COORD_SECTION")
    for idx, (x, y) in enumerate(coords, start=1):
        lines.append(f"{idx} {float(x)} {float(y)}")
    lines.append("EOF_NODE_COORD_SECTION")
    lines.append("")

    # node types
    lines.append("NODE_TYPE_SECTION")
    for sid, nid, typ in node_types:
        lines.append(f"{sid} {nid} {typ}")
    lines.append("EOF_NODE_TYPE_SECTION")
    lines.append("")

    # pairs
    lines.append("PAIR_SECTION")
    for rid, pick, cat, drop in pairs:
        lines.append(f"{rid} {pick} {cat} {drop}")
    lines.append("EOF_PAIR_SECTION")
    lines.append("")

    # vehicle capacities
    lines.append("VEHICLE_CAPACITY_SECTION")
    for vid, cap in enumerate(Q, start=1):
        lines.append(f"{vid} {vid} {cap}")
    lines.append("EOF_VEHICLE_CAPACITY_SECTION")
    lines.append("")

    # parcel quantities
    lines.append("PARCEL_QUANTITY_SECTION")
    for pid, qty in enumerate(q, start=1):
        pickup_node = N + pid + 1
        lines.append(f"{pid} {pickup_node} {qty}")
    lines.append("EOF_PARCEL_QUANTITY_SECTION")
    lines.append("")

    # depot
    lines.append("DEPOT_SECTION")
    lines.append("1")
    lines.append("EOF_DEPOT_SECTION")
    lines.append("")
    lines.append("EOF")

    return "\n".join(lines)



def write_sarp_from_vrplib(
        dataset_dir: str, vrplib_file: str,
        sarp_path: str, purpose: str
    ) -> None:
    """
    Read a VRPLIB file and write a curated .sarp file.
    kwargs are passed to curate_vrplib_to_sarp_text.
    """
    vrplib_path = "share_a_ride/precurated/" + dataset_dir + "/" + vrplib_file
    sarp_path = "share_a_ride/data/" + purpose + "/" + dataset_dir + "/" + sarp_path

    with open(vrplib_path, "r", encoding="utf-8") as f:
        vrplib_text = f.read()

    inst = _parse_vrplib(vrplib_text)
    sarp_text = _curate_vrplib_to_sarp_text(
        inst, num_vehicles=5, request_ratio=0.5,
        capacity_factor=1.2, seed=42
    )

    os.makedirs(os.path.dirname(sarp_path), exist_ok=True)
    with open(sarp_path, "w", encoding="utf-8") as f:
        f.write(sarp_text)

    print(f"Wrote curated SARP instance {vrplib_file} to {sarp_path}")



def curate_vrplib_dataset(dataset_dir: str, purpose: str) -> None:
    """
    Curate all VRPLIB instances in a directory to .sarp files.
    Params:
    - dataset_dir: name of the dataset folder under share_a_ride/precurated/
        (e.g. "Cvrplib", "Pyvrp", "Golden", "tai")
    - purpose: name of the dataset folder under share_a_ride/data/
        (e.g. "sanity", "train", "val", "test" or "benchmark")
    """
    for _, _, files in os.walk("share_a_ride/precurated/" + dataset_dir):
        for vrplib_file in files:
            # Ignore if the path containing "README"
            if "README" in vrplib_file:
                continue

            if not vrplib_file.endswith(".vrp"):
                continue

            sarp_file = vrplib_file.replace(".vrp", ".sarp")
            write_sarp_from_vrplib(dataset_dir, vrplib_file, sarp_file, purpose)


if __name__ == "__main__":
    curate_vrplib_dataset("Cvrplib", "train")
    curate_vrplib_dataset("CMT", "val")
    curate_vrplib_dataset("tai", "test")

    curate_vrplib_dataset("Pyvrp", "benchmark")
    curate_vrplib_dataset("Golden", "benchmark")
    curate_vrplib_dataset("Li", "benchmark")
