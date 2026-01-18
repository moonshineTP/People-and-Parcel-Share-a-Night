"""
Module to parse SARP instance files into ShareARideProblem objects.
There are two main formats:
- .inp files: simplified input format, used in the Hustack platform
- .sarp files: detailed SARP format described in the data documentation.
"""
import re
from typing import Any

# Type aliases
InstanceContent = dict[str, Any]
SolutionContent = dict[str, Any]



# ================ Instance Parsing ================
def parse_sarp_content(content: str) -> InstanceContent:
    """
    Parse .sarp file content string and return a InstanceContent (dict of problem features).
    This is the core parsing logic used by both file-based and content-based parsing.
    """

    # //// Preprocess content into lines
    lines = [
        line.strip() for line in content.splitlines()           # Strip whitespace
        if line.strip() and not line.strip().startswith('//')   # Remove empty lines and comments
    ]


    # //// Parse from lines
    instance_content: InstanceContent = {}
    lineid = 0

    while lineid < len(lines):
        line = lines[lineid]

        # Parse header fields
        if line.startswith('NAME'):
            instance_content['name'] = line.split(':', 1)[1].strip()
        elif line.startswith('COMMENT'):
            instance_content['comment'] = line.split(':', 1)[1].strip()
        elif line.startswith('TYPE'):
            instance_content['type'] = line.split(':', 1)[1].strip()
        elif line.startswith('DIMENSION'):
            instance_content['dimension'] = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            instance_content['uniform_capacity'] = int(line.split(':')[1].strip())
        elif line.startswith('EDGE_WEIGHT_TYPE'):
            pass  # Not used currently
        elif line.startswith('EDGE_WEIGHT_FORMAT'):
            pass  # Not used currently

        # Parse EDGE_WEIGHT_SECTION
        elif line == 'EDGE_WEIGHT_SECTION':
            lineid += 1
            dist_matrix = []
            while lineid < len(lines) and lines[lineid] != 'END_EDGE_WEIGHT_SECTION':
                row = list(map(int, lines[lineid].split()))
                dist_matrix.append(row)
                lineid += 1
            instance_content['distance_matrix'] = dist_matrix

        # Parse NODE_COORD_SECTION
        elif line == 'NODE_COORD_SECTION':
            lineid += 1
            coords_dict = {}
            node_order = []
            while lineid < len(lines) and lines[lineid] != 'END_NODE_COORD_SECTION':
                parts = lines[lineid].split()
                node_id = int(parts[0])
                coords_dict[node_id] = (float(parts[1]), float(parts[2]))
                node_order.append(node_id)
                lineid += 1
            instance_content['coords_dict'] = coords_dict
            instance_content['node_order'] = node_order

        # Parse NODE_TYPE_SECTION
        elif line == 'NODE_TYPE_SECTION':
            lineid += 1
            node_types = {}
            while lineid < len(lines) and lines[lineid] != 'END_NODE_TYPE_SECTION':
                parts = lines[lineid].split()
                node_id = int(parts[1])
                node_type = int(parts[2])
                node_types[node_id] = node_type
                lineid += 1
            instance_content['node_types'] = node_types

        # Parse PAIR_SECTION
        elif line == 'PAIR_SECTION':
            lineid += 1
            passenger_pairs = []
            parcel_pairs = []
            while lineid < len(lines) and lines[lineid] != 'END_PAIR_SECTION':
                parts = lines[lineid].split()
                pickup = int(parts[1])
                category = parts[2]
                dropoff = int(parts[3])
                if category == 'P':
                    passenger_pairs.append((pickup, dropoff))
                elif category == 'L':
                    parcel_pairs.append((pickup, dropoff))
                lineid += 1
            instance_content['passenger_pairs'] = passenger_pairs
            instance_content['N'] = len(passenger_pairs)
            instance_content['parcel_pairs'] = parcel_pairs
            instance_content['M'] = len(parcel_pairs)

        # Parse VEHICLE_CAPACITY_SECTION
        elif line == 'VEHICLE_CAPACITY_SECTION':
            lineid += 1
            capacities = []
            while lineid < len(lines) and lines[lineid] != 'END_VEHICLE_CAPACITY_SECTION':
                parts = lines[lineid].split()
                capacity = int(parts[2])
                capacities.append(capacity)
                lineid += 1
            instance_content['vehicle_capacities'] = capacities
            instance_content['K'] = len(capacities)

        # Parse PARCEL_QUANTITY_SECTION
        elif line == 'PARCEL_QUANTITY_SECTION':
            lineid += 1
            parcel_quantities_dict = {}
            while lineid < len(lines) and lines[lineid] != 'END_PARCEL_QUANTITY_SECTION':
                parts = lines[lineid].split()
                node_id = int(parts[1])
                quantity = int(parts[2])
                parcel_quantities_dict[node_id] = quantity
                lineid += 1
            instance_content['parcel_quantities_dict'] = parcel_quantities_dict

        # Parse DEPOT_SECTION
        elif line == 'DEPOT_SECTION':
            lineid += 1
            while lineid < len(lines) and lines[lineid] != 'END_DEPOT_SECTION':
                instance_content['depot'] = int(lines[lineid])
                lineid += 1

        lineid += 1

    return instance_content






def parse_hustack_content(content: str) -> InstanceContent:
    """
    Parse the simplified input format described in description.txt.
    """
    # Preprocess lines
    lines = content.strip().splitlines()
    lines = [l for l in lines if l.strip()]
    if not lines:
        raise ValueError("Empty input")

    # Initialize index
    lineid = 0
    def list_split():
        nonlocal lineid
        return list(map(int, lines[lineid].split()))

    # Line 1: N M K
    nmk = list_split()
    if len(nmk) != 3:
        raise ValueError("First line must contain N, M, K")
    N, M, K = nmk       # pylint: disable=invalid-name
    lineid += 1

    # Line 2: q
    q = []
    if M > 0:
        q = list_split()
        lineid += 1

    # Line 3: Q
    Q = list_split()    # pylint: disable=invalid-name
    lineid += 1

    # Line 4->2N + 2M + 4: Distance matrix
    dist = []
    num_nodes = 2*N + 2*M + 1
    for _nodeid in range(num_nodes):
        if lineid >= len(lines):
            raise ValueError("Insufficient lines for distance matrix")
        row = list(map(int, lines[lineid].split()))
        dist.append(row)
        lineid += 1


    # Build passenger and parcel pairs
    passenger_pairs = [(i, i + N + M) for i in range(1, N + 1)]
    parcel_pairs = [(i + N, i + 2 * N + M) for i in range(1, M + 1)]

    # Build parcel quantities dict
    parcel_quantities_dict = {}
    for lid in range(M):
        parcel_quantities_dict[N + lid + 1] = q[lid]

    # Build node_types based on canonical ordering
    node_types = {0: 0}
    for i in range(1, N + 1):
        node_types[i] = 1
    for i in range(N + 1, N + M + 1):
        node_types[i] = 2
    for i in range(N + M + 1, 2 * N + M + 1):
        node_types[i] = 3
    for i in range(2 * N + M + 1, 2 * N + 2 * M + 1):
        node_types[i] = 4


    # Build instance content dictionary matching parse_sarp_content keys
    return {
        'name': f"Hustack-N{N}-M{M}-K{K}",
        'comment': "From Hustack OJ",
        'type': "HUSTACK",                                  # Custom type
        'dimension': 2 * N + 2 * M + 1,                     # Total nodes
        'uniform_capacity': None,                           # Not used in this format
        'distance_matrix': dist,
        'coords_dict': {},                                  # No coords in this format
        'node_order': list(range(2 * N + 2 * M + 1)),
        'node_types': node_types,
        'N': N,
        'M': M,
        'K': K,
        'passenger_pairs': passenger_pairs,
        'parcel_pairs': parcel_pairs,
        'vehicle_capacities': Q,
        'parcel_quantities_dict': parcel_quantities_dict,
        'depot': 0
    }




# ================ Solution Parsing ================
def parse_sol_content(content: str) -> SolutionContent:
    """
    Parse .sol file content string and return a structured solution dictionary.

    The solution format supports:
    - Cost line: "Cost <value>"
    - Route lines: "Route #<id>: <node1> <node2> ..." or "Vehicle #<id>: ..."

    Returns:
        dict with keys:
        - 'cost': int | None - reported cost from the solution file
        - 'routes': list[dict] - list of route dicts with 'vehicle_id' and 'path' keys
    """
    # Preprocess lines
    lines = content.splitlines()


    # //// Parse lines
    reported_cost = None
    routes = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse Cost line
        if line.lower().startswith("cost"):
            try:
                parts = line.split()
                if len(parts) >= 2:
                    reported_cost = int(parts[1])
            except ValueError:
                pass
            continue

        # Parse Route/Vehicle lines
        if "Route" in line or "Vehicle" in line:
            # Extract route ID e.g. Route #1 -> 1
            route_match = re.search(r'#(\d+)', line)
            vehicle_id = int(route_match.group(1)) if route_match else len(routes) + 1

            # Extract stops - look for part after colon
            if ":" in line:
                path_str = line.split(":", 1)[1]
            else:
                path_str = line

            # Get all integers as the path
            path = [int(s) for s in re.findall(r'\b\d+\b', path_str)]
            if path:
                routes.append({
                    'vehicle_id': vehicle_id,
                    'path': path
                })

    # Return solution context content
    return {
        'cost': reported_cost,
        'routes': routes
    }




# ================ Playground ================
if __name__ == "__main__":
    pass
    # dts = Dataset.EXACT
    # problist = parse_sarp_dataset(dts)
    # for name, prob in problist.items():
    #     print(f"Instance: {name}, N={prob.N}, M={prob.M}, K={prob.K}")
