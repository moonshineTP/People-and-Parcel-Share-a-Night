"""
Module to parse SARP instance files into ShareARideProblem objects.
There are two main formats:
- .inp files: simplified input format, used in the Hustack platform
- .sarp files: detailed SARP format described in the data documentation.
"""
import os
import re
from typing import Dict, Any

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.data.classes import Dataset
from share_a_ride.data.router import path_router

# Type alias for SARP instance dictionary
Instance = Dict[str, Any]




def _parse_dotsarp_to_instance(file_path: str) -> ShareARideProblem:
    """
    Parse a SARP instance file (relative path ending at .sarp)
    and return a ShareARideProblem object.
    """

    # //// Read and preprocess file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('//')]


    # //// Parse from lines
    instance = {}
    lineid = 0

    while lineid < len(lines):
        line = lines[lineid]

        # Parse header fields
        if line.startswith('NAME'):
            instance['name'] = line.split(':', 1)[1].strip()
        elif line.startswith('COMMENT'):
            instance['comment'] = line.split(':', 1)[1].strip()
        elif line.startswith('TYPE'):
            instance['type'] = line.split(':', 1)[1].strip()
        elif line.startswith('DIMENSION'):
            instance['dimension'] = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            instance['uniform_capacity'] = int(line.split(':')[1].strip())
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
            instance['distance_matrix'] = dist_matrix


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
            instance['coords_dict'] = coords_dict
            instance['node_order'] = node_order


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
            instance['node_types'] = node_types


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
            instance['passenger_pairs'] = passenger_pairs
            instance['parcel_pairs'] = parcel_pairs


        # Parse VEHICLE_CAPACITY_SECTION
        elif line == 'VEHICLE_CAPACITY_SECTION':
            lineid += 1
            capacities = []
            while lineid < len(lines) and lines[lineid] != 'END_VEHICLE_CAPACITY_SECTION':
                parts = lines[lineid].split()
                capacity = int(parts[2])
                capacities.append(capacity)
                lineid += 1
            instance['vehicle_capacities'] = capacities


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
            instance['parcel_quantities_dict'] = parcel_quantities_dict


        # Parse DEPOT_SECTION
        elif line == 'DEPOT_SECTION':
            lineid += 1
            while lineid < len(lines) and lines[lineid] != 'END_DEPOT_SECTION':
                instance['depot'] = int(lines[lineid])
                lineid += 1

        lineid += 1


    # //// Extract problem parameters
    # Extract counts
    N = len(instance['passenger_pairs'])        # pylint: disable=invalid-name
    M = len(instance['parcel_pairs'])           # pylint: disable=invalid-name
    K = len(instance['vehicle_capacities'])     # pylint: disable=invalid-name

    # Reorder nodes to canonical order: depot - ppick - lpick - pdrop - ldrop
    # 1. Identify node IDs for each category
    depot_id = instance.get('depot', 1)
    ppick_ids = [p[0] for p in instance['passenger_pairs']]
    lpick_ids = [p[0] for p in instance['parcel_pairs']]
    pdrop_ids = [p[1] for p in instance['passenger_pairs']]
    ldrop_ids = [p[1] for p in instance['parcel_pairs']]

    canonical_ids = [depot_id] + ppick_ids + lpick_ids + pdrop_ids + ldrop_ids

    # 2. Map from node ID to original index in the distance matrix
    # We assume the original order is the order in which nodes appeared in NODE_COORD_SECTION
    node_order = instance.get('node_order', list(range(1, instance.get('dimension', 0) + 1)))
    id_to_orig_idx = {node_id: i for i, node_id in enumerate(node_order)}

    # 3. Reorder distance matrix
    old_dist = instance['distance_matrix']
    dist_matrix = []
    for i_id in canonical_ids:
        row = []
        i_orig = id_to_orig_idx[i_id]
        for j_id in canonical_ids:
            j_orig = id_to_orig_idx[j_id]
            row.append(old_dist[i_orig][j_orig])
        dist_matrix.append(row)

    # 4. Reorder coords
    coords_dict = instance.get('coords_dict', {})
    coords = []
    for node_id in canonical_ids:
        coords.append(coords_dict.get(node_id, (0.0, 0.0)))     # Default to (0.0, 0.0) if missing

    # 5. Reorder parcel quantities
    pq_dict = instance.get('parcel_quantities_dict', {})
    parcel_quantities = [pq_dict.get(node_id, 0) for node_id in lpick_ids]

    vehicle_capacities = instance['vehicle_capacities']


    # //// Create ShareARideProblem instance
    problem = ShareARideProblem(
        N=N,
        M=M,
        K=K,
        parcel_qty=parcel_quantities,
        vehicle_caps=vehicle_capacities,
        dist=dist_matrix,
        coords=coords
    )

    return problem




def _parse_inp_to_instance(content: str) -> ShareARideProblem:
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

    for _ in range(num_nodes):
        if lineid >= len(lines):
            raise ValueError("Insufficient lines for distance matrix")
        row = list(map(int, lines[lineid].split()))
        dist.append(row)
        lineid += 1

    return ShareARideProblem(N, M, K, q, Q, dist)




def parse_dataset(dataset: Dataset) -> Dict[str, ShareARideProblem]:
    """
    Process a dataset enum object to route, scrape and pass all .sarp instance of
    the dataset into ShareARideProblem instances.

    Returns a list of ShareARideProblem objects.
    """
    dataset_path = path_router(dataset.value.name, "readall")
    instances = {}

    filenames = os.listdir(dataset_path)
    # Sort filenames in natural order (numerical order for numbers in string)
    filenames.sort(key=lambda f: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', f)])

    for filename in filenames:
        if filename.endswith(".sarp"):
            inst_name = os.path.splitext(filename)[0]
            file_path = dataset_path + filename
            problem = _parse_dotsarp_to_instance(file_path)
            instances[inst_name] = problem

    return instances




# ================ Playground ================
if __name__ == "__main__":
    dts = Dataset.EXACT
    problist = parse_dataset(dts)
    for name, prob in problist.items():
        print(f"Instance: {name}, N={prob.N}, M={prob.M}, K={prob.K}")
