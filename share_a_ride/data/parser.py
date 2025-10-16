"""
Module to parse SARP instance files into ShareARideProblem objects.
"""

import os

from typing import Dict, Any

from share_a_ride.problem import ShareARideProblem

Instance = Dict[str, Any]


def parse_sarp_to_problem(file_path: str) -> ShareARideProblem:
    """
    Parse a SARP instance file and return a ShareARideProblem object.
    The file path should be absolute and have .sarp extension.
    """
    if not os.path.isabs(file_path) or not file_path.endswith('.sarp'):
        raise ValueError("file_path must be an absolute path to a .sarp file.")

    instance = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('//')]

    i = 0
    while i < len(lines):
        line = lines[i]

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
            i += 1
            dist_matrix = []
            while i < len(lines) and lines[i] != 'EOF_EDGE_WEIGHT_SECTION':
                row = list(map(int, lines[i].split()))
                dist_matrix.append(row)
                i += 1
            instance['distance_matrix'] = dist_matrix

        # Parse NODE_COORD_SECTION
        elif line == 'NODE_COORD_SECTION':
            i += 1
            coords = []
            while i < len(lines) and lines[i] != 'EOF_NODE_COORD_SECTION':
                parts = lines[i].split()
                coords.append((float(parts[1]), float(parts[2])))
                i += 1
            instance['coords'] = coords

        # Parse NODE_TYPE_SECTION
        elif line == 'NODE_TYPE_SECTION':
            i += 1
            node_types = {}
            while i < len(lines) and lines[i] != 'EOF_NODE_TYPE_SECTION':
                parts = lines[i].split()
                node_id = int(parts[1])
                node_type = int(parts[2])
                node_types[node_id] = node_type
                i += 1
            instance['node_types'] = node_types

        # Parse PAIR_SECTION
        elif line == 'PAIR_SECTION':
            i += 1
            passenger_pairs = []
            parcel_pairs = []
            while i < len(lines) and lines[i] != 'EOF_PAIR_SECTION':
                parts = lines[i].split()
                pickup = int(parts[1])
                category = parts[2]
                dropoff = int(parts[3])
                if category == 'P':
                    passenger_pairs.append((pickup, dropoff))
                elif category == 'L':
                    parcel_pairs.append((pickup, dropoff))
                i += 1
            instance['passenger_pairs'] = passenger_pairs
            instance['parcel_pairs'] = parcel_pairs

        # Parse VEHICLE_CAPACITY_SECTION
        elif line == 'VEHICLE_CAPACITY_SECTION':
            i += 1
            capacities = []
            while i < len(lines) and lines[i] != 'EOF_VEHICLE_CAPACITY_SECTION':
                parts = lines[i].split()
                capacity = int(parts[2])
                capacities.append(capacity)
                i += 1
            instance['vehicle_capacities'] = capacities

        # Parse PARCEL_QUANTITY_SECTION
        elif line == 'PARCEL_QUANTITY_SECTION':
            i += 1
            parcel_quantities = []
            while i < len(lines) and lines[i] != 'EOF_PARCEL_QUANTITY_SECTION':
                parts = lines[i].split()
                quantity = int(parts[2])
                parcel_quantities.append(quantity)
                i += 1
            instance['parcel_quantities'] = parcel_quantities

        i += 1

    # Extract problem parameters
    N = len(instance['passenger_pairs'])
    M = len(instance['parcel_pairs'])
    K = len(instance['vehicle_capacities'])

    # Convert from 1-based to 0-based indexing for internal use
    dist_matrix = instance['distance_matrix']
    coords = instance['coords']
    parcel_quantities = instance['parcel_quantities']
    vehicle_capacities = instance['vehicle_capacities']

    # Create ShareARideProblem instance
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



if __name__ == "__main__":
    prob = parse_sarp_to_problem("share_a_ride/data/sanity/H/H-n10-m10-k5.sarp")
    prob.pretty_print(verbose=1)

