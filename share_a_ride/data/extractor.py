"""
Module to extract raw data and discover resources from the file system.

This module serves as the Extract layer in the ETL pipeline.
It separates extraction into two forms:
- Content: Raw structured dictionary (InstanceContent/SolutionContent)
- Core: Validated core encapsulation objects (ShareARideProblem/Solution)
"""
import os
import re

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import Solution
from share_a_ride.data.parser import (
    parse_sarp_content, parse_hustack_content, parse_sol_content,
    InstanceContent, SolutionContent
)
from share_a_ride.data.classes import Dataset
from share_a_ride.data.router import path_router




# Content conversion to ShareARideProblem object
def _content_to_problem(instance: InstanceContent) -> ShareARideProblem:
    """
    Core logic to convert parsed instance content dictionary into ShareARideProblem object.
    Performs canonical node reordering.
    """
    # Extract counts
    N, M, K = instance['N'], instance['M'], instance['K']   # pylint: disable=invalid-name

    # Reorder nodes to canonical order: depot - ppick - lpick - pdrop - ldrop
    depot_id = instance.get('depot', 0)
    ppick_ids = [p[0] for p in instance['passenger_pairs']]
    lpick_ids = [p[0] for p in instance['parcel_pairs']]
    pdrop_ids = [p[1] for p in instance['passenger_pairs']]
    ldrop_ids = [p[1] for p in instance['parcel_pairs']]


    # //// Extract content features
    # 1. Create canonical ID list
    canonical_ids = [depot_id] + ppick_ids + lpick_ids + pdrop_ids + ldrop_ids

    # 2. Map from node ID to original index in the distance matrix
    node_order = instance['node_order']
    nodeid_to_phys_idx = {node_id: i for i, node_id in enumerate(node_order)}

    # 3. Reorder distance matrix
    old_dist = instance['distance_matrix']
    dist_matrix = []

    for i_id in canonical_ids:
        row = []

        for j_id in canonical_ids:
            row.append(old_dist[nodeid_to_phys_idx[i_id]][nodeid_to_phys_idx[j_id]])
        dist_matrix.append(row)

    # 4. Reorder coords
    coords_dict = instance['coords_dict']
    coords = []
    for nodeid in canonical_ids:
        coords.append(coords_dict.get(nodeid, (0.0, 0.0)))

    # 5. Reorder parcel quantities
    pq_dict = instance['parcel_quantities_dict']
    parcel_quantities = [pq_dict[lid] for lid in lpick_ids]

    # 6. Vehicle capacities
    vehicle_capacities = instance['vehicle_capacities']
    problem = ShareARideProblem(
        N=N, M=M, K=K,
        parcel_qty=parcel_quantities,
        vehicle_caps=vehicle_capacities,
        dist=dist_matrix,
        coords=coords,
        name=instance.get('name', 'Unknown')
    )

    return problem




# ================ Instance Extraction ================
def extract_sarp_content(rel_path: str) -> InstanceContent:
    """
    Parse a SARP instance file (relative path ending at .sarp)
    and return the parsed InstanceContent (dict of problem features).
    """
    with open(rel_path, 'r', encoding='utf-8') as f:
        raw = f.read()
        content = parse_sarp_content(raw)

    return content


def extract_sarp_core(rel_path: str) -> ShareARideProblem:
    """
    Parse a SARP instance file (relative path ending at .sarp)
    and return a ShareARideProblem object.
    """
    with open(rel_path, 'r', encoding='utf-8') as f:
        raw = f.read()
        content = parse_sarp_content(raw)

    return _content_to_problem(content)




def extract_hustack_content(rel_path: str) -> InstanceContent:
    """
    Parse a simplified Hustack .inp instance file and return the parsed InstanceContent.
    """
    with open(rel_path, 'r', encoding='utf-8') as f:
        raw = f.read()
        content = parse_hustack_content(raw)

    return content


def extract_hustack_core(rel_path: str) -> ShareARideProblem:
    """
    Parse a simplified Hustack .inp instance file and return a ShareARideProblem object.
    """
    with open(rel_path, 'r', encoding='utf-8') as f:
        raw = f.read()
        content = parse_hustack_content(raw)

    return _content_to_problem(content)




# ================ Dataset Extraction ================
def extract_instance_names(dataset_root: str) -> list[str]:
    """
    Crawls the dataset root to find all .sarp instance names.
    """
    instances_set: set[str] = set()
    if not os.path.isdir(dataset_root):
        return []

    for _root, _dirs, files in os.walk(dataset_root, topdown=True, followlinks=False):
        for file in files:
            if file.endswith(".sarp"):
                # Get path relative to dataset_root, excluding extension
                rel_path = os.path.relpath(os.path.join(_root, file), dataset_root)
                # Normalize to forward slashes and remove extension
                instances_set.add(os.path.splitext(rel_path)[0].replace("\\", "/"))

    return sorted(list(instances_set))


def extract_dataset(dataset: Dataset) -> dict[str, ShareARideProblem]:
    """
    Process a dataset enum object to route, scrape and pass all .sarp instance of
    the dataset into ShareARideProblem instances.

    Returns a list of ShareARideProblem objects.
    """
    dataset_path = path_router(dataset.value.name, "readall")
    instances = {}

    # Get all .sarp files in the dataset directory
    filenames = os.listdir(dataset_path)

    # Sort filenames in natural order (numerical order for numbers in string)
    filenames.sort(key=lambda f: [
        int(c) if c.isdigit() else c for c in re.split(r'(\d+)', f)
    ])

    # Parse each .sarp file
    for filename in filenames:
        if filename.endswith(".sarp"):
            inst_name = os.path.splitext(filename)[0]
            file_path = dataset_path + filename
            problem = extract_sarp_core(file_path)
            instances[inst_name] = problem

    return instances




# ================ Solution Extraction ================
def extract_sol_content(file_path: str) -> SolutionContent:
    """
    Parse a .sol solution file and return structured solution data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return parse_sol_content(content)


def extract_sol_core(file_path: str, problem: ShareARideProblem) -> Solution:
    """
    Parse a .sol solution file and return a Solution object.
    """
    # Extract solution content
    sol_content = extract_sol_content(file_path)

    # Build routes from solution content
    routes = [[] for _ in range(problem.K)]
    routes_data = sol_content['routes']

    for route_item in routes_data:
        vid = route_item['vehicle_id']
        path = route_item['path']

        # Assign 1-indexed vehicle ID to 0-indexed routes list
        routes[vid - 1] = path

    # Create and return Solution object
    return Solution(problem, routes)




# ================ CSV Extraction ================
def extract_csv_dataframe(file_path: str, columns: list[str]):
    """
    Extract CSV file into a DataFrame with specified columns.
    
    Args:
        file_path: Path to the CSV file
        columns: Expected column names for the DataFrame
    
    Returns:
        DataFrame with the specified columns. Returns empty DataFrame
        with correct columns if file doesn't exist or is empty.
    """
    import pandas as pd         # pylint: disable=import-outside-toplevel

    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        return pd.DataFrame({col: pd.Series(dtype="object") for col in columns})

    df: pd.DataFrame = pd.read_csv(file_path)
    return df.reindex(columns=columns)
