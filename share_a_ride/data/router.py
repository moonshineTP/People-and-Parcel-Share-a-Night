"""
Module for routing interactions along data files and folders.
This module helps in constructing paths to various data files
related to datasets, instances, attempts, and solutions.
"""
import os



ACTION_TO_EXTENSION = {
        "readfile"  : ".sarp",
        "solve"     : ".sol",
    }

DATASET_TO_PURPOSE = {
    "H"         : "sanity",
    "Exact"     : "sanity",
    "Li"        : "benchmark",
    "Solomon"   : "benchmark",
    "Pyvrp"     : "benchmark",
    "Golden"    : "benchmark",
    "Cvrplib"   : "train",
    "CMT"       : "val",
    "tai"       : "test",
}


def path_router(
        dataset: str, action: str,
        filename: str = "", solver: str = ""
    ) -> str:
    """
    Return the path to the router data file for the given dataset.
    - If action is "attempt", return the path to the attempts csv file.
    - If action is "readall", return the path to the dataset folder.
    - If action is "readfile", return the path to the .sarp instance file.
    - If action is "solve", return the path to the solution file for the given solver.
    - If action is "summarize", return the path to the summary file.
    - Else raise ValueError.

    Params:
    - dataset: name of the dataset (e.g. "Li", "Solomon", etc.)
    - action: one of "attempt", "readall", "readfile", "solve"
    - filename: name of the instance file without extension (required for "read" and "solve")
    - solver: name of the solver (required for "solve")
    
    Returns:
    - str: The path to the router data file for the given dataset and action.
    """
    base_dir = os.path.dirname(__file__)
    dataset_root = os.path.join(base_dir, DATASET_TO_PURPOSE[dataset], dataset)

    # If action is attempt, return the path to the attempt csv file
    if action == "attempt":
        if filename or solver:
            raise ValueError("Filename and solver must be None when action is 'attempt'.")
        return os.path.join(dataset_root, f"{dataset}-attempts.csv")

    # If action is readall, return the path to the dataset folder
    elif action == "readall":
        return dataset_root

    # If action is readfile, return the path to the .sarp instance file
    elif action == "readfile":
        if not filename:
            raise ValueError("Filename must be provided when action is 'readfile'.")
        return os.path.join(dataset_root, f"{filename}{ACTION_TO_EXTENSION[action]}")

    # If action is solve, return the path to the solution file
    elif action == "solve":
        if not solver:
            raise ValueError("Solver must be provided when action is 'solve'.")
        if not filename:
            raise ValueError("Filename must be provided when action is 'solve'.")
        return os.path.join(dataset_root, f"{filename}_{solver}{ACTION_TO_EXTENSION[action]}")

    # If action is summarize, return the path to the summary file
    elif action == "summarize":
        return os.path.join(dataset_root, f"{dataset}-scoreboard.csv")

    # Else, raise error
    else:
        raise ValueError(
            "Action must be either " \
            "'attempt', 'readall', 'readfile', 'solve', or 'summarize'."
        )
