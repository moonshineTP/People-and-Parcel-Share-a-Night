"""
Module for routing interactions to the dataset and vice versa.
"""

import os


def path_router(
        dataset: str, action: str,
        filename: str = None, solver: str = None
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

    act_to_ext = {
        "readfile"  : ".sarp",
        "solve"     : ".sol",
    }

    dts_to_pur = {
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

    base_dir = os.path.dirname(__file__)
    dataset_root = os.path.join(base_dir, dts_to_pur[dataset], dataset)

    # If action is attempt, return the path to the attempt csv file
    if action == "attempt":
        if filename is not None or solver is not None:
            raise ValueError("Filename and solver must be None when action is 'attempt'.")
        return os.path.join(dataset_root, f"{dataset}-attempts.csv")

    # If action is readall, return the path to the dataset folder
    elif action == "readall":
        return dataset_root

    # If action is readfile, return the path to the .sarp instance file
    elif action == "readfile":
        if filename is None:
            raise ValueError("Filename must be provided when action is 'readfile'.")
        return os.path.join(dataset_root, f"{filename}{act_to_ext[action]}")

    # If action is solve, return the path to the solution file
    elif action == "solve":
        if solver is None:
            raise ValueError("Solver must be provided when action is 'solve'.")
        if filename is None:
            raise ValueError("Filename must be provided when action is 'solve'.")
        return os.path.join(dataset_root, f"{filename}_{solver}{act_to_ext[action]}")

    # If action is summarize, return the path to the summary file
    elif action == "summarize":
        return os.path.join(dataset_root, f"{dataset}-scoreboard.csv")

    # Else, raise error
    else:
        raise ValueError(
            "Action must be either " \
            "'attempt', 'readall', 'readfile', 'solve', or 'summarize'."
        )

