"""
Module for routing interactions along data files and folders.
This module helps in constructing paths to various data files
related to datasets, instances, attempts, and solutions.
"""
from share_a_ride.data.classes import Action, Dataset


def path_router(
        dataset: str, action: str,
        filename: str = "", solver: str = ""
    ) -> str:
    """
    Route and return the path to the correct file path for the given dataset and action.
    The function supports the following actions:
    - "readall": return the path to the dataset folder itself.
    - "readfile": return the path to the .sarp instance file of the dataset.
    - "record": return the path to the attempts csv file.
    - "solve": return the path to the solution file for the given solver.
    - "summarize": return the path to the summary file.

    Params:
    - dataset: name of the dataset (e.g. "Li", "Solomon", etc.)
    - action: one of "readall", "readfile", "record", "solve", or "summarize"
    - filename: name of the instance file without extension (required for "readfile" or"solve")
    - solver: name of the solver (required for "solve")
    
    Returns:
    - str: The path to the router data file for the given dataset and action.
    """
    ds = Dataset.from_str(dataset)
    act = Action(action)
    
    dataset_root = f"share_a_ride/data/{ds.value.purpose.value}/{ds.value.name}/"

    # If action is readall, return the path to the dataset folder
    if act == Action.READALL:
        return dataset_root

    # If action is readfile, return the path to the .sarp instance file
    elif act == Action.READFILE:
        if not filename:
            raise ValueError("Filename must be provided when action is 'readfile'.")
        return dataset_root + f"{filename}{act.extension}"

    # If action is record, return the path to the attempt csv file
    elif act == Action.RECORD:
        return dataset_root + f"{ds.value.name}-attempts.csv"

    # If action is solve, return the path to the solution file
    elif act == Action.SOLVE:
        if not solver:
            raise ValueError("Solver must be provided when action is 'solve'.")
        if not filename:
            raise ValueError("Filename must be provided when action is 'solve'.")
        return dataset_root + f"{filename}_{solver}{act.extension}"

    # If action is summarize, return the path to the summary file
    elif act == Action.SUMMARIZE:
        return dataset_root + f"{ds.value.name}-scoreboard.csv"

    # Else, raise error
    else:
        raise ValueError(
            "Action must be either " \
            "'record', 'readall', 'readfile', 'solve', or 'summarize'."
        )
