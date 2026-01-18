"""
    Executor for attempting a configured solver for a dataset (single instance or whole)

    The contents of the csv file should include:
    - attempt_id: unique identifier for the attempt (incremental integer)
    - timestamp: time of the attempt (ISO 8601 format with +0 timezone and no microseconds)
    - dataset: name of the dataset
    - instance: name of the instance file
    - solver: name of the solver used
    - seed: random seed used for the attempt
    - time_limit: time limit set for the attempt (in seconds)
    - hyperparams: JSON string of hyperparameters used (or empty if none)
    - status: "done" or "overtime" or "error"
    - elapsed_time: total time taken for the attempt (in seconds)
    - cost: cost of the best solution found (or None if no solution)
    - info: JSON string of additional statistics from the solver (or empty if none)
    - note: any additional notes or comments (or empty if none)
"""
import os
import csv
import json
from typing import Optional, Tuple, Dict
from datetime import datetime, timezone

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import Solution
from share_a_ride.data.classes import Dataset, ATTEMPT_COLUMNS
from share_a_ride.data.extractor import extract_dataset
from share_a_ride.data.router import path_router
from share_a_ride.solvers.classes import Solver, SolverName, SolverMode, SolverParams




# ================ Main Functions ================
def attempt_dataset(
        dataset: Dataset,
        solver_name: SolverName,
        solver_mode: SolverMode = SolverMode.STANDARD,
        note: str = "",
        seed: Optional[int] = None,
        verbose: bool = False,
        **solver_kwargs
    ) -> Tuple[Dict[str, Optional[Solution]], Dict[str, float]]:
    """
    Attempt to solve all instances in a dataset with a given solver.
    Use attempt_instance for each instance.

    Params:
    - dataset: Dataset enum member
    - solver_name: SolverName enum member
    - solver_mode: SolverMode enum member
    - note: any additional notes or comments
    - verbose: whether to print information during execution
    - solver_kwargs: additional keyword arguments for the solver

    Returns: a tuple of:
    - solutions: List of Solution objects or None for each instance in the dataset
    - gaps: List of gaps (in percentage) for each instance where a solution was found
    """

    # Get all instance files in the dataset directory
    instances_dict: dict[str, ShareARideProblem] = extract_dataset(dataset)
    n_instances = len(instances_dict)

    if verbose:
        print(f"\n{'='*40}")
        print(f"Starting dataset attempt: '{dataset.value.name}'")
        print(f"Total instances: {n_instances}")
        print(f"Solver: {solver_name.name} ({solver_mode.name})")
        print(f"{'='*40}\n")

    # Prepare lists to collect results
    solutions = {}
    gaps = {}

    # Attempt each instance
    for inst_id, (inst_name, instance) in enumerate(instances_dict.items(), start=1):
        if verbose:
            print(f"\n[{inst_id}/{n_instances}] Processing: {inst_name}")
            print("-" * 20)

        # Use attempt_instance to solve this instance
        sol, gap = try_instance(
            dataset=dataset,
            inst_name=inst_name,
            problem=instance,
            solver_name=solver_name,
            solver_mode=solver_mode,
            note=note,
            seed=seed,
            verbose=verbose,
            **solver_kwargs
        )

        solutions[inst_name] = sol
        gaps[inst_name] = gap


    # Calculate summary statistics
    successful_attempts = sum(1 for sol in solutions.values() if sol)
    failed_attempts = n_instances - successful_attempts

    # final_summary = f"Attempted {len(instances)} instances in dataset '{dataset.value.name}':\n"
    # for summary in results_summary:
    #     final_summary += f"  - {summary}\n"

    # Logging
    if verbose:
        print(f"\n{'='*40}")
        print(f"Dataset attempt completed: '{dataset.value.name}'")
        print(f"{'='*40}")
        print(f"Successful attempts: {successful_attempts}/{n_instances}")
        print(f"Failed attempts: {failed_attempts}/{n_instances}")
        print(f"{'='*40}\n")


    return (solutions, gaps)




def try_instance(
        dataset: Dataset,
        inst_name: str,
        problem: ShareARideProblem,
        solver_name: SolverName,
        solver_mode: SolverMode,
        note: str = "",
        seed: Optional[int] = None,
        verbose: bool = False,
        incumbent: bool = False,
        **solver_kwargs
    ) -> Tuple[Optional[Solution], Optional[float]]:
    """
    Attempt to solve a single instance in a dataset with a given solver.
    If the attempt was successful, record the results in a csv file.


    Params:
    - dataset: Dataset enum member
    - inst_name: name of the instance file (must exist in dataset)
    - instance: ShareARideProblem object
    - solver_name: SolverName enum member
    - solver_mode: SolverMode enum member
    - note: any additional notes or comments
    - verbose: whether to print information during execution
    - solver_kwargs: additional keyword arguments for the solver

    Returns: a tuple of:
    - solution: Solution object if a solution is found, else None
    - gap: gap (in percentage) if a solution is found, else None
    """
    # Get path to the results csv file
    csv_path = path_router(dataset.value.name, "record")

    # Open the file and determine the next attempt_id
    attempt_id = 1
    if os.path.exists(csv_path):    # File exist
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) == 0:  # No header found
                with open(csv_path, 'a', newline='', encoding='utf-8') as f_write:
                    writer = csv.writer(f_write)
                    writer.writerow(ATTEMPT_COLUMNS)
            else:
                attempt_id = len(rows)  # Header is row 0

    else:    # Create the file and write header
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(ATTEMPT_COLUMNS)

    # Extract best known cost from scoreboard if available
    scoreboard_path = path_router(dataset.value.name, "summarize")
    best_cost = None
    if os.path.exists(scoreboard_path):     # File exist
        with open(scoreboard_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['instance'] == inst_name:
                    best_cost = int(row['best_cost'])
                    break

    # Logging
    if verbose:
        print(f"Attempting instance: {inst_name}")
        if best_cost:
            print(f"Best known cost: {best_cost}")
        print("-------------------------------")
        print()


    # //// Solve the instance
    try:
        # Timestamp
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        # Extract solver arguments data
        full_params = SolverParams.run_params(solver_name, solver_mode)
        full_params.update(solver_kwargs)
        full_params["seed"] = seed
        time_limit = full_params.get("time_limit", None)

        # Extract solver hyperparams
        hyperparams = SolverParams.hyperparams(solver_name)
        hyperparams_json = json.dumps(hyperparams) if hyperparams else ""

        # Run the solver
        solver = Solver(solver_name, solver_mode)

        if incumbent:
            sol, info = solver.run(
                problem=problem,
                verbose=verbose,
                seed=seed,
                incumbent=best_cost,
                **solver_kwargs
            )
        else:
            sol, info = solver.run(
                problem=problem,
                verbose=verbose,
                seed=seed,
                **solver_kwargs
            )

        # Extract results
        status = info.get("status", 'error')
        elapsed_time = info.get("time", 0.0)
        cost = sol.max_cost if sol is not None else None
        if verbose:
            print(f"Status: {status}, Cost: {cost}, Elapsed time: {elapsed_time}")

        # Calculate gap percentage
        gap_percentage = 0.0
        if cost and best_cost:
            gap_percentage = ((best_cost - cost) / best_cost) * 100.0
            print(f"Gap percentage: {gap_percentage:.2f}%")

        # Deduplicate fields from info for CSV
        info_copy = info.copy()
        info_copy.pop("status", None)
        info_copy.pop("time", None)
        info_json = json.dumps(info_copy) if info_copy else "{}"

        # Write to CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                attempt_id, timestamp, dataset.value.name, inst_name, solver_name.name,
                seed, time_limit, hyperparams_json, status, elapsed_time,
                cost, info_json, note
            ])

        if verbose:
            print("-------------------------------")
            for _ in range(4):
                print()

        return sol, gap_percentage

    except Exception as e:      # pylint: disable=broad-except
        if verbose:
            print(f"Error solving instance: {e}")
            print("-------------------------------")
            for _ in range(4):
                print()

        return None, None




# ================ Playground ================
if __name__ == "__main__":
    solvernames = [
        # SolverName.BEAM,
        # SolverName.GREEDY,
        # SolverName.HGS,
        SolverName.ACO,
        # SolverName.ASTAR,
        # SolverName.MCTS,
        # SolverName.ALNS,
    ]
    dts = Dataset.LI

    for slvr in solvernames:
        attempt_dataset(
            dataset=dts,
            solver_name=slvr,
            solver_mode=SolverMode.INTENSIVE,
            note="Testing executor",
            verbose=True,
            time_limit=1000.0
        )

    # instancenames = []
    # instances = parse_dataset(dts)
    # for slvr in solvernames:
    #     for name in instancenames:
    #         try_instance(
    #             dataset=dts,
    #             inst_name=name,
    #             problem=instances[name],
    #             solver_name=slvr,
    #             solver_mode=SolverMode.INTENSIVE,
    #             note="",
    #             verbose=True,
    #             time_limit=400,
    #         )
