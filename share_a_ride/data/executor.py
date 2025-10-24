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
    - status: "done" or "timeout" or "error"
    - elapsed_time: total time taken for the attempt (in seconds)
    - cost: cost of the best solution found (or None if no solution)
    - info: JSON string of additional statistics from the solver (or empty if none)
    - note: any additional notes or comments (or empty if none)
"""
import os
import csv
import json

from typing import Optional
from datetime import datetime, timezone

from share_a_ride.data.parser import parse_sarp_to_problem
from share_a_ride.data.router import path_router
from share_a_ride.solvers.algo.algo import AlgoSolver
from share_a_ride.solvers.learner.learner import LearnerSolver
from share_a_ride.core.solution import Solution



ATTEMPT_COLUMNS = [
    'attempt_id',
    'timestamp',
    'dataset',
    'instance',
    'solver',
    'seed',
    'time_limit',
    'hyperparams',
    'status',
    'elapsed_time',
    'cost',
    'info',
    'note'
]



def attempt_dataset(

        solver: AlgoSolver | LearnerSolver,
        dataset: str,
        note: str = "",
        verbose: bool = False
    ) -> tuple[list[Solution], list[float], str]:
    """
    Attempt to solve all instances in a dataset with a given solver.
    Use attempt_instance for each instance.

    Params:
    - solver: a callable in share_a_ride/solvers/algo that solve the SARP instance
    - dataset: name of the dataset folder to attempt (must exist in share_a_ride/data)
    - note: any additional notes or comments
    - verbose: whether to print information during execution

    Returns: a tuple of:
    - solutions: ist of Solution objects or None for each instance in the dataset
    - gaps: List of gaps (in percentage) for each instance where a solution was found
    - result_summary: A string summarizing the attempt information.
    """

    # Get all instance files in the dataset directory
    dataset_dir = path_router(dataset, "readall")
    instance_files = [f for f in os.listdir(dataset_dir) if f.endswith('.sarp')]

    if verbose:
        print(f"\n{'='*40}")
        print(f"Starting dataset attempt: '{dataset}'")
        print(f"Total instances: {len(instance_files)}")
        print(f"Solver: {solver.name}")
        print(f"{'='*40}\n")

    # Prepare lists to collect results
    solutions = []
    gaps = []
    results_summary = []

    # Attempt each instance
    for idx, instance_file in enumerate(instance_files, 1):
        if verbose:
            print(f"\n[{idx}/{len(instance_files)}] Processing: {instance_file}")
            print("-" * 20)

        # Remove extension
        instance_name = instance_file.replace('.sarp', '')

        # Use attempt_instance to solve this instance
        sol, gap_percentage, result_msg = attempt_instance(
            solver=solver,
            dataset=dataset,
            instance_name=instance_name,
            note=note,
            verbose=verbose
        )

        solutions.append(sol)
        gaps.append(gap_percentage)
        results_summary.append(result_msg)

        # Calculate summary statistics
        successful_attempts = sum(1 for sol in solutions if sol is not None)
        failed_attempts = len(solutions) - successful_attempts


    # Create final summary
    final_summary = f"Attempted {len(instance_files)} instances in dataset '{dataset}':\n"
    for summary in results_summary:
        final_summary += f"  - {summary}\n"


    if verbose:
        print(f"\n{'='*40}")
        print(f"Dataset attempt completed: '{dataset}'")
        print(f"{'='*40}")
        print(f"Successful attempts: {successful_attempts}/{len(instance_files)}")
        print(f"Failed attempts: {failed_attempts}/{len(instance_files)}")
        print(f"{'='*40}\n")


    return (solutions, gaps, final_summary)



def attempt_instance(
        solver: AlgoSolver | LearnerSolver,
        dataset: str,
        instance_name: str,
        note: str = "",
        verbose: bool = False
    ) -> tuple[Optional[Solution], float, str]:
    """
    Attempt to solve a single instance in a dataset with a given solver.
    Also if the attempt was successful, it saves the results in a csv file.
    Return a string describing the result of the attempt.


    Params:
    - solver: a callable in share_a_ride/solvers/algo that solve the SARP instance
    - dataset: name of the dataset folder to attempt (must exist in share_a_ride/data)
    - instance_name: name of the instance file without extension (must exist in dataset)
    - note: any additional notes or comments

    Returns: a tuple of:
    - solution: Solution object or None if no solution was found
    - gap: gap (in percentage) if a solution was found, else None
    - result_summary: A string summarizing the attempt information.
    """
    # Get the CSV file path for the dataset
    csv_path = path_router(dataset, "attempt")

    # Check if CSV exists. If not, raise error.
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file for dataset '{dataset}' not found at {csv_path}.")

    # Open the file and get next attempt_id
    with open(csv_path, 'r+', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

        # Write header if file is empty
        if len(rows) == 0:
            writer = csv.writer(f)
            writer.writerow(ATTEMPT_COLUMNS)
            attempt_id = 1
        else:
            # Next id is the number of data rows (excluding header)
            attempt_id = len(rows)


    # Get best cost from scoreboard
    scoreboard_path = path_router(dataset, "summarize")
    best_cost = None
    if os.path.exists(scoreboard_path):
        with open(scoreboard_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['instance'] == instance_name:
                    best_cost = float(row['best_cost']) if row['best_cost'] else None
                    break

    # Print the instance and best cost if found
    if verbose:
        print(f"Attempting instance: {instance_name}")
        if best_cost:
            print(f"Best known cost: {best_cost}")

    try:
        # Parse the problem instance
        instance_path = path_router(dataset, "readfile", filename=instance_name)
        prob = parse_sarp_to_problem(instance_path)

        # Get solver arguments data
        solver_name = solver.name
        seed = solver.args.get('seed', None)
        time_limit = solver.args.get('time_limit', None)
        hyperparams_json = json.dumps(solver.hyperparams) if solver.hyperparams else "{}"

        # Solve the instance
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        sol, info = solver.solve(problem=prob)

        # Extract results
        status = info['status']
        elapsed_time = info['time']
        cost = sol.max_cost if sol is not None else None

        # Calculate gap percentage
        gap_percentage = 0.0
        if cost and best_cost:
            gap_percentage = ((best_cost - cost) / best_cost) * 100.0

        if verbose:
            print(f"Status: {status}, Cost: {cost}, Elapsed time: {elapsed_time}")
            if best_cost:
                print(f"Gap percentage: {gap_percentage:.2f}%")

        # Remove redundant fields from info for CSV
        info_copy = info.copy()
        info_copy.pop('status', None)
        info_copy.pop('time', None)
        info_json = json.dumps(info_copy) if info_copy else "{}"

        # Write to CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                attempt_id, timestamp, dataset, instance_name, solver_name,
                seed, time_limit, hyperparams_json, status, elapsed_time,
                cost, info_json, note
            ])

        return (sol, gap_percentage, f"{instance_name}: {status} (cost={cost})")


    except Exception as e:
        raise e
        # if verbose:
        #     print(f"Error occurred: {str(e)}")

        # # Write error to CSV
        # timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        # with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([
        #         attempt_id, timestamp, dataset, instance_name,
        #         solver.name, None, None, "{}", "error", 0,
        #         None, "{}", f"{str(e)}"
        #     ])

        # return (None, 0.0, f"{instance_name}: error ({str(e)})")


if __name__ == "__main__":
    from share_a_ride.solvers.algo.greedy import iterative_greedy_balanced_solver
    chosen_solver = AlgoSolver(
        algo=iterative_greedy_balanced_solver,
        args={"iterations": 10000, "time_limit": 10.0, "seed": 42, "verbose": 1},
        hyperparams={
            "destroy_proba"      : 0.5,
            "destroy_steps"     : 5,
            "destroy_T"         : 1.0,
            "rebuild_proba"      : 0.3,
            "rebuild_steps"     : 2,
            "rebuild_T"         : 5.0,
        }
    )

    sol, gap, msg = attempt_instance(chosen_solver, "H", "H-n10-m10-k5", note="test attempt", verbose=True)


# TODO: Implement the SolverExecutor class for batch execution and management
class SolverExecutor:
    pass
