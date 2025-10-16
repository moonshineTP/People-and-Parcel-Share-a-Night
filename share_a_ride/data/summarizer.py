"""
Module for summarizing solve attempts dataset-wise and instance-wise.
The summary is saved as a csv file with -scoreboard suffix.
The summary has the following columns:
- instance name
- number of attempts
- successful attempts
- best cost
- best attempt's id
- best attempt's timestamp
- best solver
- best solver's args
- best solver's hyperparams
- best solver's time taken
- cost unit improvement over previous best attempt
- percentage of improvement to the previous best attempt, rounded to 2 decimal places
- notes (if any)
If there is no successful attempt, best cost and related fields are None.
If there is no previous best cost, improvement fields are None.
"""

import os
import csv
from typing import Dict, Any

from share_a_ride.data.router import path_router


def summarize_dataset(dataset: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Summarize the dataset attempts in the attempts csv file.
    This is saved as a summary csv, grouped by instance names as entry rows.
    If verbose, print the summary to standard output.
    Return a summary dictionary of all instances.
    """

    # Get paths
    attempts_file = path_router(dataset, "attempt")
    scoreboard_file = path_router(dataset, "summarize")
    dataset_folder = path_router(dataset, "readall")


    # Get all instance names from the dataset folder
    instance_files = [f for f in os.listdir(dataset_folder) if f.endswith('.sarp')]
    instances = [os.path.splitext(f)[0] for f in instance_files]


    # Summarize each instance (each call writes to scoreboard itself)
    summaries = {}
    for inst in instances:
        summary = summarize_instance(dataset, inst, verbose=verbose)
        summaries[inst] = summary


    # Print summary if verbose
    if verbose:
        print(f"{'='*60}")
        print(f"Dataset summary for {dataset}:")
        print(f"{'-'*40}")
        print(f"Total instances: {len(summaries)}")
        for inst, summary in summaries.items():
            if summary:     # Only print if there is a summary
                print(f"  - {inst}: {summary['successful_attempts']}/{summary['num_attempts']}" \
                        f" successful, best cost: {summary['best_cost']}")
        print(f"{'='*60}\n")


    return summaries



def summarize_instance(
        dataset: str,
        inst_name: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
    """
    Summarize the dataset attempts for a specific instance.
    This updates or creates an entry row in the scoreboard csv file.
    If verbose, print the summary to standard output.
    Return a summary dictionary.
    """
    attempts_file = path_router(dataset, "attempt")
    scoreboard_file = path_router(dataset, "summarize")


    # Check if attempts file exists
    if not os.path.exists(attempts_file):
        raise FileNotFoundError(f"Attempts file not found: {attempts_file}")
    if not os.path.exists(scoreboard_file):
        raise FileNotFoundError(f"Scoreboard file not found: {scoreboard_file}")


    # Read and filter attempts for the instance name
    with open(attempts_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        instance_attempts = [row for row in reader if row['instance'] == inst_name]
    if not instance_attempts:
        if verbose:
            print(f"No attempts found for instance {inst_name} in dataset {dataset}.")
        return None

    # Find previous best cost from scoreboard
    previous_best_cost = None
    fieldnames = [
        'instance', 'num_attempts', 'successful_attempts', 'best_cost',
        'best_attempt_id', 'best_timestamp', 'best_solver', 'best_solver_args',
        'best_solver_hyperparams', 'best_time_taken', 'cost_improvement',
        'percentage_improvement', 'notes'
    ]

    if os.path.getsize(scoreboard_file) == 0:
        with open(scoreboard_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
    else:
        with open(scoreboard_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['instance'] == inst_name and row['best_cost']:
                    previous_best_cost = float(row['best_cost'])


    # Calculate summary statistics
    num_attempts = len(instance_attempts)
    successful_attempts = sum(1 for att in instance_attempts if att['status'] == 'done')

    # Scan through attempts to find the best one
    best_attempt = None
    best_cost = float('inf')
    for attempt in instance_attempts:
        if attempt.get('status') == 'done' and attempt.get('cost'):
            cost = float(attempt.get('cost'))
            if cost < best_cost:
                best_cost = cost
                best_attempt = attempt


    # Build summary dictionary
    if best_attempt is None:
        summary = {
            'instance'                  : inst_name,
            'num_attempts'              : num_attempts,
            'successful_attempts'       : 0,
            'best_cost'                 : None,
            'best_attempt_id'           : None,
            'best_timestamp'            : None,
            'best_solver'               : None,
            'best_solver_args'          : None,
            'best_solver_hyperparams'   : None,
            'best_time_taken'           : None,
            'cost_improvement'          : None,
            'percentage_improvement'    : None,
            'notes'                     : None
        }

    else:
        # Calculate improvement over previous best
        cost_improvement = None
        percentage_improvement = None

        if previous_best_cost is not None:
            cost_improvement = previous_best_cost - best_cost
            percentage_improvement = round((cost_improvement / previous_best_cost) * 100, 2)

        improved = cost_improvement and cost_improvement > 1e-6

        # Build args dict from seed and time_limit
        args_dict = {}
        if best_attempt['seed']:
            args_dict['seed'] = best_attempt['seed']
        if best_attempt['time_limit']:
            args_dict['time_limit'] = best_attempt['time_limit']

        summary = {
            'instance'                  : inst_name,
            'num_attempts'              : num_attempts,
            'successful_attempts'       : successful_attempts,
            'best_cost'                 : round(best_cost, 2),
            'best_attempt_id'           : best_attempt['attempt_id'],
            'best_timestamp'            : best_attempt['timestamp'],
            'best_solver'               : best_attempt['solver'],
            'best_solver_args'          : str(args_dict),
            'best_solver_hyperparams'   : best_attempt['hyperparams'],
            'best_time_taken'           : best_attempt['elapsed_time'],
            'cost_improvement'          : cost_improvement,
            'percentage_improvement'    : percentage_improvement,
            'notes'                     : 'improved' if improved else None
        }

    # Write or update the scoreboard
    scoreboard_rows = []
    row_found = False

    if os.path.getsize(scoreboard_file) > 0:
        with open(scoreboard_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['instance'] == inst_name:
                    scoreboard_rows.append(summary)     # Update existing row
                    row_found = True
                else:
                    scoreboard_rows.append(row)         # Keep other rows unchanged

    if not row_found:
        scoreboard_rows.append(summary)

    # Write back to scoreboard
    with open(scoreboard_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scoreboard_rows)


    # Print summary if verbose
    if verbose:
        print(f"{'-'*40}")
        print(  f"Instance summary for {inst_name}:")
        print(  f"  Attempts: {num_attempts} ({successful_attempts} successful)")
        print(  f"  Best cost: {summary['best_cost']}")
        if summary['cost_improvement'] not in [None, 0]:
            print(
                f"  Improvement found:\n" \
                f"    - Cost decreased: {summary['cost_improvement']}\n" \
                f"    - Gap: {summary['percentage_improvement']} %\n"
            )
        print()


    return summary


if __name__ == "__main__":
    summarize_dataset("H", verbose=True)
