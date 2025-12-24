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
- cost unit gap over previous best attempt
- percentage of gap to the previous best attempt, rounded to 2 decimal places
- note (if any)
If there is no successful attempt, best cost and related fields are None.
If there is no previous best cost, gap fields are None.
"""
import os
import csv
from typing import Dict, Any, List, Optional

from share_a_ride.data.router import path_router
from share_a_ride.data.classes import SCOREBOARD_COLUMNS, Dataset




def summarize_dataset(dataset: Dataset, verbose: bool = False) -> Dict[str, Any]:
    """
    Summarize the dataset attempts in the attempts csv file.
    This is saved as a summary csv, grouped by instance names as entry rows.
    If verbose, print the summary to standard output.
    Return a summary dictionary of all instances.

    Params:
    - dataset: Dataset enum member
    - verbose: whether to print information during execution

    Returns:
    - summaries: Dictionary of summary dictionaries for each instance
    """
    # Get paths
    dataset_name = dataset.value.name
    dataset_path = path_router(dataset_name, "readall")
    attempts_file = path_router(dataset_name, "record")
    scoreboard_file = path_router(dataset_name, "summarize")

    # Get all instance names from the dataset folder
    instance_files = [f for f in os.listdir(dataset_path) if f.endswith('.sarp')]
    instances = [os.path.splitext(f)[0] for f in instance_files]
    n_instances = len(instances)

    if verbose:
        print(f"Starting dataset summary: '{dataset_name}'")
        print(f"Total instances: {n_instances}")
        print(f"{'='*40}\n")


    # //// Read attempts file
    attempts_by_instance = {}
    if os.path.exists(attempts_file):
        with open(attempts_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                inst = row['instance']
                # Normalize instance name by removing extension if present
                if inst.endswith('.sarp'):
                    inst = os.path.splitext(inst)[0]
                
                if inst not in attempts_by_instance:
                    attempts_by_instance[inst] = []
                attempts_by_instance[inst].append(row)


    # //// Read scoreboard file
    scoreboard_rows = []
    if os.path.exists(scoreboard_file):
        with open(scoreboard_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            scoreboard_rows = list(reader)

    scoreboard_map = {row['instance']: row for row in scoreboard_rows}


    # //// Summarize each instance
    summaries = {}
    for i, inst in enumerate(instances, start=1):
        if verbose:
            print(f"[{i}/{n_instances}] Summarizing: {inst}")

        # Get attempts info for this instance
        inst_attempts = attempts_by_instance.get(inst, [])

        # Get previous best cost
        previous_best_cost: Optional[int] = None
        if inst in scoreboard_map:
            row = scoreboard_map.get(inst, {})
            if row.get('best_cost'):
                try:
                    previous_best_cost = int(row['best_cost'])
                except ValueError:
                    previous_best_cost = None

        # Summarize
        summary = summarize_per_instance(
            dataset,
            inst,
            inst_attempts,
            previous_best_cost,
            verbose=verbose
        )

        # Store summary
        summaries[inst] = summary


    # //// Update scoreboard rows
    new_scoreboard_rows = []
    processed_instances = set()

    # Update scoreboard rớ
    for row in scoreboard_rows:
        inst = row['instance']
        if inst in summaries and summaries[inst]:
            new_scoreboard_rows.append(summaries[inst])
            processed_instances.add(inst)
        else:
            new_scoreboard_rows.append(row)

    # Append new rows (if there are new instances)
    for inst in instances:
        if inst not in processed_instances and summaries[inst]:
            new_scoreboard_rows.append(summaries[inst])


    # //// Write the content
    os.makedirs(os.path.dirname(scoreboard_file), exist_ok=True)
    with open(scoreboard_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SCOREBOARD_COLUMNS)
        writer.writeheader()
        writer.writerows(new_scoreboard_rows)


    # //// Logging
    if verbose:
        print(f"\n{'='*40}")
        print(f"Dataset summary completed: '{dataset_name}'")
        print(f"{'='*40}")

        valid_summaries = [s for s in summaries.values() if s]
        print(f"Instances with attempts: {len(valid_summaries)}/{n_instances}")

        for inst, summary in summaries.items():
            if summary:     # Only print if there is a summary
                print(
                    f"  - {inst}: {summary['successful_attempts']}/{summary['num_attempts']}" \
                    f" successful, best cost: {summary['best_cost']}, solver: {summary['best_solver']}"
                )
        print(f"{'='*60}\n")


    return summaries




def summarize_per_instance(
        dataset: Dataset,
        inst_name: str,
        instance_attempts: List[Dict[str, Any]],
        previous_best_cost: Optional[int],
        verbose: bool = False
    ) -> Dict[str, Any]:
    """
    Summarize the dataset attempts for a specific instance.
    Return a summary dictionary.

    Params:
    - dataset: Dataset enum member
    - inst_name: name of the instance file without extension
    - instance_attempts: list of attempt dictionaries for this instance
    - previous_best_cost: previous best cost from scoreboard, or None
    - verbose: whether to print information during execution

    Returns:
    - summary: Dictionary containing summary statistics for the instance
    """
    dataset_name = dataset.value.name

    # If no attempts found, return empty dict
    if not instance_attempts:
        if verbose:
            print(f"No attempts found for instance {inst_name} in dataset {dataset_name}.")
        return {}


    # //// Calculate summary statistics
    num_attempts = len(instance_attempts)
    successful_attempts = sum(1 for att in instance_attempts if att['status'] == 'done')

    # //// Iterating to find best attempt
    best_attempt = None
    best_cost = 10**18
    for attempt in instance_attempts:
        if attempt['status'] == 'done' and attempt['cost']:
            try:
                cost = float(attempt['cost'])
                if cost < best_cost:
                    best_cost = cost
                    best_attempt = attempt
            except ValueError:
                continue


    # //// Build summary dictionary
    if best_attempt is None:    # No successful attempts
        summary = {
            'dataset'                   : dataset_name,
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
            'cost_gap'                  : None,
            'pct_gap'                   : None,
            'note'                      : None
        }

    else:                       # There is a best attempt
        # Calculate gap over previous best
        cost_gap = None
        pct_gap = None
        if previous_best_cost is not None:
            cost_gap = previous_best_cost - best_cost
            pct_gap = round((cost_gap / previous_best_cost) * 100, 2)
        improved = cost_gap and cost_gap > 1e-6

        # Build args dict
        args_dict = {}
        if best_attempt['seed']:
            args_dict['seed'] = best_attempt['seed']
        if best_attempt['time_limit']:
            args_dict['time_limit'] = best_attempt['time_limit']

        summary = {
            'dataset'                   : dataset_name,
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
            'cost_gap'                  : cost_gap,
            'pct_gap'                   : pct_gap,
            'note'                      : 'improved' if improved else None
        }


    # //// Logging
    if verbose:
        print(f"{'-'*40}")
        print(  f"Instance summary for {inst_name}:")
        print(  f"  Attempts: {num_attempts} ({successful_attempts} successful)")
        print(  f"  Best cost: {summary['best_cost']}")
        if summary['cost_gap']:
            print(
                f"  Improvement found:\n" \
                f"    - Cost decreased: {summary['cost_gap']}\n" \
                f"    - Gap: {summary['pct_gap']} %\n"
            )
        print()


    return summary




# ================= Playground ==================
if __name__ == "__main__":
    summarize_dataset(Dataset.CMT, verbose=True)
