"""
Greedy balanced heuristic and its iterative improvement.
"""

import time, random
from typing import Any, List, Optional, Tuple, Dict

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, Solution
from share_a_ride.solvers.operator.repair import repair_operator
from share_a_ride.solvers.operator.destroy import destroy_operator



def greedy_balanced_solver(
        prob: ShareARideProblem,
        premature_routes: List[List[int]] = [],
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Greedy balanced heuristic:
    - At each step, choose the taxi with smallest current route cost
    - For that taxi, evaluate all feasible next actions (pick/drop).
    - Choose the action with minimal added distance (inc).
    - Repeat until all passengers and parcels are served.

    Returns:
    (sol, info): tuple where
    - sol: Solution object with routes and costs.
    - Info dictionary contains:
        + iterations: number of main loop iterations
        + actions_evaluated: total number of actions evaluated
        + time: total time taken
    """

    start_time = time.time()
    partial = PartialSolution(problem=prob, routes=premature_routes)
    taxi_states = partial.route_states

    def has_pending_work() -> bool:
        return bool(
            partial.remaining_pass_pick
            or partial.remaining_pass_drop
            or partial.remaining_parc_pick
            or partial.remaining_parc_drop
        )

    stats = {"iterations": 0, "actions_evaluated": 0}
    while has_pending_work():
        stats["iterations"] += 1

        available_taxis = [
            t_idx for t_idx, t_state in enumerate(taxi_states)
            if not t_state["ended"]
        ]
        if not available_taxis:
            break

        argmin_t_idx = min(available_taxis, key=lambda i: taxi_states[i]["cost"])
        actions = partial.possible_actions(argmin_t_idx)
        stats["actions_evaluated"] += len(actions)

        if verbose:
            print(f"Taxi with min cost: {argmin_t_idx}")
            print(f"Actions available: {actions}")

        # No feasible actions: return to depot and end route
        if not actions:
            partial.apply_return_to_depot(argmin_t_idx)
            continue

        # Else, choose action with minimal added cost
        kind, idx, inc = min(actions, key=lambda x: x[2])
        partial.apply_action(argmin_t_idx, kind, idx, inc)

        if verbose:
            print(f"Taxi: {argmin_t_idx}: {taxi_states[argmin_t_idx]['route']}\n")


    # All taxis return to depot if not already ended
    for t_idx, t_state in enumerate(taxi_states):
        if not t_state["ended"]:
            partial.apply_return_to_depot(t_idx)

    if verbose:
        print("All tasks completed.")

    # Build final solution
    sol = partial.to_solution()

    # Final stats
    elapsed = time.time() - start_time
    info = {
        "iterations": stats["iterations"],
        "actions_evaluated": stats["actions_evaluated"],
        "time": elapsed
    }

    # Validate solution and return
    if sol and not sol.is_valid():
        sol = None
    assert sol.is_valid() if sol else True
    return sol, info


def iterative_greedy_balanced_solver(
        prob: ShareARideProblem,
        iterations: int = 10,
        time_limit: float = 10.0,
        seed: int = 42,
        verbose: bool = False,
        destroy_proba: float = 0.4,
        destroy_steps: int = 15,
        destroy_T: float = 1.0,
        rebuild_proba: float = 0.3,
        rebuild_steps: int = 5,
        rebuild_T: float = 1.0,
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Iterative improvement of greedy balanced heuristic using destroy-and-rebuild.
    At each iteration:
    - Destroy a fraction of routes (randomly selected based on cost)
    - Rebuild those routes partially with some randomness
    - Complete the solution with greedy balanced heuristic
    - If improved, update the best solution found.

    Args:
        - prob: ShareARideProblem instance
        - iterations: Number of destroy-and-rebuild iterations
        - time_limit: Max time allowed (seconds)
        - seed: Random seed for reproducibility
        - verbose: If True, print progress info
        - destroy_proba: Fraction of routes to destroy (0 to 1)
        - destroy_steps: Max number of nodes to remove per route
        - destroy_T: Temperature parameter for destroy operator
        - rebuild_proba: Probability of rebuilding a destroyed route
        - rebuild_steps: Max number of nodes to add during rebuild
        - rebuild_T: Temperature parameter for rebuild operator
    
    Returns:
        (sol, info): tuple where
        - sol: Best Solution found (or None if no valid solution)
        - Info dictionary contains:
            + iterations: number of iterations performed
            + improvements: number of times solution was improved
            + actions_evaluated: total number of actions evaluated
            + nodes_destroyed: total number of nodes removed
            + nodes_rebuilt: total number of nodes added during rebuild
            + time: total time taken
            + status: "done" if completed, "timeout" if time limit reached
    """

    assert 1e-5 < destroy_proba < 1 - 1e-5
    assert 1e-5 < rebuild_proba < 1 - 1e-5
    assert 1 <= rebuild_steps <= destroy_steps

    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + time_limit if time_limit is not None else None

    # Initial solution and best cost
    best_sol, base_info = greedy_balanced_solver(prob, verbose=False)
    if not best_sol:
        return None, {"time": time.time() - start_time, "status": "error"}
    best_cost = best_sol.max_cost

    # Stats
    total_actions = base_info["actions_evaluated"]
    improvements = 0
    nodes_destroyed = 0
    nodes_rebuilt = 0
    status = "done"
    iterations_done = 0


    if verbose:
        print(f"[iter 0] initial best cost: {best_cost}")


    # ================= Main loop =================
    for it in range(1, iterations + 1):
        if deadline and time.time() >= deadline:
            status = "timeout"
            break
        iterations_done += 1

        # ============== Destroy phase ==============
        destroy_seed = 2 * seed + it
        partial_sol, destroyed_flags, removed = destroy_operator(
            best_sol,
            destroy_proba,
            destroy_steps,
            seed=destroy_seed,
            T=destroy_T
        )
        nodes_destroyed += removed

        # ============= Temporary rebuild phase ==============
        for r_idx, was_destroyed in enumerate(destroyed_flags):
            if not was_destroyed or len(partial_sol.routes[r_idx]) <= 2:
                continue
            if rng.random() > rebuild_proba:
                continue

            partial_sol, repaired_list, new_nodes_count = repair_operator(
                partial_sol,
                route_idx=r_idx,
                steps=rebuild_steps,
                T=rebuild_T,
                seed=(destroy_seed + r_idx) if destroy_seed is not None else None,
                verbose=False
            )
            nodes_rebuilt += new_nodes_count

        # ============== Greedy build phase ==============
        sol_cand, info_cand = greedy_balanced_solver(
            prob,
            premature_routes=partial_sol.routes,
            verbose=False
        )

        total_actions += info_cand["actions_evaluated"]

        # If improved, update best solution
        if (sol_cand and sol_cand.is_valid()
            and sol_cand.max_cost < best_cost
        ):
            best_sol = sol_cand
            best_cost = sol_cand.max_cost
            improvements += 1

            if verbose:
                print(f"[iter {it}] improved best to {best_cost}")


    # Final stats
    elapsed = time.time() - start_time
    info = {
        "iterations": iterations_done,
        "improvements": improvements,
        "actions_evaluated": total_actions,
        "nodes_destroyed": nodes_destroyed,
        "nodes_rebuilt": nodes_rebuilt,
        "time": elapsed,
        "status": status,
    }

    return best_sol, info
