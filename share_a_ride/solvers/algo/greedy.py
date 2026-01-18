"""
Greedy balanced heuristic and its iterative improvement.
"""
import time
import random
from typing import Any, Optional, Tuple, Dict

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, Solution

from share_a_ride.solvers.algo.utils import apply_general_action, enumerate_actions_greedily
from share_a_ride.solvers.operator.repair import repair_one_route
from share_a_ride.solvers.operator.destroy import destroy_operator
from share_a_ride.solvers.operator.relocate import relocate_operator
from share_a_ride.solvers.utils.weighter import softmax_weighter
from share_a_ride.solvers.utils.sampler import sample_from_weight



def greedy_solver(
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        num_actions: int = 5,
        t_actions: float = 0.01,
        seed: Optional[int] = None,
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
    start = time.time()
    rng = random.Random(seed)

    # Initialize partial solution if not provided
    if partial is None:
        partial = PartialSolution(problem=problem)


    # //// Main greedy loop ////
    iterations = 0
    pre_depth = partial.num_actions
    while partial.is_pending():
        iterations += 1

        actions = enumerate_actions_greedily(partial, num_actions)
        expansions = [a for a in actions if a[1] != "return"]
        if not expansions:
            expansions = actions
        if not expansions:
            if verbose:
                print("[Greedy] [Error] The partial has no feasible actions available.")

            return None, {
                "iterations": iterations,
                "time": time.time() - start,
                "actions_done": partial.num_actions - pre_depth,
                "status": "error",
            }

        # Else, choose action with minimal added cost (sampled via softmax)
        incs = [a[3] for a in expansions]
        weights = softmax_weighter(incs, t_actions)
        act_idx = sample_from_weight(rng, weights)
        action = expansions[act_idx]

        apply_general_action(partial, action)

        if verbose:
            taxi, kind, idx, _inc = action
            print(
                f"[Greedy] [Depth {partial.num_actions}] "
                f"Taxi {taxi} extended route with action {kind} on passenger/parcel {idx}"
            )

    # Finalize
    sol = partial.to_solution()
    stats = {
        "iterations": iterations,
        "time": time.time() - start,
        "actions_done": partial.num_actions - pre_depth,
        "status": "done",
    }

    # Summary
    if verbose:
        print()
        print("[Greedy] Completed.")
        print(f"[Greedy] Solution max cost: {sol.max_cost if sol else 'N/A'}")
        print(f"[Greedy] Time taken: {stats['time']:.4f} seconds")
        print("------------------------------")
        print()

    return sol, stats




def iterative_greedy_solver(    # Actually quite like a large neighborhood search
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        iterations: int = 10000,
        num_actions: int = 5,
        t_actions: float = 0.01,
        destroy_proba: float = 0.53,
        destroy_steps: int = 13,
        destroy_t: float = 1.3,
        rebuild_proba: float = 0.29,
        rebuild_steps: int = 3,
        rebuild_t: float = 1.2,
        time_limit: float = 30.0,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Iterative improvement of a greedy solution using destroy and rebuild operators

    At each iteration:
    - Destroy a fraction of routes (randomly selected based on cost)
    - Rebuild those routes partially with some randomness
    - Complete the solution with greedy balanced heuristic
    - If improved, update the best solution found.

    Args:
        - problem: ShareARideProblem instance
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
    
    Returns: (sol, info): tuple where
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
    start = time.time()
    deadline = start + time_limit
    rng = random.Random(seed)

    # Validate parameters
    assert 1e-5 < destroy_proba < 1 - 1e-5
    assert 1e-5 < rebuild_proba < 1 - 1e-5
    assert 1 <= rebuild_steps <= destroy_steps

    # Initialize partial solution if not provided
    if partial is None:
        partial = PartialSolution(problem=problem, routes=[])


    # //// Greedy initialization ////
    best_sol, greedy_info = greedy_solver(
        problem=problem,
        partial=partial,
        num_actions=num_actions,
        t_actions=t_actions,
        seed=3 * seed if seed else None,
        verbose=verbose
    )
    if not best_sol:
        return None, {"time": time.time() - start, "status": "error"}
    best_cost = best_sol.max_cost

    if verbose:
        print(f"[Iterative Greedy] [Iter 0] initial best cost: {best_cost}")


    # //// Main iterative loop ////
    actions = greedy_info["actions_done"]
    all_cost = best_cost
    improvements = 0
    nodes_destroyed = 0
    nodes_rebuilt = 0
    status = "done"
    iterations_done = 0

    for it in range(1, iterations + 1):
        if deadline and time.time() >= deadline:
            status = "overtime"
            break

        # Seed for operators
        operators_seed = None if seed is None else 2 * seed + 98 * it


        # //// Destroy phase
        destroyed_partial, destroyed_flags, removed = destroy_operator(
            best_sol,
            destroy_proba,
            destroy_steps,
            seed=operators_seed,
            t=destroy_t
        )
        nodes_destroyed += removed
        actions += removed


        # //// Temporary rebuild phase
        rebuilt_partial = destroyed_partial
        for r_idx, was_destroyed in enumerate(destroyed_flags):
            if not was_destroyed:
                continue
            if rng.random() > rebuild_proba:    # Sample rebuild
                continue

            rebuilt_partial, new_actions_count = repair_one_route(
                rebuilt_partial,
                route_idx=r_idx,
                steps=rebuild_steps,
                T=rebuild_t,
                seed=operators_seed,
            )
            nodes_rebuilt += new_actions_count
            actions += new_actions_count


        # //// Greedy completion phase (no randomness)
        new_sol, new_info = greedy_solver(
            problem,
            partial=rebuilt_partial,
            num_actions=1,
            verbose=False
        )
        all_cost += new_sol.max_cost if new_sol else 0
        actions += new_info["actions_done"]
        iterations_done += 1 if new_sol else 0


        # //// Improvement check
        if new_sol and new_sol.max_cost < best_cost:
            best_sol = new_sol
            best_cost = new_sol.max_cost
            improvements += 1

            # Verbose output
            if verbose:
                print(f"[Iterative Greedy] [Iter {it}] improved best to {best_cost}")

    # Final stats
    elapsed = time.time() - start
    stats = {
        "iterations": iterations_done,
        "actions_done": actions,
        "improvements": improvements,
        "actions_destroyed": nodes_destroyed,
        "actions_rebuilt": nodes_rebuilt,
        "average_cost": all_cost / (iterations_done + 1),   # including initial
        "time": elapsed,
        "status": status,
    }

    # Summary
    if verbose:
        print()
        print(f"[Iterative Greedy] Finished after {iterations_done} iterations.")
        print(
            f"[Iterative Greedy] Best solution max cost: "
            f"{best_sol.max_cost if best_sol else 'N/A'}."
        )
        print(f"[Iterative Greedy] Time taken: {elapsed:.4f} seconds.")
        print("------------------------------")
        print()

    return best_sol, stats




# ================ Playground ================
if __name__ == "__main__":
    from share_a_ride.solvers.algo.utils import test_problem

    # Run greedy solver
    solution, info = iterative_greedy_solver(test_problem, verbose=True)
