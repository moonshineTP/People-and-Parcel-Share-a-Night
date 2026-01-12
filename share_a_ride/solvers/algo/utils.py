"""
Utility functions for solver algorithms.
"""
import random
from typing import List, Optional, Tuple, Union

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, Solution
from share_a_ride.core.utils import generate_instance_coords
from share_a_ride.solvers.utils.weighter import weighted


Action = Tuple[int, str, int, int]

exact_problems: List[ShareARideProblem] = [
    generate_instance_coords(
        N = 3,
        M = 2,
        K = 2,
        area = 100,
        seed = 10,
    ),
    generate_instance_coords(
        N = 4,
        M = 3,
        K = 3,
        area = 100,
        seed = 11,
    ),
    generate_instance_coords(
        N = 6,
        M = 4,
        K = 2,
        area = 100,
        seed = 12,
    ),
    generate_instance_coords(
        N = 6,
        M = 4,
        K = 3,
        area = 100,
        seed = 13,
    ),
    generate_instance_coords(
        N = 8,
        M = 5,
        K = 2,
        area = 100,
        seed = 14,
    ),
    generate_instance_coords(
        N = 8,
        M = 5,
        K = 3,
        area = 100,
        seed = 15,
    ),
]

test_problem: ShareARideProblem = generate_instance_coords(
    N = 33,
    M = 17,
    K = 7,
    area = 1000,
    seed = 20,
)

relay_problems: List[ShareARideProblem] = [
    generate_instance_coords(
        N=11,
        M=7,
        K=2,
        area=1000,
        seed=200
    ),
    generate_instance_coords(
        N=19,
        M=11,
        K=3,
        area=1000,
        seed=210
    ),
    generate_instance_coords(
        N=31,
        M=19,
        K=5,
        area=1000,
        seed=220
    ),
    generate_instance_coords(
        N=69,
        M=31,
        K=7,
        area=1000,
        seed=230
    ),
    generate_instance_coords(
        N=181,
        M=69,
        K=11,
        area=1000,
        seed=240
    ),
]




def balanced_scorer(
        partial: Union[PartialSolution, Solution],
        sample_size: int = 8,
        w_std: float = 0.15,
        seed: Optional[int] = None,
    ) -> float:
    """
    Score a PartialSolution using a stochastic surrogate of the route-cost spread.

    Route costs are sampled with replacement to estimate the standard deviation,
    and the final score is ``max_cost + std_weight * std``.
    """
    rng = random.Random(seed)
    costs = sorted(partial.route_costs)
    if len(costs) == 1:
        return partial.max_cost

    sampled = rng.choices(costs, k=sample_size)
    mean = sum(sampled) / sample_size
    variance = sum((value - mean)**2 for value in sampled) / sample_size
    std_dev = variance ** 0.5

    return partial.max_cost + w_std * std_dev


def check_general_action(partial: PartialSolution, action: Action) -> bool:
    """
    Check if ``action`` is valid for ``partial`` solution.
    Delegates to PartialSolution.check_action or check_return for validation.
    """
    taxi, kind, node_idx, _inc = action

    if kind == "return":
        return partial.check_return(taxi)

    return partial.check_expand(taxi, kind, node_idx)


def apply_general_action(partial: PartialSolution, action: Action) -> None:
    """
    Apply ``action`` to ``partial`` solution in-place.
    """
    taxi, kind, node_idx, inc = action

    if kind == "return":
        partial.apply_return(taxi)
    else:
        partial.apply_extend(taxi, kind, node_idx, inc)


def enumerate_actions_greedily(
        partial: PartialSolution,
        width: Optional[int] = None,
        asymmetric: bool = True,
    ) -> List[Action]:
    """
    Greedy, beam-style enumeration of next actions.

    This mirrors the pragmatic expansion used in the solvers:
    1. Consider only active taxis and sort them by current route cost.
       If ``asymmetric`` is True, taxis with identical route configurations are
       filtered out to reduce symmetry.
    2. Enumerate candidate (non-return) actions in two passes:
       - Aggressive pass: limit the number of taxis and actions per taxi.
       - Fallback pass: enumerate without those limits if the aggressive pass
         yields no actions.
    3. Prioritize actions that keep routes balanced (new cost <= current max cost)
       before other actions, sort each

    If no non-return actions exist, return-to-depot actions are generated for
    eligible taxis (empty load) as a fallback.
    """
    if width is None:
        width = 10**9    # effectively unlimited

    problem = partial.problem


    # //// Extract and filter taxi order
    # Extract active taxis and sort by current route cost
    active_taxis = [
        idx for idx, state in enumerate(partial.states)
        if not state["ended"]
    ]
    if not active_taxis:
        return []
    taxi_order = sorted(active_taxis, key=lambda idx: partial.route_costs[idx])
    num_taxis = len(taxi_order)


    # Filter taxis with identical route configurations
    if asymmetric:
        seen_configs: set = set()
        filtered: List[int] = []
        for t_idx in taxi_order:
            route_config = tuple(partial.routes[t_idx])
            if route_config in seen_configs:
                continue
            seen_configs.add(route_config)
            filtered.append(t_idx)

        taxi_order = filtered


    # //// Heuristic expansion ////
    # Taxi limit
    def taxi_limit(aggressive: bool) -> int:
        # Keep the existing scaling heuristic but never exceed available actions.
        if not aggressive:
            return num_taxis

        return min(
            2 if problem.K >= 25
            else 3 if problem.K >= 12
            else 4 if problem.K >= 6
            else 5,
            num_taxis
        )

    # Action per taxi limit
    def action_per_taxi_limit(aggressive: bool) -> int:
        if not aggressive:
            return 10**9    # effectively unlimited

        return min(
            2 if problem.num_nodes >= 500
            else 4 if problem.num_nodes >= 200
            else 6 if problem.num_nodes >= 100
            else 8 if problem.num_nodes >= 50
            else 12 if problem.num_nodes >= 25
            else 16,
            width,
        )

    # Enumerate actions with given aggressiveness
    def enumerate_pass(aggressive: bool) -> List[Action]:
        expansions: List[Action] = []
        taxi_used = taxi_order
        if aggressive:
            taxi_used = taxi_order[:taxi_limit(aggressive=True)]

        for t_idx in taxi_used:
            # Extract actions for taxi
            assigned_actions = partial.possible_expand(t_idx)
            if aggressive:
                assigned_actions = sorted(
                    assigned_actions,
                    key=lambda item: item[2]
                )[:action_per_taxi_limit(aggressive=True)]

            # Convert to general action format
            general_actions: List[Action] = [
                (t_idx, kind, node_idx, inc)
                for kind, node_idx, inc in assigned_actions
            ]

            # Extend, sort and filter
            expansions.extend(general_actions)
            expansions.sort(key=lambda item: item[3])
            expansions = expansions[:width]

        return expansions


    # //// Perform enumeration passes
    expansions = enumerate_pass(aggressive=True)
    if not expansions:
        expansions = enumerate_pass(aggressive=False)


    # //// Prioritize actions ////
    current_max = partial.max_cost
    prioritized: List[Tuple[float, Action]] = []
    secondary: List[Tuple[float, Action]] = []

    for action in expansions:
        t_idx, kind, node_idx, inc = action
        weight = weighted(kind, inc)
        pair = (weight, action)

        if partial.route_costs[t_idx] + inc <= current_max:
            prioritized.append(pair)
        else:
            secondary.append(pair)

    prioritized.sort(key=lambda item: item[0])
    secondary.sort(key=lambda item: item[0])

    # //// Collect final actions ////
    final_actions = [action for _, action in prioritized + secondary][:width]


    # If no expansions found, consider return-to-depot actions
    if not final_actions:
        # If not enough actions to cover all nodes, raise error
        if partial.num_actions < problem.num_expansions:
            print("[Warning] No feasible actions found before closing depth")
            raise RuntimeError("Premature routes not covering all nodes.")

        # Collect return-to-depot actions
        return_actions: List[Action] = []
        for t_idx in taxi_order:
            state = partial.states[t_idx]
            if partial.check_return(t_idx):
                inc_back = problem.D[state["pos"]][0]
                return_actions.append((t_idx, "return", 0, inc_back))

        return return_actions[:width]

    # Else, return final actions
    return final_actions




# //// Testing functions
# def relay():
    # """
    # Relay test on multiple problems and multiple solvers in the module
    # """
    # import time
    # from share_a_ride.solvers.algo.aco import aco_solver
    # from share_a_ride.solvers.algo.astar import astar_solver
    # from share_a_ride.solvers.algo.beam import beam_solver
    # from share_a_ride.solvers.algo.greedy import iterative_greedy_solver
    # from share_a_ride.solvers.algo.mcts import mcts_solver

    # solvers = [
    #     ("ACO", aco_solver),
    #     ("A*", astar_solver),
    #     ("Beam", beam_solver),
    #     ("Greedy", iterative_greedy_solver),
    #     ("MCTS", mcts_solver),
    # ]

    # print(f"{'Problem':<20} | {'Solver':<10} | {'Cost':<10} | {'time':<10} | {'status':<10}")
    # print("-" * 75)


    # # //// Iterate over prolbems
    # for prob in relay_problems:
    #     prob_name = f"N={prob.N}, M={prob.M}, K={prob.K}"

    #     # Iterate over solvers
    #     for name, solver in solvers:
    #         try:
    #             # Measure the solver
    #             start = time.time()
    #             sol, _ = solver(prob, time_limit=30, seed=42)
    #             elapsed = time.time() - start
    #             elapsed_str = f"{elapsed:.2f}s"

    #             # Analyze the result
    #             cost = sol.max_cost if sol else "N/A"
    #             status = "Found" if sol else "Failed"

    #             # Print result
    #             print(
    #                 f"{prob_name:<20} | {name:<10} | {cost:<10} "
    #                 f"| {elapsed_str:<10} | {status:<10}"
    #             )

    #         except Exception as e:
    #             print(
    #                 f"{prob_name:<20} | {name:<10} | {'Error':<10} "
    #                 f"| {'N/A':<10} | {str(e):<10}"
    #             )




# ================ Playground ================
if __name__ == "__main__":
    # relay()
    pass
