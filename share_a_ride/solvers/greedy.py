from time import time
from share_a_ride.problem import ShareARideProblem
from share_a_ride.solution import Solution
from utils.helper import route_cost_from_sequence

def greedy_balanced_solver(prob: ShareARideProblem, verbose: bool = False) -> tuple:
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
            + elapsed_time: total time taken
    """
    start_time = time()
    N, M, K = prob.N, prob.M, prob.K

    # state
    remaining_pass_pick = set(range(1, N + 1))
    remaining_parc_pick = set(range(1, M + 1))
    taxi_current_pass = [0] * K
    taxi_current_parc = [set() for _ in range(K)]

    taxi_states = [
        {   
            "route": [0], 
            "pos": 0,
            "cost": 0, 
            "load": 0, 
            "passenger": 0, 
            "ended": False
        }
        for _ in range(K)
    ]

    # Statistics tracking
    stats = {
        "iterations": 0,
        "actions_evaluated": 0,
    }


    def possible_actions(t_state: dict, t_idx: int):
        """
            Generate feasible next actions for a taxi t.
            Returns a list of (action_type, node_index, incremental_cost).
        """
        pos, actions = t_state["pos"], []

        # passenger pickup
        for i in list(remaining_pass_pick):
            if t_state["passenger"] == 0:  # no passenger onboard
                inc = prob.D[pos][prob.ppick(i)]
                actions.append(("pickP", i, inc))

        # passenger drop
        if t_state["passenger"] > 0:  # passenger onboard
            inc = prob.D[pos][prob.pdrop(t_state["passenger"])]
            actions.append(("dropP", t_state["passenger"], inc))

        # parcel pickup
        for j in list(remaining_parc_pick):
            qj = prob.q[j - 1]
            if t_state["load"] + qj <= prob.Q[t_idx]:
                inc = prob.D[pos][prob.parc_pick(j)]
                actions.append(("pickL", j, inc))

        # parcel drop
        for j in taxi_current_parc[t_idx]:
            inc = prob.D[pos][prob.parc_drop(j)]
            actions.append(("dropL", j, inc))

        return actions


    def apply_action(t_state: dict, t_idx: int, kind: str, node_idx: int, inc: int):
        """Apply an action to taxi t and update global sets."""
        nonlocal remaining_pass_pick, remaining_parc_pick
        nonlocal taxi_current_pass, taxi_current_parc

        if kind == "pickP":
            t_state["route"].append(prob.ppick(node_idx))
            t_state["passenger"] = node_idx
            remaining_pass_pick.remove(node_idx)
            taxi_current_pass[t_idx] = node_idx

        elif kind == "dropP":
            t_state["route"].append(prob.pdrop(node_idx))
            t_state["passenger"] = 0
            taxi_current_pass[t_idx] = 0

        elif kind == "pickL":
            t_state["route"].append(prob.parc_pick(node_idx))
            t_state["load"] += prob.q[node_idx - 1]
            remaining_parc_pick.remove(node_idx)
            taxi_current_parc[t_idx].add(node_idx)

        elif kind == "dropL":
            t_state["route"].append(prob.parc_drop(node_idx))
            t_state["load"] -= prob.q[node_idx - 1]
            taxi_current_parc[t_idx].remove(node_idx)

        else:
            raise ValueError(f"Unknown action kind: {kind}")

        t_state["cost"] += inc
        t_state["pos"] = t_state["route"][-1]


    # Main loop: execute actions until all pickups and drops are resolved.
    while ( remaining_pass_pick or any(taxi_current_pass)
            or remaining_parc_pick or any(taxi_current_pass)):

        stats["iterations"] += 1

        # List all available taxis
        available_taxis = [
            t_idx for t_idx, t_state in enumerate(taxi_states)
            if not t_state["ended"]
        ]

        # Select taxi with minimal route cost.
        argmin_t_idx = min(available_taxis, key=lambda t_idx: taxi_states[t_idx]["cost"])
        argmin_t_state = taxi_states[argmin_t_idx]

        # Get feasible actions and record how many were evaluated.
        actions = possible_actions(argmin_t_state, argmin_t_idx)
        stats["actions_evaluated"] += len(actions)

        if verbose:
            print(f"Taxi with min cost: {argmin_t_idx}")
            print(f"Actions available: {actions}")

        # If no feasible action exists, send taxi back to depot.
        if not actions:
            assert argmin_t_state["passenger"] == 0
            assert argmin_t_state["load"] == 0

            argmin_t_state["route"].append(0)
            argmin_t_state["pos"] = 0
            argmin_t_state["cost"] += prob.D[argmin_t_state["pos"]][0]
            argmin_t_state["ended"] = True

            continue

        # Select action with minimal incremental cost.
        kind, idx, inc = min(actions, key=lambda x: x[2])
        apply_action(argmin_t_state, argmin_t_idx, kind, idx, inc)
        
        if verbose:
            print(f"Taxi: {argmin_t_idx}: {argmin_t_state['route']}")
            print()

    for t_state in taxi_states:
        if not t_state["ended"]:
            t_state["route"].append(0)
            t_state["ended"] = True

    if verbose:
        print("All tasks completed.")


    # Build final solution.
    routes, route_costs = [], []
    for t_state in taxi_states:
        routes.append(t_state["route"])
        cost = route_cost_from_sequence(t_state["route"], prob.D)
        route_costs.append(cost)

    sol = Solution(prob, routes, route_costs)
    elapsed = time() - start_time
    info = {
        "iterations": stats["iterations"],
        "actions_evaluated": stats["actions_evaluated"],
        "elapsed_time": elapsed
    }
    if sol:
        if not sol.is_valid():
            sol = None


    assert sol.is_valid() if sol else True
    return sol, info