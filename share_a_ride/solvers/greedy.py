from time import time
from share_a_ride.problem import ShareARideProblem
from share_a_ride.solution import Solution
from utils.helper import route_cost_from_sequence

def greedy_balanced_solver(prob: ShareARideProblem) -> tuple:
    """
    Greedy balanced heuristic:
        - At each step, choose the taxi with smallest current route length.
        - For that taxi, evaluate all feasible next actions (pick/drop).
        - Choose the action with minimal added distance (inc).
        - Repeat until all passengers and parcels are served.

    Returns:
        (sol, info): tuple where
        - sol: Solution object with routes and costs.
        - Info dictionary contains:
            + iterations: number of main loop iterations
            + action_evaluations: total number of action evaluations
            + depot_returns: number of times taxis returned to depot
            + elapsed_time: total time taken
    """
    start_time = time()
    N, M, K = prob.N, prob.M, prob.K

    # state
    remaining_pass_pick = set(range(1, N + 1))
    remaining_pass_drop = set()
    remaining_parc_pick = set(range(1, M + 1))
    remaining_parc_drop = set()

    taxi_states = [
        {"route": [], "pos": 0, "len": 0, "load": 0, "passenger": 0}
        for _ in range(K)
    ]

    # Statistics tracking
    stats = {
        "iterations": 0,
        "action_evaluations": 0,
        "depot_returns": 0,
    }


    def possible_actions(t: dict, taxi_idx: int):
        """
            Generate feasible next actions for a taxi t.
            Returns a list of (action_type, node_index, incremental_cost).
        """

        pos, actions = t["pos"], []

        # passenger pickup
        for i in list(remaining_pass_pick):
            if t["passenger"] == 0:  # no passenger onboard
                inc = prob.D[pos][prob.ppick(i)]
                actions.append(("pickP", i, inc))

        # passenger drop
        for i in list(remaining_pass_drop):
            if t["passenger"] == i:  # passenger onboard
                inc = prob.D[pos][prob.pdrop(i)]
                actions.append(("dropP", i, inc))

        # parcel pickup
        for j in list(remaining_parc_pick):
            qj = prob.q[j - 1]
            if t["load"] + qj <= prob.Q[taxi_idx]:
                inc = prob.D[pos][prob.parc_pick(j)]
                actions.append(("pickL", j, inc))

        # parcel drop
        for j in list(remaining_parc_drop):
            inc = prob.D[pos][prob.parc_drop(j)]
            actions.append(("dropL", j, inc))

        return actions


    def apply_action(t_state: dict, kind: str, node_idx: int, inc: int):
        """Apply an action to taxi t and update global sets."""
        nonlocal remaining_pass_pick, remaining_pass_drop

        if kind == "pickP":
            t_state["route"].append(prob.ppick(node_idx))
            t_state["passenger"] = node_idx
            remaining_pass_pick.remove(node_idx)
            remaining_pass_drop.add(node_idx)

        elif kind == "dropP":
            t_state["route"].append(prob.pdrop(node_idx))
            t_state["passenger"] = 0
            remaining_pass_drop.remove(node_idx)

        elif kind == "pickL":
            t_state["route"].append(prob.parc_pick(node_idx))
            t_state["load"] += prob.q[node_idx - 1]
            remaining_parc_pick.remove(node_idx)
            remaining_parc_drop.add(node_idx)

        elif kind == "dropL":
            t_state["route"].append(prob.parc_drop(node_idx))
            t_state["load"] -= prob.q[node_idx - 1]
            remaining_parc_drop.remove(node_idx)

        t_state["len"] += inc
        t_state["pos"] = t_state["route"][-1]


    # Main loop: execute actions until all pickups and drops are resolved.
    while ( remaining_pass_pick or remaining_pass_drop
            or remaining_parc_pick or remaining_parc_drop):
        stats["iterations"] += 1

        # Select taxi with minimal route length.
        taxi_idx = min(range(K), key=lambda t: taxi_states[t]["len"])
        t = taxi_states[taxi_idx]

        # Get feasible actions and record how many were evaluated.
        actions = possible_actions(t, taxi_idx)
        stats["action_evaluations"] += len(actions)

        if not actions:
            # If no feasible action exists, send taxi back to depot.
            t["len"] += prob.D[t["pos"]][0]
            t["pos"] = 0
            stats["depot_returns"] += 1
            continue

        # Select action with minimal incremental cost.
        kind, idx, inc = min(actions, key=lambda x: x[2])
        apply_action(t, kind, idx, inc)

    # Build final solution.
    route_lists, lengths = [], []
    for t in taxi_states:
        if t["route"]:
            L = route_cost_from_sequence(t["route"], prob.D)
            route_lists.append([0] + t["route"] + [0])
            lengths.append(L)
        else:
            route_lists.append([0, 0])
            lengths.append(0)

    sol = Solution(prob, route_lists, lengths)
    elapsed = time() - start_time
    info = {
        "iterations": stats["iterations"],
        "action_evaluations": stats["action_evaluations"],
        "depot_returns": stats["depot_returns"],
        "elapsed_time": elapsed
    }

    assert sol.is_valid()
    return sol, info