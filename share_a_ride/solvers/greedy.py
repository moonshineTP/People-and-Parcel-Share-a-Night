from share_a_ride.problem import ShareARideProblem
from share_a_ride.solution import Solution
from utils.helper import route_length_from_sequence

def greedy_balanced_solver(prob: ShareARideProblem) -> Solution:
    """
    Greedy balanced heuristic:
      - At each step, choose the taxi with smallest current route length.
      - For that taxi, evaluate all feasible next actions (pick/drop).
      - Choose the action with minimal added distance (inc).
      - Repeat until all passengers and parcels are served.
    """
    N, M, K = prob.N, prob.M, prob.K

    # state
    remaining_pass_pick = set(range(1, N + 1))
    remaining_pass_drop = set()
    remaining_parc_pick = set(range(1, M + 1))
    remaining_parc_drop = set()

    taxis = [
        {"route": [], "pos": 0, "len": 0, "load": 0, "passenger": 0}
        for _ in range(K)
    ]

    def possible_actions(t: dict, taxi_idx: int):
        """Generate feasible next actions for a taxi t."""
        pos, actions = t["pos"], []

        # passenger pickup
        for i in list(remaining_pass_pick):
            if t["passenger"] == 0:
                inc = prob.D[pos][prob.ppick(i)]
                actions.append(("pickP", i, inc))

        # passenger drop
        for i in list(remaining_pass_drop):
            if t["passenger"] == i:
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

    def apply_action(t: dict, kind: str, idx: int, inc: int, taxi_idx: int):
        """Apply an action to taxi t and update global sets."""
        if kind == "pickP":
            t["route"].append(prob.ppick(idx))
            t["passenger"] = idx
            remaining_pass_pick.remove(idx)
            remaining_pass_drop.add(idx)

        elif kind == "dropP":
            t["route"].append(prob.pdrop(idx))
            t["passenger"] = 0
            remaining_pass_drop.remove(idx)

        elif kind == "pickL":
            t["route"].append(prob.parc_pick(idx))
            t["load"] += prob.q[idx - 1]
            remaining_parc_pick.remove(idx)
            remaining_parc_drop.add(idx)

        elif kind == "dropL":
            t["route"].append(prob.parc_drop(idx))
            t["load"] -= prob.q[idx - 1]
            remaining_parc_drop.remove(idx)

        t["len"] += inc
        t["pos"] = t["route"][-1]

    # main loop
    while (
        remaining_pass_pick
        or remaining_pass_drop
        or remaining_parc_pick
        or remaining_parc_drop
    ):
        taxi_idx = min(range(K), key=lambda t: taxis[t]["len"])
        t = taxis[taxi_idx]
        actions = possible_actions(t, taxi_idx)

        if not actions:
            # return to depot if stuck
            t["len"] += prob.D[t["pos"]][0]
            t["pos"] = 0
            continue

        kind, idx, inc = min(actions, key=lambda x: x[2])
        apply_action(t, kind, idx, inc, taxi_idx)

    # build Solution
    route_lists, lengths = [], []
    for t in taxis:
        if t["route"]:
            L = route_length_from_sequence(t["route"], prob.D)
            route_lists.append([0] + t["route"] + [0])
            lengths.append(L)
        else:
            route_lists.append([0, 0])
            lengths.append(0)

    return Solution(prob, route_lists, lengths)