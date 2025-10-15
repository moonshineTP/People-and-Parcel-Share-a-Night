import math
import random
from typing import List, Optional
from share_a_ride.problem import ShareARideProblem

def build(
        prob: ShareARideProblem,
        route: List[int],
        route_idx: int,
        num_actions: int = 5,
        T: float = 1.0,
        seed: int = 42,
        verbose: bool = False
    ) -> List[int]:
    """
        Build/Improve a premature route by inserting pickups/drops
        up to num_actions steps. 
        Use a temperature-based softmax heuristic to prefer lower-cost actions.
        If no feasible action remains, return to depot (0) and stop early.
    """
    rng = random.Random(seed)
    res_route = route.copy()

    # initialize state from the partial route
    remaining_pass_pick = set(range(1, prob.N + 1))
    remaining_pass_drop = set()
    remaining_parc_pick = set(range(1, prob.M + 1))
    remaining_parc_drop = set()
    passenger = 0
    load = 0

    # parse the existing nodes in premature_route
    for node in res_route:
        if prob.is_ppick(node):
            pid = prob.rev_ppick(node)
            remaining_pass_pick.discard(pid)
            remaining_pass_drop.add(pid)
            passenger = pid
        elif prob.is_pdrop(node):
            pid = prob.rev_pdrop(node)
            remaining_pass_drop.discard(pid)
            passenger = 0
        elif prob.is_parc_pick(node):
            jid = prob.rev_parc_pick(node)
            remaining_parc_pick.discard(jid)
            remaining_parc_drop.add(jid)
            load += prob.q[jid - 1]
        elif prob.is_parc_drop(node):
            jid = prob.rev_parc_drop(node)
            remaining_parc_drop.discard(jid)
            load -= prob.q[jid - 1]

    max_capacity = prob.Q[route_idx]


    for step in range(num_actions):
        # build list of feasible actions (kind, id, incremental cost)
        actions = []
        pos = res_route[-1] if res_route else 0

        # passenger pickup / drop
        if passenger == 0:
            for pid in remaining_pass_pick:
                node = prob.ppick(pid)
                inc = prob.D[pos][node]
                actions.append(("pickP", pid, inc))
        else:
            node = prob.pdrop(passenger)
            inc = prob.D[pos][node]
            actions.append(("dropP", passenger, inc))

        # parcel pickup
        for jid in remaining_parc_pick:
            w = prob.q[jid - 1]
            if load + w <= max_capacity:
                node = prob.parc_pick(jid)
                inc = prob.D[pos][node]
                actions.append(("pickL", jid, inc))

        # parcel drop
        for jid in remaining_parc_drop:
            node = prob.parc_drop(jid)
            inc = prob.D[pos][node]
            actions.append(("dropL", jid, inc))

        if not actions:
            if res_route[-1] != 0:
                res_route.append(0)
            break

        # temperature‐based weighting
        incs = [inc for _, _, inc in actions]
        min_inc, max_inc = min(incs), max(incs)
        range_inc = max_inc - min_inc

        if range_inc < 1e-6:
            # all inc equal → uniform weights
            weights = [1.0] * len(actions)
        else:
            weights = []
            for _, _, inc in actions:
                # map inc → [0,1], higher inc → smaller normalized
                normalized = (max_inc - inc) / range_inc
                # boost small normalized with epsilon, apply temp exponent
                weights.append((normalized + 0.1) ** (1.0 / max(T, 1e-6)))

        # sample one action by these weights
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0.0
        for idx_w, w in enumerate(weights):
            cum += w
            if r <= cum:
                kind, eid, inc = actions[idx_w]
                break

        # apply chosen action
        if kind == "pickP":
            res_route.append(prob.ppick(eid))
            passenger = eid
            remaining_pass_pick.remove(eid)
            remaining_pass_drop.add(eid)
        elif kind == "dropP":
            res_route.append(prob.pdrop(eid))
            passenger = 0
            remaining_pass_drop.remove(eid)
        elif kind == "pickL":
            res_route.append(prob.parc_pick(eid))
            load += prob.q[eid - 1]
            remaining_parc_pick.remove(eid)
            remaining_parc_drop.add(eid)
        elif kind == "dropL":
            res_route.append(prob.parc_drop(eid))
            load -= prob.q[eid - 1]
            remaining_parc_drop.remove(eid)

    # ensure route ends at depot
    if res_route[-1] != 0:
        res_route.append(0)

    return res_route