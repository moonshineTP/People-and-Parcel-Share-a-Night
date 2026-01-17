import sys
import math
import random
import time
import bisect
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Iterator

Request = Tuple[int, int, str]
SwapRequest = Tuple[int, int, str]
Action = Tuple[int, str, int, int]
RelocateRequest = Tuple[int, int, int, int, str]
CostChange = Tuple[int, int, int]
RelocateAction = Tuple[RelocateRequest, CostChange]


class ShareARideProblem:
    def __init__(
            self, N: int, M: int, K: int,
            parcel_qty: List[int], vehicle_caps: List[int], dist: List[List[int]],
            coords: Optional[List[Tuple[int, int]]] = None, name: Optional[str] = None
        ):
        self.N = N
        self.M = M
        self.K = K
        self.q = list(parcel_qty)
        self.Q = list(vehicle_caps)
        self.D = [row[:] for row in dist]
        self.num_nodes = 2*N + 2*M + 1
        self.num_requests = N + M
        self.num_actions = N + 2*M + K
        self.num_expansions = N + 2*M

        self.pserve = lambda pid: (pid, pid + N + M)
        self.lpick = lambda lid: N + lid
        self.ldrop = lambda lid: 2*N + M + lid

        self.rev_ppick = lambda nodeid: nodeid
        self.rev_pdrop = lambda nodeid: nodeid - (N + M)
        self.rev_lpick = lambda n: n - N
        self.rev_ldrop = lambda n: n - (2 * N + M)

        self.is_ppick = lambda nodeid: 1 <= nodeid <= N
        self.is_pdrop = lambda nodeid: N + M + 1 <= nodeid <= 2 * N + M
        self.is_lpick = lambda nodeid: N + 1 <= nodeid <= N + M
        self.is_ldrop = lambda nodeid: 2 * N + M + 1 <= nodeid <= 2 * (N + M)

        self.coords = coords
        self.name = name


    def is_valid(self) -> bool:
        if len(self.q) != self.M:
            return False
        if len(self.Q) != self.K:
            return False
        if len(self.D) != self.num_nodes:
            return False
        if not all(len(row) == self.num_nodes for row in self.D):
            return False

        return True


    def copy(self):
        return ShareARideProblem(
            self.N, self.M, self.K,
            self.q[:], self.Q[:], [row[:] for row in self.D],
            self.coords, self.name
        )




def route_cost_from_sequence(
        seq: List[int],
        D: List[List[int]],
        verbose: bool = False
    ) -> int:
    prev, total_cost = 0, 0
    for node in seq[1:]:
        total_cost += D[prev][node]
        prev = node

    return total_cost




class Solution:
    def __init__(self, problem: ShareARideProblem,
                routes: List[List[int]], route_costs: Optional[List[int]] = None):

        if not routes:
            raise ValueError("Routes list cannot be empty.")
        if len(routes) != problem.K:
            raise ValueError(f"Expected {problem.K} routes, got {len(routes)}.")

        if not route_costs:
            new_route_costs = [
                route_cost_from_sequence(route, problem.D)
                for route in routes
            ]
        else:
            new_route_costs = route_costs

        self.problem = problem
        self.routes = routes
        self.route_costs = new_route_costs
        self.num_actions = problem.num_actions
        self.max_cost = max(new_route_costs) if new_route_costs else 0


    def is_valid(self) -> bool:
        prob = self.problem
        K = prob.K

        if len(self.routes) != K:
            return False

        visited_nodes = set()
        for route_idx, route in enumerate(self.routes):

            if not (route[0] == 0 and route[-1] == 0):
                return False


            route_len = len(route)
            parcel_load = 0
            parcel_onboard = set()


            for idx, node in enumerate(route[1:-1], start=1):


                if node in visited_nodes:
                    return False
                visited_nodes.add(node)

                if prob.is_ppick(node):
                    pid = prob.rev_ppick(node)


                    drop_node = prob.pserve(pid)[1]
                    if idx + 1 >= route_len or route[idx + 1] != drop_node:
                        return False


                elif prob.is_pdrop(node):
                    pass


                elif prob.is_lpick(node):
                    lid = prob.rev_lpick(node)

                    if lid in parcel_onboard:
                        return False
                    parcel_load += prob.q[lid - 1]
                    if parcel_load > prob.Q[route_idx]:
                        return False

                    parcel_onboard.add(lid)


                elif prob.is_ldrop(node):
                    lid = prob.rev_ldrop(node)

                    if lid not in parcel_onboard:
                        return False

                    if parcel_load - prob.q[lid - 1] < 0:
                        return False
                    parcel_load -= prob.q[lid - 1]
                    parcel_onboard.remove(lid)


            if parcel_load != 0:
                return False


        if len(visited_nodes) != prob.num_requests * 2:
            return False

        return True


    def stdin_print(self, verbose: bool = False):
        if verbose:
            print("Max cost:", self.max_cost)
            print()

        print(self.problem.K)
        for route, cost in zip(self.routes, self.route_costs):
            print(len(route))
            print(" ".join(map(str, route)))



class PartialSolution:


    def __init__(
            self,
            problem: ShareARideProblem,
            routes: Optional[List[List[int]]] = None,
        ):


        self.problem = problem
        self.routes = self._init_routes(routes)
        self.route_costs = self._init_costs(routes)


        self.min_cost = min(self.route_costs)
        self.max_cost = max(self.route_costs)
        self.avg_cost = sum(self.route_costs) / problem.K


        self.node_assignment = self._init_node_assignment()
        self.remaining_pass_serve, self.remaining_parc_pick, self.remaining_parc_drop, \
        self.states, self.num_actions = self._init_states()


    def _init_routes(self, routes: Optional[List[List[int]]] = None):
        K = self.problem.K


        if not routes:
            return [[0] for _ in range(K)]
        if len(routes) != K:
            raise ValueError(f"Expected {K} routes, got {len(routes)}.")
        for route in routes:
            if not route:
                raise ValueError("One route cannot be null")
            elif route[0] != 0:
                raise ValueError("Each route must start at depot 0.")

        return routes


    def _init_costs(self, routes: Optional[List[List[int]]] = None):
        if not routes:
            return [0] * self.problem.K
        if len(routes) != self.problem.K:
            raise ValueError("Mismatch between routes and route_costs length.")

        return [route_cost_from_sequence(route, self.problem.D) for route in routes]


    def _init_node_assignment(self):
        node_count = len(self.problem.D)
        assignment = [-1] * node_count
        for idx, route in enumerate(self.routes):
            for node in route[1:]:
                if node == 0 or node >= node_count:
                    continue
                assignment[node] = idx
        return assignment


    def _init_states(self):
        prob = self.problem


        remaining_pass_serve = set(range(1, prob.N + 1))
        remaining_parc_pick = set(range(1, prob.M + 1))
        remaining_parc_drop = set()
        route_states = []
        num_actions = 0



        for _, route in enumerate(self.routes):
            route_len = len(route)
            onboard_parcels = set()
            current_load = 0


            for idx, nodeid in enumerate(route[1:], start=1):
                if prob.is_ppick(nodeid):
                    pid = prob.rev_ppick(nodeid)
                    dropid = prob.pserve(pid)[1]
                    if idx + 1 >= route_len:
                        raise RuntimeError(
                            "Invalid route: passenger pickup not followed by drop."
                        )
                    if route[idx + 1] != dropid:
                        raise RuntimeError(
                            "Invalid route: passenger pickup not followed by correct drop."
                        )

                    num_actions += 1

                    remaining_pass_serve.discard(pid)

                elif prob.is_pdrop(nodeid):
                    pass

                elif prob.is_lpick(nodeid):
                    lid = prob.rev_lpick(nodeid)

                    onboard_parcels.add(lid)
                    current_load += prob.q[lid - 1]
                    num_actions += 1

                    remaining_parc_pick.discard(lid)
                    remaining_parc_drop.add(lid)

                elif prob.is_ldrop(nodeid):
                    lid = prob.rev_ldrop(nodeid)

                    if lid not in onboard_parcels:
                        raise RuntimeError(
                            "Invalid route: parcel drop without prior pickup."
                        )

                    onboard_parcels.remove(lid)
                    current_load -= prob.q[lid - 1]
                    num_actions += 1

                    remaining_parc_drop.discard(lid)

                else:
                    if nodeid != 0:
                        raise RuntimeError(
                            "Invalid route: node id out of range."
                        )
                    num_actions += 1


            pos = route[-1]
            ended = route_len > 1 and route[-1] == 0
            state = {
                "pos": pos,
                "parcels": onboard_parcels,
                "load": current_load,
                "actions": num_actions,
                "ended": ended
            }
            route_states.append(state)



        return (
            remaining_pass_serve,
            remaining_parc_pick,
            remaining_parc_drop,
            route_states,
            num_actions
        )


    def is_valid(self, verbose: bool = False) -> bool:
        prob = self.problem
        N, M, K = prob.N, prob.M, prob.K


        if not len(self.routes) == len(self.states) == len(self.route_costs) == K:
            return False
        if len(self.node_assignment) != len(prob.D):
            return False

        expected_pass_serve = set(range(1, N + 1))
        expected_parc_pick = set(range(1, M + 1))
        expected_parc_drop: set[int] = set()


        node_assignment_check = [-1] * len(prob.D)
        total_actions = 0
        max_cost_check = 0
        cost_sum = 0
        for t_idx in range(K):
            route = self.routes[t_idx]
            route_len = len(route)
            state = self.states[t_idx]


            if not route or route[0] != 0:


                return False
            is_ended = route_len > 1 and route[-1] == 0
            if state["ended"] != is_ended:
                return False


            parcel_onboard: set[int] = set()
            route_len = len(route)
            load = 0
            prev = route[0]
            cumulated_cost = 0


            for idx, node in enumerate(route[1:], start=1):

                if not 0 <= node < prob.num_nodes:
                    return False

                if node != 0:
                    assigned = node_assignment_check[node]
                    if assigned not in (-1, t_idx):
                        return False
                    node_assignment_check[node] = t_idx


                cumulated_cost += prob.D[prev][node]
                prev = node
                if prob.is_ppick(node):
                    pid = prob.rev_ppick(node)

                    if idx + 1 < route_len:
                        next_node = route[idx+1]
                        drop_node = prob.pserve(pid)[1]
                        if next_node != drop_node:
                            return False

                    total_actions += 1
                    expected_pass_serve.discard(pid)


                elif prob.is_pdrop(node):
                    pass


                elif prob.is_lpick(node):
                    lid = prob.rev_lpick(node)
                    if lid in parcel_onboard:
                        return False


                    load += prob.q[lid - 1]
                    if load > prob.Q[t_idx]:
                        return False


                    total_actions += 1
                    parcel_onboard.add(lid)
                    expected_parc_pick.discard(lid)
                    expected_parc_drop.add(lid)


                elif prob.is_ldrop(node):
                    lid = prob.rev_ldrop(node)
                    if lid not in parcel_onboard:
                        return False

                    load -= prob.q[lid - 1]
                    if load < 0:
                        return False

                    total_actions += 1
                    parcel_onboard.remove(lid)
                    expected_parc_drop.discard(lid)

                else:
                    if idx != route_len - 1:
                        return False
                    if load != 0 or parcel_onboard:
                        return False
                    if not is_ended:
                        return False

                    total_actions += 1


            if state["parcels"] != parcel_onboard:
                return False
            if state["load"] != load:
                return False
            if self.route_costs[t_idx] != cumulated_cost:
                return False


            max_cost_check = max(max_cost_check, cumulated_cost)
            cost_sum += cumulated_cost


        if expected_pass_serve != self.remaining_pass_serve:
            return False
        if expected_parc_pick != self.remaining_parc_pick:
            return False
        if expected_parc_drop != self.remaining_parc_drop:
            return False
        if node_assignment_check != self.node_assignment:
            return False
        if self.max_cost != max_cost_check:
            return False
        if self.num_actions != total_actions:
            return False

        return True


    def is_pending(self) -> bool:
        return self.num_actions < self.problem.num_actions


    def is_identical(self, other: "PartialSolution") -> bool:
        if self is other:
            return True

        if self.problem is not other.problem or self.num_actions != other.num_actions:
            return False

        return sorted(tuple(r[:3]) for r in self.routes) == sorted(tuple(r[:3]) for r in other.routes)



    def copy(self):
        return PartialSolution(
            problem=self.problem,
            routes=[route.copy() for route in self.routes] if self.routes else None
        )

    def enumerate_action_nodes(self, route_idx: int) -> List[Tuple[int, int]]:
        prob = self.problem
        route = self.routes[route_idx]
        action_nodes = []
        for node in route:
            if prob.is_ppick(node):
                pid = self.problem.rev_ppick(node)
                drop_node = self.problem.pserve(pid)[1]
                action_nodes.append((node, drop_node))
            else:
                action_nodes.append((node, node))

        return action_nodes

    def decrease_cost(self, route_idx: int, dec: int):
        self.route_costs[route_idx] -= dec
        self.max_cost = max(self.route_costs)
        self.min_cost = min(self.min_cost, self.route_costs[route_idx])
        self.avg_cost = sum(self.route_costs) / self.problem.K


    def possible_expand(self, t_idx: int) -> List[Tuple[str, int, int]]:
        state = self.states[t_idx]
        if state["ended"]:
            return []

        prob = self.problem
        pos = state["pos"]
        expansions: List[Tuple[str, int, int]] = []

        for pid in self.remaining_pass_serve:
            pick, drop = prob.pserve(pid)
            inc = prob.D[pos][pick] + prob.D[pick][drop]
            expansions.append(("serveP", pid, inc))

        for lid in self.remaining_parc_pick:
            parcel_weight = prob.q[lid - 1]
            if state["load"] + parcel_weight <= prob.Q[t_idx]:
                inc = prob.D[pos][prob.lpick(lid)]
                expansions.append(("pickL", lid, inc))

        for lid in state["parcels"]:
            inc = prob.D[pos][prob.ldrop(lid)]
            expansions.append(("dropL", lid, inc))

        expansions.sort(key=lambda x: x[2])
        return expansions


    def check_expand(self, route_idx: int, kind: str, actid: int) -> bool:
        state = self.states[route_idx]
        prob = self.problem

        if state["ended"]:
            return False
        if kind == "serveP":
            return actid in self.remaining_pass_serve
        if kind == "pickL":
            return (
                actid in self.remaining_parc_pick
                and state["load"] + prob.q[actid - 1] <= prob.Q[route_idx]
            )
        if kind == "dropL":
            return actid in state["parcels"]

        raise ValueError(f"Unknown action kind: {kind}")


    def check_return(self, route_idx: int) -> bool:
        state = self.states[route_idx]
        return not (state["ended"] or state["parcels"])


    def apply_extend(self, route_idx: int, kind: str, actid: int, inc: int) -> None:
        route = self.routes[route_idx]
        state = self.states[route_idx]
        prob = self.problem


        if state["ended"]:
            raise ValueError(f"Cannot apply action on ended route {route_idx}.")

        if kind == "serveP":
            pick_node, drop_node = prob.pserve(actid)
            route.append(pick_node)
            route.append(drop_node)
            self.node_assignment[pick_node] = route_idx
            self.node_assignment[drop_node] = route_idx


            self.remaining_pass_serve.discard(actid)
            state["pos"] = drop_node
            state["actions"] += 1


            self.route_costs[route_idx] += inc
            self.max_cost = max(self.max_cost, self.route_costs[route_idx])
            self.min_cost = min(self.route_costs)
            self.avg_cost = sum(self.route_costs) / self.problem.K
            self.num_actions += 1

            return

        elif kind == "pickL":
            parc_qty = prob.q[actid - 1]
            if state["load"] + parc_qty > prob.Q[route_idx]:
                raise ValueError(f"Taxi {route_idx} capacity exceeded for parcel {actid}.")

            node = prob.lpick(actid)
            state["load"] += parc_qty
            state["parcels"].add(actid)
            self.remaining_parc_pick.discard(actid)
            self.remaining_parc_drop.add(actid)

        elif kind == "dropL":
            parc_qty = prob.q[actid - 1]
            if state["load"] - parc_qty < 0:
                raise ValueError(
                    f"Taxi {route_idx} load cannot be negative after dropping parcel {actid}."
                )

            node = prob.ldrop(actid)
            state["load"] -= parc_qty
            state["parcels"].discard(actid)
            self.remaining_parc_drop.discard(actid)

        else:
            raise ValueError(f"Unknown action kind: {kind}")


        state["pos"] = node
        state["actions"] += 1
        route.append(node)
        self.node_assignment[node] = route_idx


        self.route_costs[route_idx] += inc
        self.max_cost = max(self.max_cost, self.route_costs[route_idx])
        self.min_cost = min(self.route_costs)
        self.avg_cost = sum(self.route_costs) / self.problem.K
        self.num_actions += 1


    def apply_return(self, t_idx: int) -> None:
        route = self.routes[t_idx]
        state = self.states[t_idx]


        if state["ended"]:
            return
        if state["parcels"]:
            raise ValueError(f"Taxi {t_idx} must drop all loads before returning to depot.")

        # Always append 0 and add travel cost (even if already at depot, cost is 0)
        inc = self.problem.D[state["pos"]][0]
        route.append(0)
        state["pos"] = 0
        state["actions"] += 1
        state["ended"] = True


        self.route_costs[t_idx] += inc
        self.max_cost = max(self.max_cost, self.route_costs[t_idx])
        self.min_cost = min(self.route_costs)
        self.avg_cost = sum(self.route_costs) / self.problem.K
        self.num_actions += 1


    def reverse_action(self, t_idx: int) -> None:
        route = self.routes[t_idx]
        state = self.states[t_idx]
        if len(route) <= 1:
            raise ValueError(f"No actions to reverse for taxi {t_idx}.")

        prob = self.problem
        last_node = route[-1]

        if prob.is_pdrop(last_node):

            drop_node = route.pop()
            pick_node = route.pop()

            pid = prob.rev_pdrop(drop_node)
            if prob.rev_ppick(pick_node) != pid:
                raise ValueError(
                        "Inconsistent route state: "
                        "pdrop not preceded by corresponding ppick."
                    )


            prev_node = route[-1]
            dec = prob.D[prev_node][pick_node] + prob.D[pick_node][drop_node]


            state["pos"] = prev_node
            state["actions"] -= 1
            state["ended"] = False


            self.remaining_pass_serve.add(pid)
            self.node_assignment[drop_node] = -1
            self.node_assignment[pick_node] = -1


            self.route_costs[t_idx] -= dec
            self.max_cost = max(self.route_costs)
            self.min_cost = min(self.route_costs)
            self.avg_cost = sum(self.route_costs) / self.problem.K
            self.num_actions -= 1

            return


        last_node = route.pop()
        prev_node = route[-1]
        dec = prob.D[prev_node][last_node]


        state["pos"] = prev_node
        state["actions"] -= 1
        state["ended"] = False


        if prob.is_lpick(last_node):
            lid = prob.rev_lpick(last_node)
            state["load"] -= prob.q[lid - 1]
            state["parcels"].discard(lid)

            self.remaining_parc_pick.add(lid)
            self.remaining_parc_drop.discard(lid)

        elif prob.is_ldrop(last_node):
            lid = prob.rev_ldrop(last_node)
            state["load"] += prob.q[lid - 1]
            state["parcels"].add(lid)

            self.remaining_parc_pick.discard(lid)
            self.remaining_parc_drop.add(lid)

        elif last_node == 0:
            pass

        else:

            raise ValueError(f"Unexpected node type to reverse: {last_node}")


        self.route_costs[t_idx] -= dec
        self.max_cost = max(self.route_costs)
        self.min_cost = min(self.route_costs)
        self.avg_cost = sum(self.route_costs) / self.problem.K
        self.node_assignment[last_node] = -1
        self.num_actions -= 1


    def is_completed(self, verbose=False) -> bool:
        if self.num_actions != self.problem.num_actions:
            return False

        if not all(state["ended"] for state in self.states):
            return False

        return True


    def to_solution(self) -> Optional[Solution]:
        if not self.is_completed(verbose=True):
            return None

        if not self.is_valid(verbose=True):
            return None

        solution = Solution(
            problem=self.problem,
            routes=self.routes,
            route_costs=self.route_costs
        )

        return solution


    @staticmethod
    def from_solution(sol: Solution) -> "PartialSolution":
        routes_copy = [route.copy() for route in sol.routes]
        return PartialSolution(problem=sol.problem, routes=routes_copy)


def weighted(kind: str, inc: Union[int, float], pweight: float = 0.7) -> float:
    if kind == "serveP":
        return pweight * inc
    return inc + 1


def action_weight(action: Action, pweight: float = 0.7) -> float:
    return weighted(action[1], action[3], pweight)


def softmax_weighter(incs: Sequence[Union[int, float]], t: float) -> List[float]:

    min_inc, max_inc = min(incs), max(incs)
    inc_range = max_inc - min_inc

    if inc_range < 1e-6:
        return [1.0] * len(incs)


    weights: List[float] = []
    for inc in incs:
        normalized = (inc - min_inc) / inc_range
        weights.append((1.0 - normalized + 0.1) ** (1.0 / t))

    return weights




def balanced_scorer(
        partial: Union[PartialSolution, Solution],
        sample_size: int = 8,
        w_std: float = 0.15,
        seed: Optional[int] = None,
    ) -> float:

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

    taxi, kind, node_idx, _inc = action

    if kind == "return":
        return partial.check_return(taxi)

    return partial.check_expand(taxi, kind, node_idx)


def apply_general_action(partial: PartialSolution, action: Action) -> None:

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

    if width is None:
        width = 10**9

    problem = partial.problem


    active_taxis = [
        idx for idx, state in enumerate(partial.states)
        if not state["ended"]
    ]
    if not active_taxis:
        return []
    taxi_order = sorted(active_taxis, key=lambda idx: partial.route_costs[idx])
    num_taxis = len(taxi_order)


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

    def taxi_limit(aggressive: bool) -> int:

        if not aggressive:
            return num_taxis

        return min(
            2 if problem.K >= 25
            else 3 if problem.K >= 12
            else 4 if problem.K >= 6
            else 5,
            num_taxis
        )

    def action_per_taxi_limit(aggressive: bool) -> int:
        if not aggressive:
            return 10**9

        return min(
            2 if problem.num_nodes >= 500
            else 4 if problem.num_nodes >= 200
            else 6 if problem.num_nodes >= 100
            else 8 if problem.num_nodes >= 50
            else 12 if problem.num_nodes >= 25
            else 16,
            width,
        )

    def enumerate_pass(aggressive: bool) -> List[Action]:
        expansions: List[Action] = []
        taxi_used = taxi_order
        if aggressive:
            taxi_used = taxi_order[:taxi_limit(aggressive=True)]

        for t_idx in taxi_used:

            assigned_actions = partial.possible_expand(t_idx)
            if aggressive:
                assigned_actions = sorted(
                    assigned_actions,
                    key=lambda item: item[2]
                )[:action_per_taxi_limit(aggressive=True)]


            general_actions: List[Action] = [
                (t_idx, kind, node_idx, inc)
                for kind, node_idx, inc in assigned_actions
            ]


            expansions.extend(general_actions)
            expansions.sort(key=lambda item: item[3])
            expansions = expansions[:width]

        return expansions


    expansions = enumerate_pass(aggressive=True)
    if not expansions:
        expansions = enumerate_pass(aggressive=False)


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

    final_actions = [action for _, action in prioritized + secondary][:width]



    if not final_actions:

        if partial.num_actions < problem.num_expansions:

            raise RuntimeError("Premature routes not covering all nodes.")

        return_actions: List[Action] = []
        for t_idx in taxi_order:
            state = partial.states[t_idx]
            if partial.check_return(t_idx):
                inc_back = problem.D[state["pos"]][0]
                return_actions.append((t_idx, "return", 0, inc_back))

        return return_actions[:width]

    return final_actions



def sample_from_weight(rng: random.Random, weights: Sequence[Union[int, float]]) -> int:
    total_weight = sum(weights)

    if total_weight < 1e-10:
        res_idx = rng.randrange(len(weights))
    else:
        rand_val = rng.random() * total_weight
        cumsum = 0.0
        res_idx = 0
        for i, weight in enumerate(weights):
            cumsum += weight
            if rand_val <= cumsum:
                res_idx = i
                break

    return res_idx


def repair_one_route(
        partial: PartialSolution,
        route_idx: int,
        steps: int,
        T: float = 1.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[PartialSolution, int]:

    rng = random.Random(seed)
    added_actions = 0
    for _ in range(steps):
        state = partial.states[route_idx]
        if state["ended"]:
            break

        actions = partial.possible_expand(route_idx)
        if not actions:
            partial.apply_return(route_idx)
            added_actions += 1
            break

        incs = [weighted(kind, inc) for kind, _, inc in actions]
        weights = softmax_weighter(incs, T)
        selected_idx = sample_from_weight(rng, weights)

        kind, node_idx, inc = actions[selected_idx]
        partial.apply_extend(route_idx, kind, node_idx, inc)
        added_actions += 1

    return partial, added_actions




def repair_operator(
        partial: PartialSolution,
        repair_proba: Optional[float] = None,
        steps: Optional[int] = None,
        T: float = 1.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[PartialSolution, List[bool], int]:
    rng = random.Random(seed)
    if repair_proba is None:
        repair_proba = 1.0
    if steps is None:
        steps = 10**9

    routes = list(range(partial.problem.K))
    num_routes = partial.problem.K
    approx_repair_count = round(repair_proba * num_routes + 0.5)
    repair_count = min(num_routes, max(1, approx_repair_count))

    selected_routes = rng.sample(routes, repair_count)
    total_added_actions = 0
    modified = [False] * num_routes

    for r_idx in selected_routes:
        partial, added_actions = repair_one_route(
            partial=partial,
            route_idx=r_idx,
            steps=steps,
            T=T,
            seed=rng.randint(0, 1_000_000),
            verbose=verbose
        )

        total_added_actions += added_actions
        modified[r_idx] = True


    return partial, modified, total_added_actions



def destroy_one_route(
        problem: ShareARideProblem,
        route: List[int],
        route_idx: int,
        steps: int = 10,
        verbose: bool = False
    ) -> Tuple[List[int], int]:

    res_route = route[:]
    actions_removed = 0

    while actions_removed < steps and len(res_route) > 1:
        nodeid = res_route.pop()
        if problem.is_pdrop(nodeid):
            pid = problem.rev_pdrop(nodeid)
            pickid = problem.pserve(pid)[0]
            if res_route.pop() != pickid:
                raise RuntimeError(
                    "The destroyed route is likely invalid beforehand."
                )
        elif problem.is_lpick(nodeid) or problem.is_ldrop(nodeid) or nodeid == 0:
            pass
        else:
            raise RuntimeError(
                "The destroyed route is likely invalid beforehand."
            )

        actions_removed += 1

    return res_route, actions_removed




def destroy_operator(
        sol: Union[PartialSolution, Solution],
        destroy_proba: float,
        destroy_steps: int,
        seed: Optional[int] = None,
        t: float = 1.0,
        verbose: bool = False
    ) -> Tuple[PartialSolution, List[bool], int]:

    rng = random.Random(seed)
    K = sol.problem.K
    routes = [route[:] for route in sol.routes]
    costs = sol.route_costs

    approx_destroyed_count = round(destroy_proba * K + 0.5)
    destroyed_count = min(K, max(1, approx_destroyed_count))
    weights = softmax_weighter(costs, t=t)


    selected_ids: List[int] = []
    available_ids = list(range(K))
    available_weights = weights[:]

    for _ in range(destroyed_count):
        if not available_ids:
            break

        selected_idx = sample_from_weight(rng, available_weights)
        selected_ids.append(available_ids[selected_idx])

        available_ids.pop(selected_idx)
        available_weights.pop(selected_idx)


    flags = [False] * K
    actions_removed = 0
    for idx in selected_ids:
        route = routes[idx]

        if len(route) <= 2:
            continue

        reduced_route, num_removed = destroy_one_route(
            sol.problem, route, idx, steps=destroy_steps, verbose=verbose
        )

        if num_removed > 0:
            routes[idx] = reduced_route
            flags[idx] = True
            actions_removed += num_removed
    new_partial = PartialSolution(problem=sol.problem, routes=routes)
    return new_partial, flags, actions_removed




class MinMaxPfsumArray:

    class Block:

        def __init__(self, data):
            self.arr = data[:]
            self.size = len(self.arr)
            self.recalc()


        def recalc(self):
            self.size = len(self.arr)
            self.sum = sum(self.arr)
            csum = 0
            mn = 10**18
            mx = -10**18
            for x in self.arr:
                csum += x
                mn = min(mn, csum)
                mx = max(mx, csum)
            self.min_pref = mn
            self.max_pref = mx


        def insert(self, idx, entry):
            self.arr.insert(idx, entry)
            self.recalc()


        def erase(self, idx):
            del self.arr[idx]
            self.recalc()


    def __init__(self, data: List[int]):
        self.block_arr = []
        self.num_data = 0
        self.block_prefix: List[int] = []
        self.build(data)


    def build(self, data: List[int]):
        self.block_arr.clear()

        self.num_data: int = len(data)
        self.block_size = max(0, int(math.sqrt(self.num_data))) + 2

        for i in range(0, self.num_data, self.block_size):
            self.block_arr.append(self.Block(data[i:i + self.block_size]))
        self.num_block = len(self.block_arr)

        self._rebuild_indexing()


    def _rebuild_indexing(self):
        self.block_prefix = []
        cumid = 0
        for b in self.block_arr:
            self.block_prefix.append(cumid)
            cumid += b.size

        self.num_data = cumid


    def _find_block(self, idx: int) -> Tuple[int, int]:
        if idx > self.num_data:
            idx = self.num_data
        bid = bisect.bisect_right(self.block_prefix, idx) - 1
        iid = idx - self.block_prefix[bid]

        return bid, iid


    def insert(self, idx, val):
        if idx == self.num_data:
            if not self.block_arr:
                self.block_arr.append(self.Block([val]))
            else:
                last = self.block_arr[-1]

                if last.size >= 2 * self.block_size:
                    self.block_arr.append(self.Block([val]))
                else:
                    last.insert(last.size, val)


            self.num_data += 1
            self._rebuild_indexing()
            return

        bid, iid = self._find_block(idx)
        blk = self.block_arr[bid]
        blk.insert(iid, val)

        if blk.size > 2 * self.block_size:
            arr = blk.arr
            mid = len(arr) // 2
            left = self.Block(arr[:mid])
            right = self.Block(arr[mid:])
            self.block_arr[bid:bid + 1] = [left, right]

        self.num_data += 1
        self._rebuild_indexing()


    def delete(self, idx):
        bid, iid = self._find_block(idx)
        self.block_arr[bid].erase(iid)

        if self.block_arr[bid].size == 0:
            del self.block_arr[bid]
        else:
            min_size = max(1, self.block_size // 2)
            if self.block_arr[bid].size < min_size:
                if bid + 1 < len(self.block_arr):
                    nxt = self.block_arr[bid + 1]
                    if self.block_arr[bid].size + nxt.size <= 2 * self.block_size:
                        merged = self.block_arr[bid].arr + nxt.arr
                        self.block_arr[bid:bid + 2] = [self.Block(merged)]

                elif bid - 1 >= 0:
                    prv = self.block_arr[bid - 1]
                    if prv.size + self.block_arr[bid].size <= 2 * self.block_size:
                        merged = prv.arr + self.block_arr[bid].arr
                        self.block_arr[bid - 1:bid + 1] = [self.Block(merged)]

        self.num_data -= 1
        self._rebuild_indexing()


    def query_min_prefix(self, l, r):
        ans = 10**18
        pos = 0
        prefix = 0
        ans = 10**18
        pos = 0
        for b in self.block_arr:
            blen = b.size
            if pos + blen <= l:
                prefix += b.sum
                pos += blen
                continue
            if pos >= r:
                break
            start = max(0, l - pos)
            end   = min(blen, r - pos)

            if start > 0:
                for i in range(start):
                    prefix += b.arr[i]
            if start == 0 and end == blen:
                ans = min(ans, prefix + b.min_pref)
                prefix += b.sum
            else:
                for i in range(start, end):
                    prefix += b.arr[i]
                    ans = min(ans, prefix)

            pos += blen

        return ans


    def query_max_prefix(self, l, r):
        ans = float('-inf')
        pos = 0
        prefix = 0
        ans = float('-inf')
        pos = 0
        for b in self.block_arr:
            blen = b.size
            if pos + blen <= l:
                prefix += b.sum
                pos += blen
                continue
            if pos >= r:
                break

            start = max(0, l - pos)
            end   = min(blen, r - pos)

            if start > 0:
                for i in range(start):
                    prefix += b.arr[i]

            if start == 0 and end == blen:
                ans = max(ans, prefix + b.max_pref)
                prefix += b.sum
            else:
                for i in range(start, end):
                    prefix += b.arr[i]
                    ans = max(ans, prefix)

            pos += blen

        return ans


    def get_data_point(self, idx) -> int:
        if idx < 0 or idx >= self.num_data:
            raise IndexError("Index out of bounds")

        bid, iid = self._find_block(idx)

        return self.block_arr[bid].arr[iid]


    def get_data_segment(self, l: int, r: int) -> List[int]:
        if l < 0 or r < 0 or l > r or r > self.num_data:
            raise IndexError("Invalid segment range")

        result: List[int] = []
        pos = 0
        for b in self.block_arr:
            blen = b.size
            if pos >= r:
                break
            if pos + blen <= l:
                pos += blen
                continue

            start = max(0, l - pos)
            end = min(blen, r - pos)
            if end > start:
                result.extend(b.arr[start:end])

            pos += blen

        return result


    def get_data(self) -> List[int]:
        return self.get_data_segment(0, self.num_data)



def cost_decrement_relocate(
        partial: PartialSolution,
        rfidx: int,
        rtidx: int,
        pfidx: int,
        qfidx: int,
        ptidx: int,
        qtidx: int
    ) -> CostChange:
    D = partial.problem.D
    cur_cost = partial.max_cost

    routef = partial.routes[rfidx]
    routet = partial.routes[rtidx]
    rfcost = partial.route_costs[rfidx]
    rtcost = partial.route_costs[rtidx]
    pf = routef[pfidx]
    qf = routef[qfidx]

    pprev = routef[pfidx - 1]
    pnext = routef[pfidx + 1]
    qprev = routef[qfidx - 1]
    qnext = routef[qfidx + 1]
    cost_delta_f = 0

    if pfidx + 1 == qfidx:
        cost_delta_f -= D[pprev][pf] + D[pf][qf] + D[qf][qnext]
        cost_delta_f += D[pprev][qnext]
    else:
        cost_delta_f -= D[pprev][pf] + D[pf][pnext] + D[qprev][qf] + D[qf][qnext]
        cost_delta_f += D[pprev][pnext] + D[qprev][qnext]

    rfcost_after = rfcost + cost_delta_f
    pprev_to = routet[ptidx - 1]
    pnext_to = routet[ptidx]
    qprev_to = routet[qtidx - 2]
    qnext_to = routet[qtidx - 1]
    cost_delta_t = 0

    if qtidx == ptidx + 1:
        cost_delta_t -= D[pprev_to][qnext_to]
        cost_delta_t += D[pprev_to][pf] + D[pf][qf] + D[qf][qnext_to]
    else:
        cost_delta_t -= D[pprev_to][pnext_to] + D[qprev_to][qnext_to]
        cost_delta_t += D[pprev_to][pf] + D[pf][pnext_to] + D[qprev_to][qf] + D[qf][qnext_to]

    rtcost_after = rtcost + cost_delta_t



    remain_costs = [
        partial.route_costs[i]
        for i in range(partial.problem.K) if i != rfidx and i != rtidx
    ]
    next_cost = max(rfcost_after, rtcost_after, *remain_costs)

    return rfcost_after, rtcost_after, cur_cost - next_cost




def relocate_from_to(
        partial: PartialSolution,
        route_from_idx: int,
        route_to_idx: int,
        steps: int,
        mode: str,
        uplift: int = 1,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[PartialSolution, List[bool], int]:

    rng = random.Random(seed)


    prob = partial.problem
    current_par = partial.copy()


    route_from = current_par.routes[route_from_idx]
    route_to = current_par.routes[route_to_idx]
    n_from = len(route_from)
    n_to = len(route_to)


    if n_from < 5:
        return partial, [False] * prob.K, 0



    def build_ds(route: List[int], n: int):
        load_deltas = [0] * n
        for i, node in enumerate(route):
            if prob.is_lpick(node):
                jid = prob.rev_lpick(node)
                dq = prob.q[jid - 1]
            elif prob.is_ldrop(node):
                jid = prob.rev_ldrop(node)
                dq = -prob.q[jid - 1]
            else:
                dq = 0

            load_deltas[i] = dq

        load_delta_manager = MinMaxPfsumArray(load_deltas)

        return load_delta_manager


    load_delta_from_manager = build_ds(route_from, n_from)
    load_delta_to_manager = build_ds(route_to, n_to)


    cap_from = prob.Q[route_from_idx]
    cap_to = prob.Q[route_to_idx]



    def check_consecutive(req: RelocateRequest) -> bool:
        pfidx, qfidx, ptidx, qtidx, kind = req

        if kind == "serveL":
            return True


        return qtidx == ptidx + 1


    def check_load(req: RelocateRequest) -> bool:
        pfidx, qfidx, ptidx, qtidx, kind = req
        if kind == "serveP":
            return True


        pf = route_from[pfidx]
        if prob.is_lpick(pf):
            jid = prob.rev_lpick(pf)
            load_delta = prob.q[jid - 1]
        else:
            load_delta = 0


        load_min_fr = load_delta_from_manager.query_min_prefix(pfidx, qfidx)
        load_max_fr = load_delta_from_manager.query_max_prefix(pfidx, qfidx)
        if load_min_fr - load_delta < 0:
            return False
        if load_max_fr - load_delta > cap_from:
            return False


        load_min_to = load_delta_to_manager.query_min_prefix(ptidx - 1, qtidx - 1)
        load_max_to = load_delta_to_manager.query_max_prefix(ptidx - 1, qtidx - 1)
        if load_min_to + load_delta < 0:
            return False
        if load_max_to + load_delta > cap_to:
            return False

        return True


    def check_relocate(req: RelocateRequest) -> Optional[CostChange]:

        if not check_consecutive(req):
            return None
        if not check_load(req):
            return None


        cost_change = cost_decrement_relocate(
            current_par, route_from_idx, route_to_idx,
            req[0], req[1], req[2], req[3],
        )
        return cost_change


    def find_candidates() -> Iterator[Tuple[RelocateRequest, CostChange]]:
        pos_from = {node: i for i, node in enumerate(route_from)}


        delete_pairs = []
        for pfidx, pickup_node in enumerate(route_from[1:], start=1):
            if prob.is_ppick(pickup_node):
                qfidx = pfidx + 1
                delete_pairs.append((pfidx, qfidx, "serveP"))

            elif prob.is_lpick(pickup_node):
                jid = prob.rev_lpick(pickup_node)
                drop_node = prob.ldrop(jid)
                qfidx = pos_from.get(drop_node)

                if qfidx is not None and qfidx > pfidx:
                    delete_pairs.append((pfidx, qfidx, "serveL"))


        insert_pairs_pserve = [
            (ptidx, ptidx + 1)
            for ptidx in range(1, n_to)
            if not prob.is_ppick(route_to[ptidx - 1])
        ]
        insert_pairs_lserve = [
            (ptidx, qtidx)
            for ptidx in range(1, n_to)
            if not prob.is_ppick(route_to[ptidx - 1])
            for qtidx in range(ptidx + 1, n_to + 1)
            if not prob.is_ppick(route_to[qtidx - 2])
        ]


        for (pfidx, qfidx, kind) in delete_pairs:
            insert_pairs = insert_pairs_pserve if kind == "serveP" else insert_pairs_lserve
            for (ptidx, qtidx) in insert_pairs:
                request = (pfidx, qfidx, ptidx, qtidx, kind)
                costchange = check_relocate(request)


                if costchange is None:
                    continue


                _, _, dec = costchange
                if dec < uplift:
                    continue

                if mode == 'first':
                    yield (request, costchange)
                    return
                else:
                    yield (request, costchange)


    def select_candidate() -> Optional[Tuple[RelocateRequest, CostChange]]:
        cand_list = list(find_candidates())
        if not cand_list:
            return None
        if mode == 'stochastic':
            return rng.choice(cand_list)
        elif mode == 'best':

            return max(cand_list, key=lambda x: x[1][2])
        else:
            return cand_list[0]



    def update_partial_solution(action: RelocateAction):
        nonlocal route_from, route_to, current_par


        (p_from, q_from, p_to, q_to, _), (new_cost_from, new_cost_to, dec) = action


        pf = route_from[p_from]
        qf = route_from[q_from]



        del route_from[q_from]
        del route_from[p_from]


        route_to.insert(p_to, pf)
        route_to.insert(q_to, qf)


        current_par.routes[route_from_idx] = route_from
        current_par.routes[route_to_idx] = route_to


        current_par.route_costs[route_from_idx] = new_cost_from
        current_par.route_costs[route_to_idx] = new_cost_to
        current_par.max_cost -= dec


        current_par.node_assignment[pf] = route_to_idx
        current_par.node_assignment[qf] = route_to_idx


    def update_precalc(action: RelocateAction):
        nonlocal load_delta_from_manager, load_delta_to_manager, route_from, route_to


        (pfidx, qfidx, ptidx, qtidx, _), _costchange = action
        pf = route_from[pfidx]
        qf = route_from[qfidx]


        def node_load_delta(nodeid: int) -> int:
            if prob.is_lpick(nodeid):
                jid = prob.rev_lpick(nodeid)
                return prob.q[jid - 1]
            elif prob.is_ldrop(nodeid):
                jid = prob.rev_ldrop(nodeid)
                return -prob.q[jid - 1]
            else:
                return 0



        load_delta_from_manager.delete(qfidx)
        load_delta_from_manager.delete(pfidx)


        load_delta_to_manager.insert(ptidx, node_load_delta(pf))
        load_delta_to_manager.insert(qtidx, node_load_delta(qf))


    def relocate_to_convergence() -> Tuple[List[bool], int]:
        nonlocal n_from, n_to, route_from, route_to

        reloc_done = 0
        modified_routes = [False] * prob.K
        while reloc_done < steps:
            cand = select_candidate()
            if cand is None:
                break


            update_precalc(cand)
            update_partial_solution(cand)


            reloc_done += 1
            modified_routes[route_from_idx] = True
            modified_routes[route_to_idx] = True


            n_from -= 2
            n_to += 2
            if n_from < 5:
                break

        return modified_routes, reloc_done

    modified_pair, reloc_done = relocate_to_convergence()

    return current_par, modified_pair, reloc_done




def relocate_operator(
        partial: PartialSolution,
        steps: Optional[int] = None,
        mode: str = 'first',
        uplift: int = 1,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[PartialSolution, List[bool], int]:
    k_vehicles = partial.problem.K
    if k_vehicles < 2:
        return partial.copy(), [False] * k_vehicles, 0

    if steps == None:
        steps = 10**9

    rng = random.Random(seed)

    current_par: PartialSolution = partial.copy()
    modified_total: List[bool] = [False] * k_vehicles
    total_moves = 0

    while total_moves < steps:
        taxi_cost: List[Tuple[int, int]] = list(enumerate(current_par.route_costs))
        donor_index = max(taxi_cost, key=lambda x: x[1])[0]
        receiver_indices = [
            idx for idx, _ in sorted(taxi_cost, key=lambda x: x[1])
        ]

        if len(current_par.routes[donor_index]) < 5:
            break

        improved = False
        for r_idx in receiver_indices:
            if r_idx == donor_index:
                continue

            if len(current_par.routes[r_idx]) < 2:
                continue

            remain = steps - total_moves
            new_partial, modified_pair, moves_made = relocate_from_to(
                current_par,
                route_from_idx=donor_index,
                route_to_idx=r_idx,
                steps=remain,
                mode=mode,
                uplift=uplift,
                seed=rng.randint(10, 10**9),
                verbose=verbose,
            )

            if moves_made > 0:
                current_par = new_partial
                total_moves += moves_made
                for i in range(k_vehicles):
                    if modified_pair[i]:
                        modified_total[i] = True

                improved = True
                break

        if not improved:
            break

    return current_par, modified_total, total_moves




def greedy_solver(
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        num_actions: int = 7,
        t_actions: float = 0.01,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    start = time.time()
    rng = random.Random(seed)

    if partial is None:
        partial = PartialSolution(problem=problem)

    iterations = 0
    pre_depth = partial.num_actions
    while partial.is_pending():
        iterations += 1

        actions = enumerate_actions_greedily(partial, num_actions)
        expansions = [a for a in actions if a[1] != "return"]
        if not expansions:
            expansions = actions
        if not expansions:
            return None, {
                "iterations": iterations,
                "time": time.time() - start,
                "actions_done": partial.num_actions - pre_depth,
                "status": "error",
            }

        incs = [a[3] for a in expansions]
        weights = softmax_weighter(incs, t_actions)
        act_idx = sample_from_weight(rng, weights)
        action = expansions[act_idx]

        apply_general_action(partial, action)


    sol = partial.to_solution()
    stats = {
        "iterations": iterations,
        "time": time.time() - start,
        "actions_done": partial.num_actions - pre_depth,
        "status": "done",
    }


    return sol, stats




def iterative_greedy_solver(
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        iterations: int = 10000,
        num_actions: int = 7,
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

    start = time.time()
    deadline = start + time_limit
    rng = random.Random(seed)


    assert 1e-5 < destroy_proba < 1 - 1e-5
    assert 1e-5 < rebuild_proba < 1 - 1e-5
    assert 1 <= rebuild_steps <= destroy_steps


    if partial is None:
        partial = PartialSolution(problem=problem, routes=[])



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

        operators_seed = None if seed is None else 2 * seed + 98 * it

        destroyed_partial, destroyed_flags, removed = destroy_operator(
            best_sol,
            destroy_proba,
            destroy_steps,
            seed=operators_seed,
            t=destroy_t
        )
        nodes_destroyed += removed
        actions += removed

        rebuilt_partial = destroyed_partial
        for r_idx, was_destroyed in enumerate(destroyed_flags):
            if not was_destroyed:
                continue
            if rng.random() > rebuild_proba:
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


        new_sol, new_info = greedy_solver(
            problem,
            partial=rebuilt_partial,
            num_actions=1,
            verbose=False
        )
        all_cost += new_sol.max_cost if new_sol else 0
        actions += new_info["actions_done"]
        iterations_done += 1 if new_sol else 0


        if new_sol and new_sol.max_cost < best_cost:
            best_sol = new_sol
            best_cost = new_sol.max_cost
            improvements += 1

    best_partial = PartialSolution.from_solution(best_sol)
    refined_partial, _, _ = relocate_operator(
        best_partial,
        mode='first',
        seed=None if seed is None else 4 * seed + 123
    )
    best_sol = refined_partial.to_solution();  assert best_sol
    best_cost = best_sol.max_cost


    elapsed = time.time() - start
    stats = {
        "iterations": iterations_done,
        "actions_done": actions,
        "improvements": improvements,
        "actions_destroyed": nodes_destroyed,
        "actions_rebuilt": nodes_rebuilt,
        "average_cost": all_cost / (iterations_done + 1),
        "time": elapsed,
        "status": status,
    }

    return best_sol, stats




def read_instance() -> ShareARideProblem:
    N, M, K = map(int, sys.stdin.readline().strip().split())
    q = list(map(int, sys.stdin.readline().split()))
    Q = list(map(int, sys.stdin.readline().split()))
    D = [[0] * (2 * N + 2 * M + 1) for _ in range(2 * N + 2 * M + 1)]
    for r in range(2 * N + 2 * M + 1):
        line = sys.stdin.readline().strip()
        D[r] = list(map(int, line.split()))

    return ShareARideProblem(N, M, K, q, Q, D)




def main(verbose: bool = False):
    problem: ShareARideProblem = read_instance()

    n = problem.num_nodes

    n = problem.num_nodes
    if n <= 100:
        cutoff_depth, cutoff_depth_inc, cutoff_iter, width, uct_c = 9, 3, 9000, 6, 0.6
    elif n <= 250:
        cutoff_depth, cutoff_depth_inc, cutoff_iter, width, uct_c = 8, 2, 4000, 4, 0.5
    elif n <= 500:
        cutoff_depth, cutoff_depth_inc, cutoff_iter, width, uct_c = 7, 1, 1400, 3, 0.3
    elif n <= 1000:
        cutoff_depth, cutoff_depth_inc, cutoff_iter, width, uct_c = 6, 0, 600, 2, 0.1
    else:
        cutoff_depth, cutoff_depth_inc, cutoff_iter, width, uct_c = 5, 0, 250, 2, 0.05

    best_solution, stats = iterative_greedy_solver(
        problem=problem,
        iterations=25000,
        time_limit=250.0,
        verbose=verbose,
        seed=42
    )
    assert best_solution
    best_solution.stdin_print(verbose=verbose)




if __name__ == "__main__":
    main(verbose=False)
