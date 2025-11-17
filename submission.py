
import sys
import time
import random

from typing import List, Optional, Tuple, Dict, Any


class ShareARideProblem:
    def __init__(
            self, N: int, M: int, K: int,
            parcel_qty: List[int], vehicle_caps: List[int],
            dist: List[List[int]], coords: Optional[List[Tuple[int, int]]] = None
        ):

        
        self.N = N
        self.M = M
        self.K = K
        self.q = list(parcel_qty)
        self.Q = list(vehicle_caps)
        self.D = [row[:] for row in dist]
        self.num_nodes = 2*N + 2*M + 1
        self.num_requests = N + M

        
        self.ppick = lambda i: i
        self.pdrop = lambda i: N + M + i
        self.parc_pick = lambda j: N + j
        self.parc_drop = lambda j: 2*N + M + j

        self.rev_ppick = lambda i: i
        self.rev_pdrop = lambda n: n - (N + M)
        self.rev_parc_pick = lambda n: n - N
        self.rev_parc_drop = lambda n: n - (2 * N + M)

        
        self.is_ppick = lambda x: 1 <= x <= N
        self.is_pdrop = lambda x: N + M + 1 <= x <= 2 * N + M
        self.is_parc_pick = lambda x: N + 1 <= x <= N + M
        self.is_parc_drop = lambda x: 2 * N + M + 1 <= x <= 2 * (N + M)

        
        self.coords = coords


    def is_valid(self) -> bool:
        try:
            assert len(self.q) == self.M
            assert len(self.Q) == self.K
            assert len(self.D) == self.num_nodes
            assert all(len(row) == self.num_nodes for row in self.D)
            assert len(self.coords) == self.num_nodes \
                if self.coords is not None else True

            return True
        except:
            return False

    def copy(self):
        return ShareARideProblem(self.N, self.M, self.K,
            list(self.q), list(self.Q), [row[:] for row in self.D]
        )


    def stdin_print(self):
        print(self.N, self.M, self.K)
        print(*self.q)
        print(*self.Q)
        for row in self.D:
            print(*row)





def route_cost_from_sequence(
        seq: List[int], 
        D: List[List[int]], 
        verbose: bool = False
    ) -> int:
    

    assert seq and seq[0] == 0

    prev, total_cost = 0, 0
    for node in seq[1:]:
        if verbose: 
            print(D[prev][node], end=" ")

        total_cost += D[prev][node]
        prev = node
   
    if verbose:
        print()

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
        self.n_actions = 2 * (problem.N + problem.M) + problem.K
        self.max_cost = max(new_route_costs) if new_route_costs else 0


    def is_valid(self) -> bool:
        

        prob = self.problem
        N, M, K = prob.N, prob.M, prob.K
        if len(self.routes) != K:
            return False

        for route_idx, route in enumerate(self.routes):
            
            if not (route[0] == 0 and route[-1] == 0):
                return False

            
            passenger_onboard = set()
            parcel_onboard = set()
            parcel_load = 0

            visited_pass_pick = set()
            visited_parc_pick = set()

            for node in route[1:-1]:

                
                if prob.is_ppick(node):
                    idx = prob.rev_ppick(node)

                    
                    if idx in visited_pass_pick:
                        return False
                    if len(passenger_onboard) >= 1:
                        return False

                    
                    passenger_onboard.add(idx)
                    visited_pass_pick.add(idx)

                
                elif prob.is_pdrop(node):
                    idx = prob.rev_pdrop(node)

                    
                    if idx not in passenger_onboard:
                        return False

                    
                    passenger_onboard.remove(idx)

                
                elif prob.is_parc_pick(node):
                    jd = prob.rev_parc_pick(node)

                    
                    if jd in visited_parc_pick or jd in parcel_onboard:
                        return False
                    parcel_load += prob.q[jd - 1]
                    if parcel_load > prob.Q[route_idx]:
                        return False

                    
                    visited_parc_pick.add(jd)
                    parcel_onboard.add(jd)

                
                elif prob.is_parc_drop(node):
                    jd = prob.rev_parc_drop(node)

                    
                    if jd not in parcel_onboard:
                        return False

                    
                    assert parcel_load - prob.q[jd - 1] >= 0
                    parcel_load -= prob.q[jd - 1]
                    parcel_onboard.remove(jd)

            
            if passenger_onboard:
                return False
            if parcel_load != 0:
                return False

        return True


    def stdin_print(self, verbose: bool = False):
        
        assert len(self.routes) == len(self.route_costs)

        print(self.problem.K)
        for route, cost in zip(self.routes, self.route_costs):
            print(len(route))
            print(" ".join(map(str, route)))
            if verbose:
                print(f"// Route cost: {cost}")
                print("----------------")

        if verbose:
            print(f"//// Max route cost: {self.max_cost} ////")



class PartialSolution:
    

    def __init__(
            self,
            problem: ShareARideProblem,
            routes: List[List[int]],
        ):
        

        
        self.problem = problem
        self.routes = self._init_routes(routes)
        self.route_costs = self._init_costs(routes)

        
        self.max_cost = max(self.route_costs)
        self.avg_cost = sum(self.route_costs) / problem.K
        self.node_assignment = self._init_node_assignment()
        (   self.remaining_pass_pick, self.remaining_pass_drop, self.remaining_parc_pick,
            self.remaining_parc_drop, self.route_states
        ) = self._init_states()

        self.n_actions = sum(len(route) - 1 for route in self.routes)


    def _init_routes(self, routes):
        K = self.problem.K

        
        if not routes:
            return [[0] for _ in range(K)]
        if len(routes) != K:
            raise ValueError(f"Expected {K} routes, got {len(routes)}.")
        for route in routes:
            if not route:
                raise ValueError("One route cannot be null")
            if route[0] != 0:
                raise ValueError("Each route must start at depot 0.")

        return routes


    def _init_costs(self, routes):
        
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

        
        remaining_pass_pick = set(range(1, prob.N + 1))
        remaining_pass_drop = set()
        remaining_parc_pick = set(range(1, prob.M + 1))
        remaining_parc_drop = set()
        route_states = []

        for idx, route in enumerate(self.routes):
            onboard_passenger = 0
            onboard_parcels = set()
            current_load = 0

            for node in route[1:]:
                if prob.is_ppick(node):
                    pid = prob.rev_ppick(node)
                    remaining_pass_pick.discard(pid)
                    remaining_pass_drop.add(pid)
                    onboard_passenger = pid
                elif prob.is_pdrop(node):
                    pid = prob.rev_pdrop(node)
                    remaining_pass_drop.discard(pid)
                    if onboard_passenger == pid:
                        onboard_passenger = 0
                elif prob.is_parc_pick(node):
                    jid = prob.rev_parc_pick(node)
                    remaining_parc_pick.discard(jid)
                    remaining_parc_drop.add(jid)
                    onboard_parcels.add(jid)
                    current_load += prob.q[jid - 1]
                elif prob.is_parc_drop(node):
                    jid = prob.rev_parc_drop(node)
                    if jid in onboard_parcels:
                        onboard_parcels.remove(jid)
                        current_load -= prob.q[jid - 1]
                    remaining_parc_drop.discard(jid)

            pos = route[-1]
            ended = len(route) > 1 and route[-1] == 0
            state = {
                "route": route,
                "pos": pos,
                "cost": self.route_costs[idx],
                "load": current_load,
                "passenger": onboard_passenger,
                "parcels": onboard_parcels.copy(),
                "actions": len(route) - 1,
                "ended": ended
            }
            route_states.append(state)

        return (
            remaining_pass_pick,
            remaining_pass_drop,
            remaining_parc_pick,
            remaining_parc_drop,
            route_states
        )


    def is_valid(self) -> bool:
        
        prob = self.problem
        N, M, K = prob.N, prob.M, prob.K

        
        if not len(self.routes) == len(self.route_states) == len(self.route_costs) == K:
            return False
        if len(self.node_assignment) != len(prob.D):
            return False

        
        expected_pass_pick = set(range(1, N + 1))
        expected_pass_drop: set[int] = set()
        expected_parc_pick = set(range(1, M + 1))
        expected_parc_drop: set[int] = set()

        
        node_assignment_check = [-1] * len(prob.D)
        total_actions = 0
        max_cost_check = 0
        cost_sum = 0

        
        for t_idx in range(K):
            route = self.routes[t_idx]
            state = self.route_states[t_idx]

            if not route or route[0] != 0:
                return False
            if state["route"] != route:
                return False
            if state["pos"] != route[-1]:
                return False
            if state["actions"] != len(route) - 1:
                return False
            ended_now = len(route) > 1 and route[-1] == 0
            if state["ended"] != ended_now:
                return False

            passenger_onboard: set[int] = set()
            parcel_onboard: set[int] = set()
            load = 0
            prev = route[0]
            computed_cost = 0

            
            for node in route[1:]:
                if not 0 <= node < len(prob.D):
                    return False

                computed_cost += prob.D[prev][node]
                prev = node

                if node != 0:
                    assigned = node_assignment_check[node]
                    if assigned != -1 and assigned != t_idx:
                        return False
                    node_assignment_check[node] = t_idx

                if prob.is_ppick(node):
                    pid = prob.rev_ppick(node)
                    if pid in passenger_onboard or passenger_onboard:
                        return False
                    passenger_onboard.add(pid)
                    expected_pass_pick.discard(pid)
                    expected_pass_drop.add(pid)

                elif prob.is_pdrop(node):
                    pid = prob.rev_pdrop(node)
                    if pid not in passenger_onboard:
                        return False
                    passenger_onboard.remove(pid)
                    expected_pass_drop.discard(pid)

                elif prob.is_parc_pick(node):
                    jid = prob.rev_parc_pick(node)
                    if jid in parcel_onboard:
                        return False
                    load += prob.q[jid - 1]
                    if load > prob.Q[t_idx]:
                        return False
                    parcel_onboard.add(jid)
                    expected_parc_pick.discard(jid)
                    expected_parc_drop.add(jid)

                elif prob.is_parc_drop(node):
                    jid = prob.rev_parc_drop(node)
                    if jid not in parcel_onboard:
                        return False
                    load -= prob.q[jid - 1]
                    parcel_onboard.remove(jid)
                    expected_parc_drop.discard(jid)

            current_passenger = next(iter(passenger_onboard)) if passenger_onboard else 0
            if state["passenger"] != current_passenger:
                return False
            if state["parcels"] != parcel_onboard:
                return False
            if state["load"] != load:
                return False
            if state["cost"] != computed_cost or self.route_costs[t_idx] != computed_cost:
                return False

            
            total_actions += len(route) - 1
            max_cost_check = max(max_cost_check, computed_cost)
            cost_sum += computed_cost


        
        if expected_pass_pick != self.remaining_pass_pick:
            return False
        if expected_pass_drop != self.remaining_pass_drop:
            return False
        if expected_parc_pick != self.remaining_parc_pick:
            return False
        if expected_parc_drop != self.remaining_parc_drop:
            return False
        if node_assignment_check != self.node_assignment:
            return False
        if self.max_cost != max_cost_check:
            return False
        if self.n_actions != total_actions:
            return False

        return True


    def is_identical(self, other: "PartialSolution") -> bool:
        
        if self is other:
            return True

        
        if self.problem is not other.problem:
            return False
        if self.n_actions != other.n_actions:
            return False

        
        def _canonical_node_assignment(
                ps: PartialSolution
            ) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]]:
            per_route: List[Tuple[int, ...]] = []
            unassigned: List[int] = []
            buckets: Dict[int, List[int]] = {}

            for node_idx, assigned in enumerate(ps.node_assignment):
                if node_idx == 0:
                    continue
                if assigned == -1:
                    unassigned.append(node_idx)
                else:
                    buckets.setdefault(assigned, []).append(node_idx)

            for nodes in buckets.values():
                per_route.append(tuple(sorted(nodes)))

            per_route.sort()
            unassigned.sort()
            return tuple(per_route), tuple(unassigned)

        if _canonical_node_assignment(self) != _canonical_node_assignment(other):
            return False

        
        def _canonical_route_signature(
                ps: PartialSolution
            ) -> List[Tuple[int, int, int]]:
            summary: List[Tuple[int, int, int]] = []
            for idx, route in enumerate(ps.routes):
                first = route[1] if len(route) > 1 else -1
                second = route[2] if len(route) > 2 else -1
                summary.append((first, second, ps.route_costs[idx]))

            summary.sort()
            return summary

        if _canonical_route_signature(self) != _canonical_route_signature(other):
            return False

        return True


    def copy(self):
        
        return PartialSolution(
            problem=self.problem,
            routes=[route.copy() for route in self.routes]
        )


    def stdin_print(self, verbose: bool = False):
        
        assert len(self.routes) == len(self.route_costs)

        print(self.problem.K)
        for route, cost in zip(self.routes, self.route_costs):
            print(len(route))
            print(" ".join(map(str, route)))
            if verbose:
                print(f"// Route cost: {cost}")
                print("----------------")

        if verbose:
            print(f"//// Max route cost: {self.max_cost} ////")


    def possible_actions(self, t_idx: int) -> List[Tuple[str, int, int]]:
        

        state = self.route_states[t_idx]
        if state["ended"]:
            return []

        prob = self.problem
        pos = state["pos"]
        actions: List[Tuple[str, int, int]] = []

        if state["passenger"] == 0:
            for pid in list(self.remaining_pass_pick):
                inc = prob.D[pos][prob.ppick(pid)]
                actions.append(("pickP", pid, inc))
        else:
            pid = state["passenger"]
            inc = prob.D[pos][prob.pdrop(pid)]
            actions.append(("dropP", pid, inc))

        for jid in list(self.remaining_parc_pick):
            parcel_weight = prob.q[jid - 1]
            if state["load"] + parcel_weight <= prob.Q[t_idx]:
                inc = prob.D[pos][prob.parc_pick(jid)]
                actions.append(("pickL", jid, inc))

        for jid in list(state["parcels"]):
            inc = prob.D[pos][prob.parc_drop(jid)]
            actions.append(("dropL", jid, inc))


        actions.sort(key=lambda x: x[2])  
        return actions


    def apply_action(self, t_idx: int, kind: str, node_idx: int, inc: int) -> None:
        
        state = self.route_states[t_idx]
        if state["ended"]:
            raise ValueError(f"Cannot apply action on ended route {t_idx}.")

        prob = self.problem

        if kind == "pickP":
            if state["passenger"] != 0:
                raise ValueError(f"Taxi {t_idx} already has passenger {state['passenger']}.")
            node = prob.ppick(node_idx)
            state["passenger"] = node_idx
            self.remaining_pass_pick.discard(node_idx)
            self.remaining_pass_drop.add(node_idx)

        elif kind == "dropP":
            if state["passenger"] != node_idx:
                raise ValueError(f"Taxi {t_idx} is not carrying passenger {node_idx}.")
            node = prob.pdrop(node_idx)
            state["passenger"] = 0
            self.remaining_pass_drop.discard(node_idx)

        elif kind == "pickL":
            parcel_weight = prob.q[node_idx - 1]
            if state["load"] + parcel_weight > prob.Q[t_idx]:
                raise ValueError(f"Taxi {t_idx} capacity exceeded for parcel {node_idx}.")
            node = prob.parc_pick(node_idx)
            state["load"] += parcel_weight
            state["parcels"].add(node_idx)
            self.remaining_parc_pick.discard(node_idx)
            self.remaining_parc_drop.add(node_idx)

        elif kind == "dropL":
            if node_idx not in state["parcels"]:
                raise ValueError(f"Taxi {t_idx} does not carry parcel {node_idx}.")
            node = prob.parc_drop(node_idx)
            state["load"] -= prob.q[node_idx - 1]
            state["parcels"].discard(node_idx)
            self.remaining_parc_drop.discard(node_idx)

        else:
            raise ValueError(f"Unknown action kind: {kind}")

        state["route"].append(node)
        state["cost"] += inc
        state["pos"] = node
        state["actions"] += 1
        self.node_assignment[node] = t_idx
        self.route_costs[t_idx] = state["cost"]
        self.max_cost = max(self.max_cost, state["cost"])
        self.avg_cost = sum(self.route_costs) / self.problem.K
        self.n_actions += 1


    def apply_return_to_depot(self, t_idx: int) -> None:
        
        state = self.route_states[t_idx]

        
        if state["ended"]:
            return
        if state["pos"] == 0 and len(state["route"]) > 1:
            state["ended"] = True
            return

        
        if state["passenger"] != 0 or state["parcels"]:
            raise ValueError(f"Taxi {t_idx} must drop all loads before returning to depot.")

        
        state["cost"] += self.problem.D[state["pos"]][0]
        state["route"].append(0)
        state["pos"] = 0
        state["actions"] += 1
        state["ended"] = True
        self.route_costs[t_idx] = state["cost"]
        self.max_cost = max(self.max_cost, state["cost"])
        self.avg_cost = sum(self.route_costs) / self.problem.K
        self.n_actions += 1


    def reverse_action(self, t_idx: int) -> None:
        
        state = self.route_states[t_idx]

        if len(state["route"]) <= 1:
            raise ValueError(f"No actions to reverse for taxi {t_idx}.")

        
        last_node = state["route"].pop()
        prev_node = state["route"][-1]
        dec_cost = self.problem.D[prev_node][last_node]
        state["cost"] -= dec_cost
        state["pos"] = prev_node
        state["actions"] -= 1
        state["ended"] = False


        
        prob = self.problem
        if prob.is_ppick(last_node):
            pid = prob.rev_ppick(last_node)
            state["passenger"] = 0
            self.remaining_pass_pick.add(pid)
            self.remaining_pass_drop.discard(pid)
        elif prob.is_pdrop(last_node):
            pid = prob.rev_pdrop(last_node)
            state["passenger"] = pid
            self.remaining_pass_pick.discard(pid)
            self.remaining_pass_drop.add(pid)
        elif prob.is_parc_pick(last_node):
            jid = prob.rev_parc_pick(last_node)
            state["load"] -= prob.q[jid - 1]
            state["parcels"].discard(jid)
            self.remaining_parc_pick.add(jid)
            self.remaining_parc_drop.discard(jid)
        elif prob.is_parc_drop(last_node):
            jid = prob.rev_parc_drop(last_node)
            state["load"] += prob.q[jid - 1]
            state["parcels"].add(jid)
            self.remaining_parc_pick.discard(jid)
            self.remaining_parc_drop.add(jid)
        else:
            
            state["ended"] = False

        
        self.route_costs[t_idx] = state["cost"]
        self.max_cost = max(self.route_costs)
        self.avg_cost = sum(self.route_costs) / self.problem.K
        self.node_assignment[last_node] = -1
        self.n_actions -= 1


    def is_complete(self) -> bool:
        
        return all(state["ended"] for state in self.route_states)


    def to_solution(self) -> Optional[Solution]:
        
        if not self.is_complete():
            print("Cannot convert to Solution: not all routes have ended at depot.")
            return None

        solution = Solution(
            problem=self.problem,
            routes=self.routes,
            route_costs=self.route_costs
        )

        if solution is None or not solution.is_valid():
            print("Warning: Converted solution is not valid.")

        return solution


    @staticmethod
    def from_solution(sol: Solution) -> "PartialSolution":
        
        routes_copy = [route.copy() for route in sol.routes]
        return PartialSolution(problem=sol.problem, routes=routes_copy)



class PartialSolutionSwarm:
    

    def __init__(
            self,
            solutions: Optional[List[PartialSolution]] = None,
            n_partials: Optional[int] = None
        ):
        
        if not solutions:
            if n_partials is None or n_partials <= 0:
                raise ValueError("Must provide either solutions list or positive n_partials.")
            self.parsol_list = []
            self.parsol_nact = []
            self.costs = []
            self.min_cost = 0
            self.max_cost = 0
            self.avg_cost = 0.0
            self.best_parsol = None
            return

        self.parsol_list = solutions
        self.parsol_nact = [sol.n_actions for sol in solutions]
        self.costs = [sol.max_cost for sol in solutions]
        self.min_cost = min(self.costs)
        self.max_cost = max(self.costs)
        self.avg_cost = sum(sol.max_cost for sol in solutions) / len(solutions)
        self.best_parsol = min(solutions, key=lambda s: s.max_cost)


    def apply_action_one(
        self, sol_idx: int, t_idx: int, kind: str, node_idx: int, inc: int
    ):
        
        parsol = self.parsol_list[sol_idx]
        parsol.apply_action(t_idx, kind, node_idx, inc)

        
        self.parsol_nact[sol_idx] = parsol.n_actions
        self.costs[sol_idx] = parsol.max_cost

        self.min_cost = min(self.costs)
        self.max_cost = max(self.costs)
        self.avg_cost = sum(self.costs) / len(self.costs)
        if parsol.max_cost == self.min_cost:
            self.best_parsol = parsol


    def apply_return_to_depot_one(self, sol_idx: int, t_idx: int):
        
        parsol = self.parsol_list[sol_idx]
        parsol.apply_return_to_depot(t_idx)

        
        self.parsol_nact[sol_idx] = parsol.n_actions
        self.costs[sol_idx] = parsol.max_cost

        self.min_cost = min(self.costs)
        self.max_cost = max(self.costs)
        self.avg_cost = sum(self.costs) / len(self.costs)
        if parsol.max_cost == self.min_cost:
            self.best_parsol = parsol


    def copy(self):
        
        copied_solutions = [sol.copy() for sol in self.parsol_list]
        return PartialSolutionSwarm(solutions=copied_solutions)


    def extract_best_solution(self) -> Optional[Solution]:
        
        if self.best_parsol and self.best_parsol.is_complete():
            return self.best_parsol.to_solution()

        return None






from typing import Iterator, List, Tuple, Optional




from typing import Callable, Union, Sequence
import math
import bisect


class TreeSegment:
    
    def __init__(
        self,
        data: Sequence[Union[int, float]],
        op: Callable[[Union[int, float], Union[int, float]], Union[int, float]],
        identity: Union[int, float],
        sum_like: bool = True,
        add_neutral: Union[int, float] = 0,
    ):

        self.n_elements = len(data)
        self.op = op
        self.identity = identity

        
        
        self.sum_like = sum_like

        
        self.n_leaves = 1
        while self.n_leaves < self.n_elements:
            self.n_leaves *= 2

        
        self.data = [self.identity] * (2 * self.n_leaves)
        
        self.lazy = [add_neutral] * (2 * self.n_leaves)

        
        for i in range(self.n_elements):
            self.data[self.n_leaves + i] = data[i]
        for i in range(self.n_leaves - 1, 0, -1):
            self.data[i] = self.op(self.data[2 * i], self.data[2 * i + 1])


    
    
    def _apply(self, x: int, val: Union[int, float], length: int):
        if self.sum_like:
            
            self.data[x] += val * length
        else:
            
            self.data[x] += val
        if x < self.n_leaves:
            self.lazy[x] += val


    
    def _push(self, x: int, length: int):
        if self.lazy[x] != 0:
            self._apply(2 * x, self.lazy[x], length // 2)
            self._apply(2 * x + 1, self.lazy[x], length // 2)
            self.lazy[x] = 0


    
    def _update(self, l: int, r: int, val: Union[int, float], x: int, lx: int, rx: int):
        
        if lx >= r or rx <= l:
            return

        
        if lx >= l and rx <= r:
            self._apply(x, val, rx - lx)
            return
        self._push(x, rx - lx)

        
        m = (lx + rx) // 2
        self._update(l, r, val, 2 * x, lx, m)
        self._update(l, r, val, 2 * x + 1, m, rx)
        self.data[x] = self.op(self.data[2 * x], self.data[2 * x + 1])


    
    def _query(self, l: int, r: int, x: int, lx: int, rx: int) -> Union[int, float]:
        
        if lx >= r or rx <= l:
            return self.identity

        
        if lx >= l and rx <= r:
            return self.data[x]

        
        self._push(x, rx - lx)
        m = (lx + rx) // 2
        left = self._query(l, r, 2 * x, lx, m)
        right = self._query(l, r, 2 * x + 1, m, rx)
        return self.op(left, right)


    
    
    def update(self, l: int, r: int, val: Union[int, float]):
        
        self._update(l, r, val, 1, 0, self.n_leaves)


    
    def query(self, l: int, r: int) -> Union[int, float]:
        
        return self._query(l, r, 1, 0, self.n_leaves)



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
            mn = float('inf')
            mx = float('-inf')
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
        
        assert data
        self.block_arr = []
        self.n_data = 0                      
        self.block_prefix: List[int] = []       

        self.build(data)


    def build(self, data: List[int]):
        
        self.block_arr.clear()

        self.n_data: int = len(data)
        self.block_size = max(0, int(math.sqrt(self.n_data))) + 2

        for i in range(0, self.n_data, self.block_size):    
            self.block_arr.append(self.Block(data[i:i + self.block_size]))
        self.n_block = len(self.block_arr)

        self._rebuild_indexing()


    def _rebuild_indexing(self):
        
        self.block_prefix = []

        cumid = 0
        for b in self.block_arr:
            self.block_prefix.append(cumid)
            cumid += b.size

        self.n_data = cumid


    def _find_block(self, idx: int) -> Tuple[int, int]:
        
        assert self.block_arr, "No blocks present"
        assert 0 <= idx < self.n_data, "Index out of bounds"
        if idx > self.n_data:
            idx = self.n_data

        
        
        bid = bisect.bisect_right(self.block_prefix, idx) - 1
        
        
        iid = idx - self.block_prefix[bid]

        return bid, iid


    def insert(self, idx, val):
        
        
        if idx == self.n_data:
            if not self.block_arr:
                self.block_arr.append(self.Block([val]))
            else:
                last = self.block_arr[-1]
                
                if last.size >= 2 * self.block_size:
                    self.block_arr.append(self.Block([val]))
                else:
                    last.insert(last.size, val)

            
            self.n_data += 1
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

        
        self.n_data += 1
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

        
        self.n_data -= 1
        self._rebuild_indexing()


    def query_min_prefix(self, l, r):
        
        ans = float('inf')
        pos = 0
        prefix = 0          
        ans = float('inf')  
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
        
        if idx < 0 or idx >= self.n_data:
            raise IndexError("Index out of bounds")

        bid, iid = self._find_block(idx)

        return self.block_arr[bid].arr[iid]


    def get_data_segment(self, l: int, r: int) -> List[int]:
        
        if l < 0 or r < 0 or l > r or r > self.n_data:
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
        
        return self.get_data_segment(0, self.n_data)






def cost_decrement_relocate(
        partial: PartialSolution,
        from_route_idx: int,
        to_route_idx: int,
        p_idx_from: int,
        q_idx_from: int,
        p_idx_to: int,
        q_idx_to: int
    ) -> Tuple[int, int, int]:
    
    from_route = partial.routes[from_route_idx]
    to_route = partial.routes[to_route_idx]

    
    assert from_route_idx != to_route_idx, \
        "from_route_idx and to_route_idx must be different for relocate."
    assert from_route[p_idx_from] != 0 and from_route[q_idx_from] != 0, \
        "Cannot relocate depot nodes."
    assert 1 <= p_idx_from < q_idx_from, "Invalid pickup/drop indices in from_route."
    assert 1 <= p_idx_to < q_idx_to, "Invalid pickup/drop indices in to_route"

    
    D = partial.problem.D
    cur_cost = partial.max_cost

    
    p_from = from_route[p_idx_from]
    q_from = from_route[q_idx_from]


    
    
    if p_idx_from + 1 == q_idx_from:    
        prev_p_from = from_route[p_idx_from - 1]
        next_q_from = from_route[q_idx_from + 1]
        in_out_from_before = (
            D[prev_p_from][p_from] + D[p_from][q_from] + D[q_from][next_q_from]
        )
        in_out_from_after = (
            D[prev_p_from][next_q_from]
        )
    else:   
        prev_p_from = from_route[p_idx_from - 1]
        next_p_from = from_route[p_idx_from + 1]
        prev_q_from = from_route[q_idx_from - 1]
        next_q_from = from_route[q_idx_from + 1]
        in_out_from_before = (
            D[prev_p_from][p_from] + D[p_from][next_p_from]
            + D[prev_q_from][q_from] + D[q_from][next_q_from]
        )
        in_out_from_after = (
            D[prev_p_from][next_p_from] + D[prev_q_from][next_q_from]
        )

    from_route_next_cost = (
        partial.route_costs[from_route_idx]
        - in_out_from_before + in_out_from_after
    )


    
    
    if q_idx_to == p_idx_to + 1:
        prev_p_from = to_route[p_idx_to - 1]
        next_q_to = to_route[q_idx_to - 1]
        in_out_to_before = (
            D[prev_p_from][next_q_to]
        )
        in_out_to_after = (
            D[prev_p_from][p_from] + D[p_from][q_from] + D[q_from][next_q_to]
        )
    else:
        prev_p_from = to_route[p_idx_to - 1]
        next_p_from = to_route[p_idx_to]
        prev_q_to = to_route[q_idx_to - 2]
        next_q_to = to_route[q_idx_to - 1]
        in_out_to_before = (
            D[prev_p_from][next_p_from] + D[prev_q_to][next_q_to]
        )
        in_out_to_after = (
            D[prev_p_from][p_from] + D[p_from][next_p_from]
            + D[prev_q_to][q_from] + D[q_from][next_q_to]
        )

    
    to_route_next_cost = (
        partial.route_costs[to_route_idx]
        + in_out_to_after - in_out_to_before
    )

    
    next_cost = max(
        from_route_next_cost,
        to_route_next_cost,
        *(
            partial.route_costs[i]
            for i in range(partial.problem.K)
            if i != from_route_idx and i != to_route_idx
        )
    )

    
    return from_route_next_cost, to_route_next_cost, cur_cost - next_cost


def relocate_from_to(
        partial: PartialSolution,   
        from_route_idx: int,        
        to_route_idx: int,          
        steps: int,                 
        mode: str,                  
        uplift: int = 1,            
        seed: int = 42,             
        verbose: bool = False       
    ) -> Tuple[PartialSolution, List[bool], int]:
    
    
    rng = random.Random(seed)

    
    prob = partial.problem
    current_par = partial.copy()
    
    route_from = current_par.routes[from_route_idx]
    route_to = current_par.routes[to_route_idx]
    n_from = len(route_from)
    n_to = len(route_to)

    if n_from < 5:
        return partial, [False] * prob.K, 0    


    
    def build_segment_deltas(route: List[int], n: int):
        pass_delta = [0] * n
        parc_delta = [0] * n
        for i, node in enumerate(route):
            dp = 0
            dq = 0
            if prob.is_ppick(node):
                dp = 1
            elif prob.is_pdrop(node):
                dp = -1
            elif prob.is_parc_pick(node):
                jid = prob.rev_parc_pick(node)
                dq = prob.q[jid - 1]
            elif prob.is_parc_drop(node):
                jid = prob.rev_parc_drop(node)
                dq = -prob.q[jid - 1]
            pass_delta[i] = dp
            parc_delta[i] = dq

        pass_delta_manager = MinMaxPfsumArray(pass_delta)
        parc_delta_manager = MinMaxPfsumArray(parc_delta)

        return pass_delta_manager, parc_delta_manager


    
    pass_delta_from, parc_delta_from = build_segment_deltas(route_from, n_from)
    pass_delta_to, parc_delta_to = build_segment_deltas(route_to, n_to)

    
    cap_from = prob.Q[from_route_idx]
    cap_to = prob.Q[to_route_idx]


    
    def check_passenger(p_idx_a: int, q_idx_a: int, p_idx_b: int, q_idx_b: int) -> bool:
        
        
        
        node_from_pick = route_from[p_idx_a]
        d_pass = 1 if prob.is_ppick(node_from_pick) else 0

        
        if d_pass == 0:
            return True

        
        min_from = pass_delta_from.query_min_prefix(p_idx_a, q_idx_a)
        max_from = pass_delta_from.query_max_prefix(p_idx_a, q_idx_a)
        if min_from - d_pass < 0 or max_from - d_pass > 1:
            return False

        
        min_to = pass_delta_to.query_min_prefix(p_idx_b - 1, q_idx_b - 1)
        max_to = pass_delta_to.query_max_prefix(p_idx_b - 1, q_idx_b - 1)
        if min_to + d_pass < 0 or max_to + d_pass > 1:
            return False

        return True


    def check_parcel(p_idx_a: int, q_idx_a: int, p_idx_b: int, q_idx_b: int) -> bool:
        
        
        node_from_pick = route_from[p_idx_a]
        if prob.is_parc_pick(node_from_pick):
            jid = prob.rev_parc_pick(node_from_pick)
            d_parc = prob.q[jid - 1]
        else:
            d_parc = 0

        
        if d_parc == 0:
            return True

        
        min_from = parc_delta_from.query_min_prefix(p_idx_a, q_idx_a)
        max_from = parc_delta_from.query_max_prefix(p_idx_a, q_idx_a)
        if min_from - d_parc < 0 or max_from - d_parc > cap_from:
            return False

        
        min_to = parc_delta_to.query_min_prefix(p_idx_b - 1, q_idx_b - 1)
        max_to = parc_delta_to.query_max_prefix(p_idx_b - 1, q_idx_b - 1)
        if min_to + d_parc < 0 or max_to + d_parc > cap_to:
            return False

        return True


    def check_relocate(
            p_idx_a: int, q_idx_a: int, p_idx_b: int, q_idx_b: int
        ) -> Tuple[bool, int, int, int]:
        
        if not check_passenger(p_idx_a, q_idx_a, p_idx_b, q_idx_b):
            return False, 0, 0, 0
        if not check_parcel(p_idx_a, q_idx_a, p_idx_b, q_idx_b):
            return False, 0, 0, 0

        
        after_cost_a, after_cost_b, dec = cost_decrement_relocate(
            current_par,
            from_route_idx, to_route_idx,
            p_idx_a, q_idx_a,
            p_idx_b, q_idx_b,
        )
        return True, after_cost_a, after_cost_b, dec


    def find_candidates() -> Iterator[Tuple[int, int, int, int, int, int, int]]:
        
        
        pos_from = {node: i for i, node in enumerate(route_from)}

        
        pickup_indices_from = [
            i for i in range(1, n_from - 1)
            if prob.is_ppick(route_from[i]) or prob.is_parc_pick(route_from[i])
        ]

        
        insertion_pairs_to = [
            (p_to, q_to)
            for p_to in range(1, n_to)
            for q_to in range(p_to + 1, n_to + 1)
        ]

        for p_idx_a in pickup_indices_from:
            
            node_a = route_from[p_idx_a]
            if prob.is_ppick(node_a):
                pid = prob.rev_ppick(node_a)
                q_node_a = prob.pdrop(pid)
            else:
                jid = prob.rev_parc_pick(node_a)
                q_node_a = prob.parc_drop(jid)
            q_idx_a = pos_from.get(q_node_a)
            if q_idx_a is None or q_idx_a <= p_idx_a:
                continue

            
            for p_idx_b, q_idx_b in insertion_pairs_to:
                feasible, after_cost_a, after_cost_b, dec = check_relocate(
                    p_idx_a, q_idx_a, p_idx_b, q_idx_b
                )
                if not feasible or dec < uplift:
                    continue

                if mode == 'first':
                    yield (p_idx_a, q_idx_a, p_idx_b, q_idx_b, after_cost_a, after_cost_b, dec)
                    return
                else:
                    yield (p_idx_a, q_idx_a, p_idx_b, q_idx_b, after_cost_a, after_cost_b, dec)


    def select_candidate() -> Optional[Tuple[int, int, int, int, int, int, int]]:
        
        cand_list = list(find_candidates())
        if not cand_list:
            return None
        if mode == 'stochastic':
            return rng.choice(cand_list)
        elif mode == 'best':
            
            return max(cand_list, key=lambda x: x[6])
        else:
            return cand_list[0]


    
    def update_partial_solution(action: Tuple[int, int, int, int, int, int, int]):
        
        (p_from, q_from, p_to, q_to, new_cost_from, new_cost_to, dec) = action
        nonlocal route_from, route_to, current_par

        
        node_p = route_from[p_from]
        node_q = route_from[q_from]

        
        del route_from[q_from]
        del route_from[p_from]

        
        route_to.insert(p_to, node_p)
        route_to.insert(q_to, node_q)

        
        current_par.routes[from_route_idx] = route_from
        current_par.routes[to_route_idx] = route_to

        
        current_par.route_costs[from_route_idx] = new_cost_from
        current_par.route_costs[to_route_idx] = new_cost_to
        current_par.max_cost -= dec

        return


    def update_precalc(action: Tuple[int, int, int, int, int, int, int]):
        
        nonlocal pass_delta_from, parc_delta_from, pass_delta_to, parc_delta_to
        nonlocal route_from, route_to

        p_from, q_from, p_to, q_to, *_ = action

        
        def node_deltas(node: int) -> Tuple[int, int]:
            if prob.is_ppick(node):
                return 1, 0
            if prob.is_pdrop(node):
                return -1, 0
            if prob.is_parc_pick(node):
                jid = prob.rev_parc_pick(node)
                return 0, prob.q[jid - 1]
            if prob.is_parc_drop(node):
                jid = prob.rev_parc_drop(node)
                return 0, -prob.q[jid - 1]
            return 0, 0

        
        
        pass_delta_from.delete(q_from)
        parc_delta_from.delete(q_from)
        pass_delta_from.delete(p_from)
        parc_delta_from.delete(p_from)


        
        pass_delta_to.insert(p_to, node_deltas(route_from[p_from])[0])
        parc_delta_to.insert(p_to, node_deltas(route_from[p_from])[1])
        pass_delta_to.insert(q_to, node_deltas(route_from[q_from])[0])
        parc_delta_to.insert(q_to, node_deltas(route_from[q_from])[1])

        return


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
            modified_routes[from_route_idx] = True
            modified_routes[to_route_idx] = True

            
            n_from -= 2
            n_to += 2
            if n_from < 5:
                break

            
            if verbose:
                pf, qf, pt, qt, _, __, dec = cand
                print(f"[Relocate {from_route_idx}->{to_route_idx}] moved request "
                      f"(P:{pf},D:{qf}) to ({pt},{qt}) dec={dec}"
                )

            
            if mode == 'first':
                break

        return modified_routes, reloc_done


    
    modified_pair, reloc_done = relocate_to_convergence()
    return current_par, modified_pair, reloc_done



def relocate_operator(
        partial: PartialSolution,           
        steps: Optional[int] = None,        
        mode: str = 'first',                
        uplift: int = 1,                    
        seed: int = 42,                     
        verbose: bool = False,              
    ) -> Tuple[PartialSolution, List[bool], int]:
    
    K = partial.problem.K
    if K < 2:
        return partial.copy(), [False] * K, 0

    
    max_steps = steps if steps is not None else 10**9

    
    rng = random.Random(seed)

    
    current_par: PartialSolution = partial.copy()
    modified_total: List[bool] = [False] * K
    total_moves = 0


    
    
    while total_moves < max_steps:
        
        
        costs: List[Tuple[int, int]] = list(enumerate(current_par.route_costs))
        donor_idx = max(costs, key=lambda x: x[1])[0]
        receivers = [idx for idx, _ in sorted(costs, key=lambda x: x[1])][:max(4, K // 2)]

        
        
        if len(current_par.routes[donor_idx]) < 5:
            break


        
        improved = False
        for r_idx in receivers:
            if r_idx == donor_idx:      
                continue
            
            if len(current_par.routes[r_idx]) < 2:
                continue

            
            remain = max_steps - total_moves
            new_partial, modified_pair, moves_made = relocate_from_to(
                current_par,
                from_route_idx=donor_idx,
                to_route_idx=r_idx,
                steps=remain,
                mode=mode,
                uplift=uplift,
                seed=rng.randint(10, 10**9),  
                verbose=verbose,
            )

            
            if moves_made > 0:
                current_par = new_partial
                total_moves += moves_made
                for i in range(K):
                    if modified_pair[i]:
                        modified_total[i] = True
                improved = True

                
                if verbose:
                    print(f"{moves_made} relocation made from route {donor_idx} to route {r_idx}")

                break   

        if not improved:
            break   

    return current_par, modified_total, total_moves




def _generate_cost_value(
        i: int, j: int, D: List[List[int]], rng: random.Random,
        low: int, high: int, lmbd: Optional[float], asymmetric: bool
    ) -> int:
    
    if i == j:
        return 0
    if asymmetric:
        if lmbd is not None:
            return _sample_poisson(rng, low, high, lmbd)
        return rng.randint(low, high)
    if j < i:
        return D[j][i]
    if lmbd is not None:
        return _sample_poisson(rng, low, high, lmbd)
    return rng.randint(low, high)


def _sample_poisson(
        rng: random.Random,
        low: int,
        high: int,
        lmbd: float,
    ) -> int:
    

    while True:
        
        L = math.exp(-lmbd)
        k = 0
        p = 1.0

        
        while p > L:
            k += 1
            p *= rng.random()
        value = k - 1

        
        if low <= value <= high:
            return value



def random_distance_matrix(
        n: int,
        low: int = 5,
        high: int = 20,
        lmbd: float = 10.0,
        asymmetric: bool = False,
        seed: int = 42,
    ) -> List[List[int]]:
    
    rng = random.Random(seed)
    D = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            D[i][j] = _generate_cost_value(i, j, D, rng, low, high, lmbd, asymmetric)

    return D


def euclidean_distance_matrix(
        coords: List[Tuple[int, int]]
    ) -> List[List[int]]:
    
    n = len(coords)
    D = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            dist = int(round(math.hypot(
                coords[i][0] - coords[j][0],
                coords[i][1] - coords[j][1]
            )))
            D[i][j] = D[j][i] = dist

    return D


def generate_instance_lazy(
        N: int, M: int, K: int,
        low: int = 10, high: int = 50, lmbd: float = 20.0,
        qlow: int = 5, qhigh: int = 15, qlmbd: float = 10.0,
        Qlow: int = 15, Qhigh: int = 30, Qlmbd: float = 20.0,
        use_poisson: bool = False,
        seed: int = 42
    ) -> ShareARideProblem:
    
    rng = random.Random(seed)
    n_nodes = 2*N + 2*M + 1

    if use_poisson:
        q = [_sample_poisson(rng, qlow, qhigh, qlmbd) for _ in range(M)]
        Q = [_sample_poisson(rng, Qlow, Qhigh, Qlmbd) for _ in range(K)]
        D = random_distance_matrix(n_nodes, low=low, high=high, lmbd=lmbd,
                               asymmetric=True, seed=seed)

    else:
        q = [rng.randint(qlow, qhigh) for _ in range(M)]
        Q = [rng.randint(Qlow, Qhigh) for _ in range(K)]
        D = random_distance_matrix(n_nodes, low=low, high=high, lmbd=lmbd,
                               asymmetric=True, seed=seed)

    return ShareARideProblem(N, M, K, q, Q, D)


def generate_instance_coords(
        N: int, M: int, K: int,
        area: int = 100,
        qlow: int = 5, qhigh: int = 15, qlmbd: float = 10.0,
        Qlow: int = 20, Qhigh: int = 45, Qlmbd: float = 30.0,
        seed: int = 42,
    ) -> ShareARideProblem:

    
    rng = random.Random(seed)
    total_points = 1 + 2 * N + 2 * M

    
    coords: List[Tuple[int, int]] = [(area // 2, area // 2)]
    used_coords: set[Tuple[int, int]] = {(area // 2, area // 2)}

    while len(coords) < total_points:
        new_coord = (
            round(rng.random() * area + 0.5), 
            round(rng.random() * area + 0.5)
        )
        if new_coord not in used_coords:
            coords.append(new_coord)
            used_coords.add(new_coord)

    D = euclidean_distance_matrix(coords)
    q = [rng.randint(qlow, qhigh) for _ in range(M)]
    Q = [rng.randint(Qlow, Qhigh) for _ in range(K)]
    prob = ShareARideProblem(N, M, K, q, Q, D, coords)

    return prob








from typing import List, Tuple, Optional







from typing import List

def sample_from_weight(rng: random.Random, weights: List[float]) -> int:
    
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





from typing import List, Union

def softmax_weighter(incs: List[Union[int, float]], T: float) -> List[float]:
    
    min_inc, max_inc = min(incs), max(incs)
    inc_range = max_inc - min_inc

    if inc_range < 1e-6:     
        return [1.0] * len(incs)

    
    weights = []
    for inc in incs:
        normalized = (inc - min_inc) / inc_range
        weights.append((1.0 - normalized + 0.1) ** (1.0 / T))

    return weights




def repair_operator(
        partial: PartialSolution,
        route_idx: int,
        steps: int = 5,
        T: float = 1.0,
        seed: Optional[int] = 42,
        verbose: bool = False
    ) -> Tuple[PartialSolution, List[bool], int]:
    

    assert steps > 0, "Number of steps must be positive."
    assert T > 1e-5, "Temperature T must be positive."

    rng = random.Random(seed)
    added_nodes = 0  

    
    for _ in range(steps):
        state = partial.route_states[route_idx]
        if state["ended"]:
            break

        actions = partial.possible_actions(route_idx)
        if verbose:
            print(f"[build] route {route_idx} available actions: {actions}")

        if not actions:
            if verbose:
                print(f"[build] route {route_idx} has no feasible actions, ending.")

                partial.apply_return_to_depot(route_idx)
                added_nodes += 1
                break

        incs = [action[2] for action in actions]
        weights = softmax_weighter(incs, T)
        selected_idx = sample_from_weight(rng, weights)

        kind, node_idx, inc = actions[selected_idx]

        if verbose:
            print(f"[build] route {route_idx} selected action: {actions[selected_idx]}")

        partial.apply_action(route_idx, kind, node_idx, inc)
        added_nodes += 1

    if verbose:
        print(f"[build] route {route_idx} finished building, added {added_nodes} nodes.")

    modified = [r_idx == route_idx for r_idx in range(partial.problem.K)]
    return partial, modified, added_nodes








from typing import List, Tuple






def destroy_one_route(
        route: List[int],
        route_idx: int,
        steps: int = 10,
        verbose: bool = False
    ) -> List[int]:
    

    res_route = route[:-1]                              
    remove_len = min(steps, max(0, len(res_route) - 1)) 
    if remove_len <= 0:
        return route[:]

    suffix_start = len(res_route) - remove_len
    destroyed_route = res_route[:suffix_start]
    if not destroyed_route:
        destroyed_route = [0]

    if verbose:
        print(f"[Operator: Destroy]: last {remove_len} nodes from route {route_idx} removed.")

    return destroyed_route



def destroy_operator(
        sol: Solution,
        destroy_proba: float,
        destroy_steps: int,
        seed: int = 42,
        T: float = 1.0
    ) -> Tuple[PartialSolution, List[bool], int]:
    
    rng = random.Random(seed)

    routes = [route[:] for route in sol.routes]
    costs = sol.route_costs
    flags = [False] * len(routes)
    num_removed = 0

    if not routes:
        return PartialSolution(problem=sol.problem, routes=routes), flags, num_removed
    approx_destroyed_count = round(destroy_proba * len(routes) + 0.5)
    destroyed_count = min(sol.problem.K, max(1, approx_destroyed_count))

    
    min_cost = min(costs) if costs else 0.0
    max_cost = max(costs) if costs else 1.0
    cost_range = max_cost - min_cost
    temperature = max(T, 1e-6)

    if cost_range < 1e-6:
        
        selected_ids = rng.sample(range(sol.problem.K), destroyed_count)
    else:
        
        weights = []
        for cost in costs:
            normalized = (cost - min_cost) / cost_range
            weights.append((normalized + 0.1) ** (1.0 / temperature))

        
        selected_ids = []
        available_ids = list(range(sol.problem.K))
        available_weights = weights
        for _ in range(destroyed_count):
            total_weight = sum(available_weights)

            if total_weight < 1e-10:    
                selected_ids.extend(
                    available_ids[:destroyed_count - len(selected_ids)]
                )
                break
            else:                       
                selected_idx = sample_from_weight(rng, available_weights)
                selected_ids.append(available_ids[selected_idx])

                
                available_ids.pop(selected_idx)
                available_weights.pop(selected_idx)

                if not available_ids:
                    break


    
    for idx in selected_ids:
        route = routes[idx]

        
        if len(route) <= 2:
            continue

        
        reduced = destroy_one_route(route, idx, steps=destroy_steps, verbose=False)
        removed = max(0, len(route) - len(reduced))

        if removed > 0:
            routes[idx] = reduced
            flags[idx] = True
            num_removed += removed

    partial_sol = PartialSolution(problem=sol.problem, routes=routes)

    return partial_sol, flags, num_removed





def greedy_balanced_solver(
        problem: ShareARideProblem,
        premature_routes: List[List[int]] = [],
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    

    start_time = time.time()
    partial = PartialSolution(problem=problem, routes=premature_routes)
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

        
        if not actions:
            partial.apply_return_to_depot(argmin_t_idx)
            continue

        
        kind, idx, inc = min(actions, key=lambda x: x[2])
        partial.apply_action(argmin_t_idx, kind, idx, inc)

        if verbose:
            print(f"[Greedy] Taxi {argmin_t_idx} extended route with {kind} {idx} (inc {inc})")


    
    for t_idx, t_state in enumerate(taxi_states):
        if not t_state["ended"]:
            partial.apply_return_to_depot(t_idx)

    
    sol = partial.to_solution()

    
    elapsed = time.time() - start_time
    info = {
        "iterations": stats["iterations"],
        "actions_evaluated": stats["actions_evaluated"],
        "time": elapsed
    }

    
    if sol and not sol.is_valid():
        sol = None
    assert sol.is_valid() if sol else True

    
    if verbose:
        print("[Greedy] All tasks completed.")
        print(f"[Greedy] Solution max cost: {sol.max_cost if sol else 'N/A'}")
        print(f"[Greedy] Time taken: {elapsed:.4f} seconds")

    return sol, info


def iterative_greedy_balanced_solver(
        problem: ShareARideProblem,
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
    

    assert 1e-5 < destroy_proba < 1 - 1e-5
    assert 1e-5 < rebuild_proba < 1 - 1e-5
    assert 1 <= rebuild_steps <= destroy_steps

    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + time_limit if time_limit is not None else None

    
    best_sol, base_info = greedy_balanced_solver(problem, verbose=False)
    if not best_sol:
        return None, {"time": time.time() - start_time, "status": "error"}
    best_cost = best_sol.max_cost

    
    total_actions = base_info["actions_evaluated"]
    improvements = 0
    nodes_destroyed = 0
    nodes_rebuilt = 0
    status = "done"
    iterations_done = 0


    if verbose:
        print(f"[Iterative Greedy] [Iter 0] initial best cost: {best_cost}")


    
    for it in range(1, iterations + 1):
        if deadline and time.time() >= deadline:
            status = "timeout"
            break
        iterations_done += 1

        
        destroy_seed = 2 * seed + it
        partial_sol, destroyed_flags, removed = destroy_operator(
            best_sol,
            destroy_proba,
            destroy_steps,
            seed=destroy_seed,
            T=destroy_T
        )
        nodes_destroyed += removed

        
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

        
        sol_cand, info_cand = greedy_balanced_solver(
            problem,
            premature_routes=partial_sol.routes,
            verbose=False
        )

        total_actions += info_cand["actions_evaluated"]

        
        if (sol_cand and sol_cand.is_valid()
            and sol_cand.max_cost < best_cost
        ):
            best_sol = sol_cand
            best_cost = sol_cand.max_cost
            improvements += 1

            if verbose:
                print(f"[Iterative Greedy] [Iter {it}] improved best to {best_cost}")


    
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

    
    if verbose:
        print(f"[Iterative Greedy] Finished after {iterations_done} iterations.")
        print(
            f"[Iterative Greedy] Best solution max cost: "
            f"{best_sol.max_cost if best_sol else 'N/A'}."
        )
        print(f"[Iterative Greedy] Time taken: {elapsed:.4f} seconds.")

    return best_sol, info



    
    problem = generate_instance_coords(
        N=200, M=300, K=7, area=1000, seed=12345
    )
    
    
    
    

    sol, msg = greedy_balanced_solver(problem)
    assert sol
    
    
    

    st = time.time()
    par = PartialSolution.from_solution(sol)
    next_par, modified, n_moves = relocate_operator(
        par,
        steps=None,
        mode='first',
        seed=100,
        verbose=True
    )

    sol_after = next_par.to_solution()
    assert sol_after
    
    
    

    print()
    print(f"Relocate operator performed {n_moves} moves, modified routes: {modified}")
    print(f"Solution cost before: {sol.max_cost}, after: {sol_after.max_cost}")
    print(f"Relocate operator time: {time.time() - st:.4f} seconds" )






import cProfile



from typing import Any, List, Optional, Tuple, Dict







import heapq
from typing import List, Optional, Tuple







def cost_decrement_intra_swap(
        partial: PartialSolution,
        route_idx: int,
        a_idx: int,
        b_idx: int
    ) -> int:
    
    assert a_idx != b_idx, "Indices to swap must be different."
    if a_idx > b_idx:
        a_idx, b_idx = b_idx, a_idx

    route = partial.routes[route_idx]
    assert route[a_idx] != 0 and route[b_idx] != 0, "Cannot swap depot nodes."

    D = partial.problem.D

    
    def is_node(idx: int) -> Optional[int]:
        
        
        return route[idx] \
            if 0 <= idx <= partial.route_states[route_idx]["actions"] else None

    def successor_cost(from_node: int, to_node: Optional[int]) -> int:
        if to_node is None:
            return 0
        return D[from_node][to_node]


    
    if a_idx < b_idx - 1:
        delta = (
            D[route[a_idx - 1]][route[a_idx]]
            + successor_cost(route[a_idx], is_node(a_idx + 1))
            + D[route[b_idx - 1]][route[b_idx]]
            + successor_cost(route[b_idx], is_node(b_idx + 1))
            - D[route[a_idx - 1]][route[b_idx]]
            - successor_cost(route[b_idx], is_node(a_idx + 1))
            - D[route[b_idx - 1]][route[a_idx]]
            - successor_cost(route[a_idx], is_node(b_idx + 1))
        )
    else:
        delta = (
            D[route[a_idx - 1]][route[a_idx]] + D[route[a_idx]][route[b_idx]]
            + successor_cost(route[b_idx], is_node(b_idx + 1))
            - D[route[a_idx - 1]][route[b_idx]] - D[route[b_idx]][route[a_idx]]
            - successor_cost(route[a_idx], is_node(b_idx + 1))
        )

    return delta




def intra_swap_one_route_operator(
        partial: PartialSolution,               
        route_idx: int,                     
        steps: Optional[int] = None,        
        mode: str = 'first',                
        uplift: int = 1,                    
        seed: int = 42,                     
        verbose: bool = False               
    ) -> Tuple[PartialSolution, List[bool], int]:
    

    
    rng = random.Random(seed)

    
    current_par = partial.copy()
    prob = current_par.problem
    K = prob.K
    route = current_par.routes[route_idx]
    n = len(route)


    
    if n < 5:
        return current_par, [False] * K, 0
    if steps is None:
        steps = n ** 2

    
    pos = {node: idx for idx, node in enumerate(route)}
    
    pass_load: List[int] = [0] * n           
    pass_delta: List[int] = [0] * n          
    parc_load: List[int] = [0] * n           
    parc_delta: List[int] = [0] * n          

    onboard_pass = 0
    onboard_parcels = 0
    for i in range(n):
        node = route[i]

        delta_pass = 0
        delta_parc = 0
        if prob.is_ppick(node):
            delta_pass = 1
        elif prob.is_pdrop(node):
            delta_pass = -1
        elif prob.is_parc_pick(node):
            jid = prob.rev_parc_pick(node)
            delta_parc = prob.q[jid - 1]
        elif prob.is_parc_drop(node):
            jid = prob.rev_parc_drop(node)
            delta_parc = -prob.q[jid - 1]

        onboard_pass += delta_pass
        onboard_parcels += delta_parc
        pass_load[i] = onboard_pass
        pass_delta[i] = delta_pass
        parc_load[i] = onboard_parcels
        parc_delta[i] = delta_parc


    
    min_pass_segment = TreeSegment(
        data=pass_load,
        op=min,
        identity=float('inf'),
        sum_like=False
    )
    max_pass_segment = TreeSegment(
        data=pass_load,
        op=max,
        identity=0,
        sum_like=False
    )
    min_parc_segment = TreeSegment(
        data=parc_load,
        op=min,
        identity=float('inf'),
        sum_like=False
    )
    max_parc_segment = TreeSegment(
        data=parc_load,
        op=max,
        identity=0,
        sum_like=False
    )


    
    
    def check_precedence(a: int, b: int) -> bool:
        assert a != b

        def new_idx(idx: int) -> int:
            if idx == a:
                return b
            if idx == b:
                return a
            return idx

        def check_node(idx_old: int) -> bool:
            node = route[idx_old]
            
            if prob.is_ppick(node):
                pid = prob.rev_ppick(node)
                pair = prob.pdrop(pid)
                drop_idx = pos.get(pair)    

                if drop_idx is None:                    
                    return True
                return new_idx(idx_old) < new_idx(drop_idx)

            if prob.is_pdrop(node):
                pid = prob.rev_pdrop(node)
                pair = prob.ppick(pid)
                pickup_idx = pos.get(pair)  

                if pickup_idx is None:                  
                    return False
                return new_idx(pickup_idx) < new_idx(idx_old)

            
            if prob.is_parc_pick(node):
                jid = prob.rev_parc_pick(node)
                pair = prob.parc_drop(jid)
                drop_idx = pos.get(pair)
                if drop_idx is None:
                    return True
                return new_idx(idx_old) < new_idx(drop_idx)
            if prob.is_parc_drop(node):
                jid = prob.rev_parc_drop(node)
                pair = prob.parc_pick(jid)
                pickup_idx = pos.get(pair)
                if pickup_idx is None:
                    return False
                return new_idx(pickup_idx) < new_idx(idx_old)

            
            return False

        return check_node(a) and check_node(b)


    def check_passenger(a: int, b: int) -> bool:
        assert a != b

        if a > b:
            a, b = b, a

        swap_delta_pass = pass_delta[b] - pass_delta[a]
        if swap_delta_pass > 0:
            return max_pass_segment.query(a, b) + swap_delta_pass <= 1
        elif swap_delta_pass < 0:
            return min_pass_segment.query(a, b) + swap_delta_pass >= 0
        else:
            return True


    def check_parcel(a: int, b: int) -> bool:
        assert a != b

        if a > b:
            a, b = b, a

        swap_delta_parc = parc_delta[b] - parc_delta[a]
        if swap_delta_parc > 0:
            return max_parc_segment.query(a, b) <= prob.Q[route_idx] - swap_delta_parc
        elif swap_delta_parc < 0:
            return min_parc_segment.query(a, b) >= -swap_delta_parc
        else:
            return True


    
    def check_swap(a: int, b: int) -> Tuple[bool, int]:
        if not (check_precedence(a, b)
            and check_passenger(a, b)
            and check_parcel(a, b)
        ):
            return False, 0

        
        dec = cost_decrement_intra_swap(
            current_par, route_idx, a, b
        )
        return True, dec

    
    def find_candidates():
        for a in range(1, n - 1):
            for b in range(a + 1, n - 1):
                feasible, dec = check_swap(a, b)
                if not feasible or dec < uplift:
                    continue

                if mode == 'first':     
                    yield (a, b, dec)
                    return
                else:
                    yield (a, b, dec)


    
    def select_candidate():
        cand_list = list(find_candidates())
        if not cand_list:
            return
        if mode == 'stochastic':
            return rng.choice(cand_list)
        elif mode == 'best':
            return max(cand_list, key=lambda x: x[2])
        else:
            return cand_list[0]


    
    
    swaps_done = 0
    best_improvement = 0
    modified = [False] * K


    def update_partial_solution(action: Tuple[int, int, int]) :
        nonlocal route
        nonlocal best_improvement
        nonlocal modified
        nonlocal current_par

        a, b, dec = action
        route[a], route[b] = route[b], route[a]
        current_par.route_costs[route_idx] -= dec
        current_par.max_cost = max(current_par.max_cost, current_par.route_costs[route_idx])


    def update_precalc(action: Tuple[int, int, int]) :
        nonlocal route
        nonlocal pos
        nonlocal pass_delta
        nonlocal parc_delta
        nonlocal min_pass_segment
        nonlocal min_parc_segment
        nonlocal max_pass_segment
        nonlocal max_parc_segment

        a, b, _ = action
        if a > b:
            a, b = b, a

        
        
        dpass = pass_delta[b] - pass_delta[a]
        if dpass != 0:
            min_pass_segment.update(a, b, dpass)
            max_pass_segment.update(a, b, dpass)
        dparc = parc_delta[b] - parc_delta[a]
        if dparc != 0:
            min_parc_segment.update(a, b, dparc)
            max_parc_segment.update(a, b, dparc)

        
        pass_delta[a], pass_delta[b] = pass_delta[b], pass_delta[a]
        parc_delta[a], parc_delta[b] = parc_delta[b], parc_delta[a]

        
        pos[route[a]] = a
        pos[route[b]] = b



    def swap_until_convergence():
        nonlocal swaps_done
        while True:
            if steps is not None and swaps_done >= steps:
                break

            action = select_candidate()
            if action is None:
                break

            update_partial_solution(action)
            update_precalc(action)

            swaps_done += 1
            modified[route_idx] = True

            if verbose:
                a, b, dec = action
                print(
                    f"[Route {route_idx}] Swapped positions {a} and {b} "
                    + f"(nodes {route[b]} and {route[a]}). Cost decrease: {dec}."
                )

    swap_until_convergence()
    return current_par, modified, swaps_done



def intra_swap_operator(
        partial: PartialSolution,               
        steps: Optional[int] = None,        
        mode: str = 'first',                
        uplift: int = 1,                    
        seed: int = 42,                     
        verbose: bool = False
    ) -> Tuple[PartialSolution, List[bool], int]:
    
    if steps is None:
        steps = 10**9

    total_swaps = 0
    modified: List[bool] = [False] * partial.problem.K
    current_par: PartialSolution = partial.copy()
    for route_it in range(partial.problem.K):
        tmp_par, modified_one, n_swaps_one = intra_swap_one_route_operator(
            current_par,
            route_idx=route_it,
            steps=(steps - total_swaps),
            mode=mode,
            uplift=uplift,
            seed=seed,
            verbose=verbose
        )

        current_par = tmp_par
        total_swaps += n_swaps_one
        if modified_one[route_it]:
            modified[route_it] = True

        if verbose:
            print(f"Route {route_it}: performed {n_swaps_one} intra-route swaps.")

    return current_par, modified, total_swaps



def cost_decrement_inter_swap(
        partial: PartialSolution,
        route_a_idx: int, route_b_idx: int,
        p_idx_a: int, d_idx_a: int,
        p_idx_b: int, d_idx_b: int
    ) -> Tuple[int, int, int]:
    
    route_a = partial.routes[route_a_idx]
    route_b = partial.routes[route_b_idx]
    assert route_a[p_idx_a] != 0 and route_b[p_idx_b] != 0, "Cannot swap depot nodes."

    route_cost_a = partial.route_costs[route_a_idx]
    route_cost_b = partial.route_costs[route_b_idx]
    max_cost_before = partial.max_cost


    
    D = partial.problem.D

    
    
    
    def _succ_cost(from_node: int, routechar: str, idx: int) -> int:
        
        if routechar == 'a':
            route = route_a
            route_idx = route_a_idx
        else:
            route = route_b
            route_idx = route_b_idx

        if idx >= partial.route_states[route_idx]["actions"]:
            return 0
        return D[from_node][route[idx + 1]]

    
    if p_idx_a + 1 == d_idx_a:
        in_out_cost_a_before = (
            D[route_a[p_idx_a - 1]][route_a[p_idx_a]]
            + D[route_a[p_idx_a]][route_a[d_idx_a]]
            + _succ_cost(route_a[d_idx_a], 'a', d_idx_a)
        )
        in_out_cost_a_after = (
            D[route_a[p_idx_a - 1]][route_b[p_idx_b]]
            + D[route_b[p_idx_b]][route_b[d_idx_b]]
            + _succ_cost(route_b[d_idx_b], 'a', d_idx_a)
        )
    else:
        in_out_cost_a_before = (
            D[route_a[p_idx_a - 1]][route_a[p_idx_a]]
            + D[route_a[p_idx_a]][route_a[p_idx_a + 1]]
            + D[route_a[d_idx_a - 1]][route_a[d_idx_a]]
            + _succ_cost(route_a[d_idx_a], 'a', d_idx_a)
        )
        in_out_cost_a_after = (
            D[route_a[p_idx_a - 1]][route_b[p_idx_b]]
            + D[route_b[p_idx_b]][route_a[p_idx_a + 1]]
            + D[route_a[d_idx_a - 1]][route_b[d_idx_b]]
            + _succ_cost(route_b[d_idx_b], 'a', d_idx_a)
        )
    if p_idx_b + 1 == d_idx_b:
        in_out_cost_b_before = (
            D[route_b[p_idx_b - 1]][route_b[p_idx_b]]
            + D[route_b[p_idx_b]][route_b[d_idx_b]]
            + _succ_cost(route_b[d_idx_b], 'b', d_idx_b)
        )
        in_out_cost_b_after = (
            D[route_b[p_idx_b - 1]][route_a[p_idx_a]]
            + D[route_a[p_idx_a]][route_a[d_idx_a]]
            + _succ_cost(route_a[d_idx_a], 'b', d_idx_b)
        )
    else:
        in_out_cost_b_before = (
            D[route_b[p_idx_b - 1]][route_b[p_idx_b]] 
            + D[route_b[p_idx_b]][route_b[p_idx_b + 1]]
            + D[route_b[d_idx_b - 1]][route_b[d_idx_b]]
            + _succ_cost(route_b[d_idx_b], 'b', d_idx_b)
        )
        in_out_cost_b_after = (
            D[route_b[p_idx_b - 1]][route_a[p_idx_a]]
            + D[route_a[p_idx_a]][route_b[p_idx_b + 1]]
            + D[route_b[d_idx_b - 1]][route_a[d_idx_a]]
            + _succ_cost(route_a[d_idx_a], 'b', d_idx_b)
        )

    
    route_a_next_cost = route_cost_a - in_out_cost_a_before + in_out_cost_a_after
    route_b_next_cost = route_cost_b - in_out_cost_b_before + in_out_cost_b_after
    max_cost_after = max(
        route_a_next_cost,
        route_b_next_cost,
        *(
            partial.route_costs[i] for i in range(partial.problem.K)
            if i != route_a_idx and i != route_b_idx
        )
    )

    return route_a_next_cost, route_b_next_cost, max_cost_before - max_cost_after



def inter_swap_route_pair_operator (
        partial: PartialSolution,
        route_a_idx: int,
        route_b_idx: int,
        steps: Optional[int] = None,
        mode: str = 'first',
        uplift: int = 1,
        seed: int = 42,
        verbose: bool = False
    ) -> Tuple[PartialSolution, List[bool], int]:

    

    
    rng = random.Random(seed)

    
    current_par = partial.copy()
    prob = current_par.problem
    
    route_a = current_par.routes[route_a_idx]
    route_b = current_par.routes[route_b_idx]
    n_a = len(route_a)
    n_b = len(route_b)

    
    if n_a < 5 or n_b < 5:
        return current_par, [False] * prob.K, 0

    
    def build_loads_and_trees(route: List[int]):
        n = len(route)
        pass_load = [0] * n
        pass_delta = [0] * n
        parc_load = [0] * n
        parc_delta = [0] * n
        onboard_pass = 0
        onboard_parc = 0
        for i, node in enumerate(route):
            dp = 0
            dq = 0
            if prob.is_ppick(node):
                dp = 1
            elif prob.is_pdrop(node):
                dp = -1
            elif prob.is_parc_pick(node):
                jid = prob.rev_parc_pick(node)
                dq = prob.q[jid - 1]
            elif prob.is_parc_drop(node):
                jid = prob.rev_parc_drop(node)
                dq = -prob.q[jid - 1]
            onboard_pass += dp
            onboard_parc += dq
            pass_load[i] = onboard_pass
            pass_delta[i] = dp
            parc_load[i] = onboard_parc
            parc_delta[i] = dq

        
        min_pass_segment = TreeSegment(
            data=pass_load, op=min, identity=float('inf'), sum_like=False
        )
        max_pass_segment = TreeSegment(
            data=pass_load, op=max, identity=0, sum_like=False
        )
        min_parc_segment = TreeSegment(
            data=parc_load, op=min, identity=float('inf'), sum_like=False
        )
        max_parc_segment = TreeSegment(
            data=parc_load, op=max, identity=0, sum_like=False
        )
        pos = {node: i for i, node in enumerate(route)}
        return pos, pass_delta, parc_delta, (
            min_pass_segment, max_pass_segment, min_parc_segment, max_parc_segment
        )

    
    pos_a, pass_delta_a, parc_delta_a, trees_a = build_loads_and_trees(route_a)
    pos_b, pass_delta_b, parc_delta_b, trees_b = build_loads_and_trees(route_b)
    min_pass_seg_a, max_pass_seg_a, min_parc_seg_a, max_parc_seg_a = trees_a
    min_pass_seg_b, max_pass_seg_b, min_parc_seg_b, max_parc_seg_b = trees_b

    cap_a = prob.Q[route_a_idx]
    cap_b = prob.Q[route_b_idx]



    
    def check_passenger(p_idx_a: int, q_idx_a: int, p_idx_b: int, q_idx_b: int) -> bool:
        
        
        delta_a = pass_delta_b[p_idx_b] - pass_delta_a[p_idx_a]
        if delta_a != 0:
            min_a = min_pass_seg_a.query(p_idx_a, q_idx_a)
            max_a = max_pass_seg_a.query(p_idx_a, q_idx_a)
            if min_a + delta_a < 0 or max_a + delta_a > 1:
                return False

        
        delta_b = pass_delta_a[p_idx_a] - pass_delta_b[p_idx_b]
        if delta_b != 0:
            min_b = min_pass_seg_b.query(p_idx_b, q_idx_b)
            max_b = max_pass_seg_b.query(p_idx_b, q_idx_b)
            if min_b + delta_b < 0 or max_b + delta_b > 1:
                return False

        return True


    def check_parcel(p_idx_a: int, q_idx_a: int, p_idx_b: int, q_idx_b: int) -> bool:
        
        
        delta_a = parc_delta_b[p_idx_b] - parc_delta_a[p_idx_a]
        if delta_a != 0:
            min_a = min_parc_seg_a.query(p_idx_a, q_idx_a)
            max_a = max_parc_seg_a.query(p_idx_a, q_idx_a)
            if min_a + delta_a < 0 or max_a + delta_a > cap_a:
                return False
        
        delta_b = parc_delta_a[p_idx_a] - parc_delta_b[p_idx_b]
        if delta_b != 0:
            min_b = min_parc_seg_b.query(p_idx_b, q_idx_b)
            max_b = max_parc_seg_b.query(p_idx_b, q_idx_b)
            if min_b + delta_b < 0 or max_b + delta_b > cap_b:
                return False
        return True


    def check_swap(
            p_idx_a: int, q_idx_a: int, p_idx_b: int, q_idx_b: int
        ) -> Tuple[bool, int, int, int]:
        if not check_passenger(p_idx_a, q_idx_a, p_idx_b, q_idx_b):
            return False, 0, 0, 0
        if not check_parcel(p_idx_a, q_idx_a, p_idx_b, q_idx_b):
            return False, 0, 0, 0

        
        after_cost_a, after_cost_b, dec = cost_decrement_inter_swap(
            current_par,
            route_a_idx, route_b_idx,
            p_idx_a, q_idx_a,
            p_idx_b, q_idx_b,
        )
        return True, after_cost_a, after_cost_b, dec



    
    def find_candidates():
        pickup_indices_a = [
            i for i in range(n_a)
            if prob.is_ppick(route_a[i]) or prob.is_parc_pick(route_a[i])
        ]
        pickup_indices_b = [
            j for j in range(n_b)
            if prob.is_ppick(route_b[j]) or prob.is_parc_pick(route_b[j])
        ]

        for p_idx_a in pickup_indices_a:
            
            p_node_a = route_a[p_idx_a]
            if prob.is_ppick(p_node_a):
                pass_id = prob.rev_ppick(p_node_a)
                q_node_a = prob.pdrop(pass_id)
            else:
                parc_id = prob.rev_parc_pick(p_node_a)
                q_node_a = prob.parc_drop(parc_id)
            q_idx_a = pos_a.get(q_node_a)
            if q_idx_a is None:
                continue

            for p_idx_b in pickup_indices_b:
                
                p_node_b = route_b[p_idx_b]
                if prob.is_ppick(p_node_b):
                    pass_idx_b = prob.rev_ppick(p_node_b)
                    q_node_b = prob.pdrop(pass_idx_b)
                else:
                    parc_id_b = prob.rev_parc_pick(p_node_b)
                    q_node_b = prob.parc_drop(parc_id_b)
                q_idx_b = pos_b.get(q_node_b)
                if q_idx_b is None:
                    continue

                
                
                

                feasible, after_cost_a, after_cost_b, dec = check_swap(
                    p_idx_a, q_idx_a, p_idx_b, q_idx_b
                )
                if not feasible or dec < uplift:
                    continue
                if mode == 'first':
                    yield (p_idx_a, q_idx_a, p_idx_b, q_idx_b, after_cost_a, after_cost_b, dec)
                    return
                else:
                    yield (p_idx_a, q_idx_a, p_idx_b, q_idx_b, after_cost_a, after_cost_b, dec)


    def select_candidate():
        cand_list = list(find_candidates())
        if not cand_list:
            return None
        if mode == 'stochastic':
            return rng.choice(cand_list)
        elif mode == 'best':
            return max(cand_list, key=lambda x: x[4])
        else:
            return cand_list[0]


    
    
    swaps_done = 0
    best_improvement = 0
    modified = [False] * prob.K

    
    if steps is None:
        steps = (n_a + n_b) ** 2


    def update_partial_solution(action: Tuple[int, int, int, int, int, int, int]):
        nonlocal route_a, route_b, current_par
        pA, qA, pB, qB, new_cost_a, new_cost_b, decrement = action

        
        a1, a2 = route_a[pA], route_a[qA]
        b1, b2 = route_b[pB], route_b[qB]
        route_a[pA], route_a[qA] = b1, b2
        route_b[pB], route_b[qB] = a1, a2

        
        current_par.route_costs[route_a_idx] = new_cost_a
        current_par.route_costs[route_b_idx] = new_cost_b
        current_par.max_cost -= decrement


    def update_precalc(action: Tuple[int, int, int, int, int, int, int]):
        nonlocal pos_a, pos_b
        nonlocal pass_delta_a, pass_delta_b
        nonlocal parc_delta_a, parc_delta_b
        nonlocal min_pass_seg_a, max_pass_seg_a, min_parc_seg_a, max_parc_seg_a
        nonlocal min_pass_seg_b, max_pass_seg_b, min_parc_seg_b, max_parc_seg_b

        pA, qA, pB, qB, _, __, ___ = action

        
        dpass_a = pass_delta_b[pB] - pass_delta_a[pA]
        dparc_a = parc_delta_b[pB] - parc_delta_a[pA]
        if dpass_a != 0:
            min_pass_seg_a.update(pA, qA, dpass_a)
            max_pass_seg_a.update(pA, qA, dpass_a)
        if dparc_a != 0:
            min_parc_seg_a.update(pA, qA, dparc_a)
            max_parc_seg_a.update(pA, qA, dparc_a)

        dpass_b = pass_delta_a[pA] - pass_delta_b[pB]
        dparc_b = parc_delta_a[pA] - parc_delta_b[pB]
        if dpass_b != 0:
            min_pass_seg_b.update(pB, qB, dpass_b)
            max_pass_seg_b.update(pB, qB, dpass_b)
        if dparc_b != 0:
            min_parc_seg_b.update(pB, qB, dparc_b)
            max_parc_seg_b.update(pB, qB, dparc_b)

        
        pass_delta_a[pA], pass_delta_b[pB] = pass_delta_b[pB], pass_delta_a[pA]
        pass_delta_a[qA], pass_delta_b[qB] = pass_delta_b[qB], pass_delta_a[qA]
        parc_delta_a[pA], parc_delta_b[pB] = parc_delta_b[pB], parc_delta_a[pA]
        parc_delta_a[qA], parc_delta_b[qB] = parc_delta_b[qB], parc_delta_a[qA]

        
        pos_a[route_a[pA]] = pA
        pos_a[route_a[qA]] = qA
        pos_b[route_b[pB]] = pB
        pos_b[route_b[qB]] = qB


    def swap_until_convergence():
        nonlocal swaps_done, modified, best_improvement
        while swaps_done < steps:
            action = select_candidate()
            if action is None:
                break

            update_partial_solution(action)
            update_precalc(action)

            best_improvement += action[6]
            modified[route_a_idx] = True
            modified[route_b_idx] = True
            swaps_done += 1

            if verbose:
                pA, qA, pB, qB, _, __, dec = action
                print(
                    f"[Routes {route_a_idx} & {route_b_idx}] "
                    + f"Swapped nodes at positions ({pA}, {qA}) and ({pB}, {qB}). "
                    + f"Cost decrease: {dec}."
                )

    swap_until_convergence()
    return current_par, modified, swaps_done



def inter_swap_operator(
        partial: PartialSolution,               
        steps: Optional[int] = None,        
        mode: str = 'first',                
        uplift: int = 1,                    
        seed: int = 42,                     
        verbose: bool = False               
    ) -> Tuple[PartialSolution, List[bool], int]:
    
    rng = random.Random(seed)

    K = partial.problem.K
    if K < 2:
        return partial.copy(), [False] * K, 0

    current_par: PartialSolution = partial.copy()
    modified: List[bool] = [False] * K
    total_swaps = 0

    
    max_steps = steps if steps is not None else 10**9

    
    max_heap: List[Tuple[int, int]] = [(-c, i) for i, c in enumerate(current_par.route_costs)]
    min_heap: List[Tuple[int, int]] = [(c, i) for i, c in enumerate(current_par.route_costs)]
    heapq.heapify(max_heap)
    heapq.heapify(min_heap)

    def pop_valid_max() -> Optional[Tuple[int, int]]:
        while max_heap:
            negc, idx = heapq.heappop(max_heap)
            if -negc == current_par.route_costs[idx]:
                return -negc, idx
        return None

    def pop_valid_min(exclude_idx: Optional[int] = None) -> Optional[Tuple[int, int]]:
        while min_heap:
            c, idx = heapq.heappop(min_heap)
            if idx == exclude_idx:
                continue
            if c == current_par.route_costs[idx]:
                return c, idx
        return None

    def push_idx(idx: int):
        c = current_par.route_costs[idx]
        heapq.heappush(max_heap, (-c, idx))
        heapq.heappush(min_heap, (c, idx))


    
    while True:
        if steps is not None and total_swaps >= max_steps:
            break
        top = pop_valid_max()
        if top is None:
            break
        _, max_idx = top

        
        popped_mins: List[Tuple[int, int]] = []
        improved = False
        while True:
            mn = pop_valid_min(exclude_idx=max_idx)
            if mn is None:
                break
            _, min_idx = mn
            popped_mins.append(mn)

            next_par, modified_pair, n_swaps_pair = inter_swap_route_pair_operator(
                current_par,
                route_a_idx=max_idx,
                route_b_idx=min_idx,
                steps=(max_steps - total_swaps),
                mode=mode,
                uplift=uplift,
                seed=rng.randint(10, 10**9),
                verbose=verbose
            )
            if n_swaps_pair > 0:
                
                current_par = next_par
                push_idx(max_idx)
                push_idx(min_idx)

                total_swaps += n_swaps_pair
                if modified_pair[max_idx]:
                    modified[max_idx] = True
                if modified_pair[min_idx]:
                    modified[min_idx] = True
                improved = True

                if verbose:
                    print(
                        f"Inter-route swap between routes {max_idx} and {min_idx} "
                        + f"performed {n_swaps_pair} swaps."
                    )

                break

        
        
        for c, idx in popped_mins:
            heapq.heappush(min_heap, (c, idx))

        if not improved:
            
            
            
            
            push_idx(max_idx)
            break

    return current_par, modified, total_swaps







def parsol_scorer(
        parsol: PartialSolution,
        sample_size: int = 15,
        w_std: float = 0.15,
        seed: int = 42,
    ) -> float:
    
    rng = random.Random(seed)
    effective_size = max(1, sample_size)
    costs = parsol.route_costs
    if len(costs) == 1:
        return parsol.max_cost

    sampled = rng.choices(costs, k=effective_size)
    mean = math.fsum(sampled) / len(sampled)
    variance = math.fsum((value - mean) ** 2 for value in sampled) / len(sampled)
    std_dev = math.sqrt(max(0.0, variance))

    return parsol.max_cost + w_std * std_dev



def beam_search_swarm_solver(
        problem: ShareARideProblem,
        cost_function: Any = parsol_scorer,
        initial: Optional[PartialSolutionSwarm] = None,
        l_width: int = 10,
        r_intra: float = 0.75,
        r_inter: float = 0.90,
        f_intra: float = 0.05,
        f_inter: float = 0.10,
        verbose: bool = False
    ) -> Tuple[PartialSolutionSwarm, Dict[str, Any]]:
    
    
    start = time.time()
    total_actions = max(1, 2 * (problem.N + problem.M) + problem.K)

    
    def _clamp(value: float) -> float:
        return min(max(value, 0.0), 1.0)
    r_intra_clamped = _clamp(r_intra)
    r_inter_clamped = _clamp(r_inter)
    f_intra_clamped = _clamp(f_intra)
    f_inter_clamped = _clamp(f_inter)

    
    until_intra_depth = max(0, int(total_actions * r_intra_clamped))
    until_inter_depth = max(0, int(total_actions * r_inter_clamped))
    intra_frequency = max(1, int(total_actions * f_intra_clamped))
    inter_frequency = max(1, int(total_actions * f_inter_clamped))


    
    def expand(parsol: PartialSolution) -> List[PartialSolution]:
        candidates: List[PartialSolution] = []

        
        available_taxis = [
            idx for idx, state in enumerate(parsol.route_states)
            if not state["ended"]
        ]
        if not available_taxis:
            return candidates

        
        taxi_order = sorted(
            available_taxis,
            key=lambda idx: parsol.route_states[idx]["cost"]
        )
        taxi_considered = min(
            2 if problem.K >= 50
            else 3 if problem.K >= 20
            else 4,
            len(taxi_order)
        )
        taxi_branches = taxi_order[:taxi_considered]

        
        
        closing_depth = max(0, total_actions - 2 * parsol.problem.K)

        
        for t_idx in taxi_branches:
            state = parsol.route_states[t_idx]
            actions = parsol.possible_actions(t_idx)

            
            can_return = (
                state["passenger"] == 0 and
                not state["parcels"] and
                not state["ended"] and
                state["pos"] != 0
            )

            
            if actions:
                
                action_limit = min(
                    1 if problem.num_nodes >= 500   
                    else 2 if problem.num_nodes >= 200  
                    else 4, 
                    len(actions)
                )
                for kind, node_idx, inc in sorted(actions, key=lambda item: item[2])[:action_limit]:
                    parsol.apply_action(t_idx, kind, node_idx, inc)
                    candidates.append(parsol.copy())
                    parsol.reverse_action(t_idx)

                if can_return and depth >= closing_depth:
                    parsol.apply_return_to_depot(t_idx)
                    candidates.append(parsol.copy())
                    parsol.reverse_action(t_idx)

            
            elif can_return:
                parsol.apply_return_to_depot(t_idx)
                candidates.append(parsol.copy())
                parsol.reverse_action(t_idx)

            
            
            else:
                if verbose:
                    print(
                        f"[BeamSearch] Taxi {t_idx} has no feasible actions "
                        f"and cannot return to depot."
                    )

        
        candidates.sort(key=cost_function)
        return candidates


    
    def apply_local_refinements(
            parsols: List[PartialSolution],
            use_intra: bool,
            use_inter: bool,
            seed_offset: int
        ) -> List[PartialSolution]:
        if not (use_intra or use_inter):
            return parsols

        
        refined: List[PartialSolution] = []
        for idx, base in enumerate(parsols):
            updated = base
            if use_intra:
                updated, _, _ = intra_swap_operator(
                    updated,
                    steps=None,
                    mode='first',
                    uplift=1,
                    seed=1337 + seed_offset + idx,
                    verbose=False
                )
            if use_inter:
                updated, _, _ = inter_swap_operator(
                    updated,
                    steps=None,
                    mode='first',
                    uplift=1,
                    seed=2671 + seed_offset + idx,
                    verbose=False
                )
            refined.append(updated)

        
        return refined


    
    if initial is None:
        initial = PartialSolutionSwarm(
            solutions=[PartialSolution(problem=problem, routes=[])]
        )
    beam = initial.parsol_list
    depth = initial.parsol_list[0].n_actions  
    iterations = 0

    assert all(ps.n_actions == depth for ps in beam), \
        "All initial partial solutions must have the same action count."


    
    while beam:
        
        if all(ps.is_complete() for ps in beam):
            break

        
        iterations += 1

        
        use_intra_phase = (
            intra_frequency is not None and
            depth >= until_intra_depth and
            (depth - until_intra_depth) % intra_frequency == 0
        )
        use_inter_phase = (
            inter_frequency is not None and
            depth >= until_inter_depth and
            (depth - until_inter_depth) % inter_frequency == 0
        )

        
        if use_intra_phase or use_inter_phase:
            beam = apply_local_refinements(
                beam,
                use_intra_phase,
                use_inter_phase,
                seed_offset=depth
            )

        
        diversity_relaxed = False
        beam.sort(key=cost_function)
        next_beam: List[Tuple[float, PartialSolution]] = []

        
        
        def _insert_candidate(cost: float, parsol: PartialSolution) -> None:
            
            if any(parsol.is_identical(other) for _, other in next_beam):
                return

            insert_idx = len(next_beam)
            while insert_idx > 0 and cost < next_beam[insert_idx - 1][0]:
                insert_idx -= 1
            next_beam.insert(insert_idx, (cost, parsol))
            if len(next_beam) > l_width:
                next_beam.pop()


        
        for ps in beam:
            if ps.n_actions != depth:
                continue

            
            for cand in expand(ps)[:min(5, l_width)]:
                candidate_cost = cost_function(cand)
                _insert_candidate(candidate_cost, cand)


        
        if not next_beam:
            raise RuntimeError("Beam search stalled: no candidates generated.")

        
        beam = [item[1] for item in next_beam]
        depth += 1

        
        
        
        
        
        
        
        

        
        if verbose:
            if diversity_relaxed:
                print(
                    f"[BeamSearch] Depth {depth}. Diversity relaxed due to empty beam."
                )
            max_costs = [ps.max_cost for ps in beam]
            print(
                f"[BeamSearch] Depth {depth}. Max_cost range: "
                f"{min(max_costs)} - {max(max_costs)}. "
                f"Avg max_cost: {sum(max_costs) / len(max_costs):.1f}"
            )

    
    beam.sort(key=cost_function)
    swarm = PartialSolutionSwarm(solutions=beam)
    search_info = {
        "iterations": iterations,
        "time": time.time() - start,
    }

    
    if verbose:
        print(f"[BeamSearch] Completed. Final beam size {len(beam)}")
        print(f"[BeamSearch] Beam max_cost range: {swarm.min_cost} - {swarm.max_cost}")
        print(f"[BeamSearch] Avg max_cost: {swarm.avg_cost}")
        print(f"[BeamSearch] Time taken: {search_info['time']:.4f} seconds")

    return swarm, search_info



def beam_search_solver(
        problem: ShareARideProblem,
        cost_function: Any = parsol_scorer,
        initial: Optional[PartialSolutionSwarm] = None,
        l_width: int = 10,
        r_intra: float = 0.75,
        r_inter: float = 0.90,
        f_intra: float = 0.05,
        f_inter: float = 0.10,
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    
    solswarm, msg = beam_search_swarm_solver(
        problem, cost_function, initial, l_width,
        r_intra, r_inter, f_intra, f_inter, verbose
    )

    best_sol = solswarm.extract_best_solution()
    if verbose and best_sol:
        print(f"[BeamSearch] Best solution max_cost: {best_sol.max_cost}")

    return best_sol, msg

import heapq
from dataclasses import dataclass, field
from itertools import count
from typing import Callable, Dict, List, Optional, Tuple


Action = Tuple[int, str, int, int]
ValueFunction = Callable[[PartialSolution], float]
SelectionPolicy = Callable[[PartialSolution, List[Action]], Optional[Action]]
SimulationPolicy = Callable[[PartialSolution], Optional[PartialSolution]]
DefensePolicy = Callable[[PartialSolution], Optional[Solution]]

FAILED_ROLLOUT_COST = 10**12  


def _enumerate_actions_greedily(partial: PartialSolution, width: Optional[int]) -> List[Action]:
    
    assert width is not None and width > 0, "Width must be a positive integer"

    
    available_taxis = [
        idx for idx, state in enumerate(partial.route_states)
        if not state["ended"]
    ]
    if not available_taxis:
        return []
    taxi_order = sorted(
        available_taxis,
        key=lambda idx: partial.route_states[idx]["cost"],
    )
    problem = partial.problem
    taxi_considered = min(
        2 if problem.K >= 50
        else 3 if problem.K >= 20
        else 4,
        len(taxi_order),
    )
    taxi_branches = taxi_order[:taxi_considered]


    
    
    total_actions = max(1, 2 * (problem.N + problem.M) + problem.K)
    closing_depth = max(0, total_actions - 2 * problem.K)


    
    actions: List[Action] = []
    for t_idx in taxi_branches:
        
        state = partial.route_states[t_idx]
        potential_actions = partial.possible_actions(t_idx)
        can_return = (
            state["passenger"] == 0 and
            not state["parcels"] and
            not state["ended"] and
            state["pos"] != 0
        )

        
        if potential_actions:
            
            action_limit = min(
                1 if problem.num_nodes >= 500
                else 2 if problem.num_nodes >= 200
                else 4,
                len(potential_actions),
            )
            limited_actions = sorted(
                potential_actions, key=lambda item: item[2]
            )[:action_limit]

            
            for kind, node_idx, inc in limited_actions:
                actions.append((t_idx, kind, node_idx, inc))


        
        if can_return and partial.n_actions >= closing_depth:
            inc_back = problem.D[state["pos"]][0]
            actions.append((t_idx, "return", 0, inc_back))

    
    filtered_actions = [
        a for a in actions
        if not (a[1] == "return" and partial.n_actions < closing_depth)
    ]

    
    final_actions = sorted(filtered_actions, key=lambda item: item[3])[:width]

    return final_actions



def _apply_action(partial: PartialSolution, action: Action) -> None:
    
    taxi, kind, node_idx, inc = action
    if kind == "return":
        partial.apply_return_to_depot(taxi)
    else:
        partial.apply_action(taxi, kind, node_idx, inc)





@dataclass
class RewardFunction:
    
    visits: int = 0
    min_value: float = float("inf")
    max_value: float = float("-inf")


    def update(self, value: float) -> None:
        
        if not math.isfinite(value):
            return
        self.visits += 1
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)


    def reward_from_value(self, value: float) -> float:
        
        if not math.isfinite(value):
            return 0.0
        if self.visits == 0:
            return 0.5
        if self.max_value == self.min_value:
            return 0.5
        span = self.max_value - self.min_value
        norm = (value - self.min_value) / span
        return max(0.0, min(1.0, norm))



@dataclass
class MCTSNode:
    
    partial: PartialSolution  
    parent: Optional["MCTSNode"] = None  
    action: Optional[Action] = None  
    width: Optional[int] = None  
    children: List["MCTSNode"] = field(default_factory=list)    
    visits: int = 0  
    total_cost: int = 0     
    total_reward: float = 0.0  
    untried_actions: List[Action] = field(default_factory=list)  


    def __post_init__(self) -> None:
        self.untried_actions = _enumerate_actions_greedily(self.partial, self.width)


    @property
    def is_terminal(self) -> bool:
        
        return self.partial.is_complete()


    @property
    def average_reward(self) -> float:
        
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits


    @property
    def average_cost(self) -> float:
        
        if self.visits == 0:
            return 0.0
        return self.total_cost / self.visits


    def uct_score(self, uct_c: float) -> float:
        
        if self.visits == 0:
            return float("inf")

        exploit_term = self.average_reward

        parent_visits = self.parent.visits if self.parent else self.visits
        explore_term = uct_c * math.sqrt(
            math.log(parent_visits + 1) / self.visits
        )

        return exploit_term + explore_term



def _select(root: MCTSNode, exploration: float) -> List[MCTSNode]:
    
    path = [root]
    current = root
    while True:
        if current.untried_actions:
            return path
        if not current.children:
            return path
        current = max(current.children, key=lambda child: child.uct_score(exploration))
        path.append(current)



def _expand(
    node: MCTSNode,
    selection_policy: SelectionPolicy,
    width: Optional[int]
) -> Optional[MCTSNode]:
    
    
    if not node.untried_actions:
        node.untried_actions = _enumerate_actions_greedily(node.partial, width)

    
    if not node.untried_actions:
        return None

    
    action = selection_policy(node.partial, node.untried_actions)
    if action is None:
        return None

    
    try:
        node.untried_actions.remove(action)
    except ValueError:
        pass

    
    child_partial = node.partial.copy()
    _apply_action(child_partial, action)

    
    child = MCTSNode(child_partial, parent=node, action=action, width=width)
    node.children.append(child)

    
    return child



def _backpropagate(path: List[MCTSNode], cost: int, reward: float) -> None:
    
    for node in reversed(path):
        node.visits += 1
        node.total_reward += reward
        node.total_cost += cost


def _gather_leaves(
    node: MCTSNode,
    value_function: ValueFunction,
    limit: Optional[int] = None,
) -> List[MCTSNode]:
    
    if limit is None:
        limit = 10 ** 6  

    assert limit is not None and limit > 0, "Limit must be positive"

    
    heap: List[Tuple[float, int, MCTSNode]] = []
    ticket = count()

    
    def _collect_limited(current: MCTSNode) -> None:
        
        if not current.children:
            score = value_function(current.partial)
            entry = (score, next(ticket), current)
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)
            return

        
        for child in current.children:
            _collect_limited(child)

    _collect_limited(node)

    
    ordered = sorted(heap, key=lambda item: item[0], reverse=True)
    return [item[2] for item in ordered]



def _run_mcts(
    problem: ShareARideProblem,
    partial: Optional[PartialSolution],

    value_function: ValueFunction,
    selection_policy: SelectionPolicy,
    simulation_policy: SimulationPolicy,

    width: Optional[int],
    uct_c: float,
    max_iters: Optional[int],

    seed: Optional[int],
    time_limit: Optional[float],
    verbose: bool,
) -> Tuple[MCTSNode, Optional[PartialSolution], Dict[str, float]]:
    
    start = time.time()
    reward_function = RewardFunction()

    if seed is not None:
        random.seed(seed)
    if partial is None:
        partial = PartialSolution(problem=problem, routes=[])

    
    base_partial = partial or PartialSolution(problem=problem, routes=[])
    root = MCTSNode(base_partial, width=width)

    
    iterations = 0
    best_rollout: Optional[PartialSolution] = None
    best_rollout_cost = 10 ** 9
    max_depth = 0


    
    while True:
        
        if max_iters is not None and iterations >= max_iters:
            if verbose:
                print(f"[MCTS] Reached max iterations: {max_iters}")
            break
        if time_limit is not None and (time.time() - start) >= time_limit:
            if verbose:
                print(f"[MCTS] Reached time limit: {time_limit:.2f}s")
            break

        
        path = _select(root, uct_c)
        leaf = path[-1]
        current_depth = len(path) - 1
        if current_depth > max_depth:
            max_depth = current_depth

        
        if not leaf.is_terminal:
            child = _expand(leaf, selection_policy, width)
            if child is not None:
                path.append(child)
                working = child
            else:
                working = leaf
        else:
            break   

        
        rollout_result = simulation_policy(working.partial.copy())
        
        if rollout_result and rollout_result.is_complete():
            rollout_cost = rollout_result.max_cost  
            if rollout_cost < best_rollout_cost:
                best_rollout_cost = rollout_cost
                best_rollout = rollout_result

            
            
            value = float(value_function(rollout_result))

            
            reward_function.update(value)
            reward = reward_function.reward_from_value(value)

        
        else:
            rollout_cost = FAILED_ROLLOUT_COST  
            value = -float(rollout_cost)
            reward_function.update(value)
            reward = reward_function.reward_from_value(value)

        
        _backpropagate(path, rollout_cost, reward)
        iterations += 1

        
        if verbose and (iterations % 1000 == 0):
            elapsed = time.time() - start
            print(
                f"[MCTS] [Iteration {iterations}] "
                f"Best rollout cost={best_rollout_cost:.3f} "
                f"MaxDepth={max_depth} Time={elapsed:.2f}s"
            )

    


    
    info = {
        "iterations": iterations,
        "time": time.time() - start,
        "best_rollout_cost": best_rollout_cost,
    }

    
    if verbose:
        print(
            f"[MCTS] Iterations count={iterations} "
            f"Time={info['time']:.3f}s. Best rollout cost={best_rollout_cost:.3f}"
        )

    return root, best_rollout, info



def mcts_enumerator(
    problem: ShareARideProblem,
    partial: Optional[PartialSolution],

    
    value_function: ValueFunction,
    selection_policy: SelectionPolicy,
    simulation_policy: SimulationPolicy,

    
    best_k: int = 5,
    width: Optional[int] = 5,
    uct_c: float = math.sqrt(2),
    max_iters: Optional[int] = 500,

    
    seed: Optional[int] = None,
    time_limit: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[List[PartialSolution], Dict[str, float]]:
    

    tree, _, info = _run_mcts(
        problem,
        partial,
        width=width,
        uct_c=uct_c,
        max_iters=max_iters,
        value_function=value_function,
        selection_policy=selection_policy,
        simulation_policy=simulation_policy,
        time_limit=time_limit,
        seed=seed,
        verbose=verbose,
    )

    top_leaves = _gather_leaves(
        tree,
        value_function=value_function,
        limit=max(1, best_k),
    )
    
    top = [leaf.partial.copy() for leaf in top_leaves]

    return top, info



def mcts_solver(
    problem: ShareARideProblem,
    partial: Optional[PartialSolution],

    value_function: ValueFunction,
    selection_policy: SelectionPolicy,
    simulation_policy: SimulationPolicy,
    defense_policy: DefensePolicy,

    width: Optional[int] = 5,
    uct_c: float = math.sqrt(2),
    max_iters: Optional[int] = 1000,

    seed: Optional[int] = None,
    time_limit: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[Optional[Solution], Dict[str, float]]:
    
    start = time.time()

    
    _, best_rollout, info = _run_mcts(
        problem=problem,
        partial=partial,
        value_function=value_function,
        selection_policy=selection_policy,
        simulation_policy=simulation_policy,
        width=width,
        uct_c=uct_c,
        max_iters=max_iters,
        seed=seed,
        time_limit=time_limit,
        verbose=verbose,
    )

    
    if best_rollout is None:
        info["used_best_rollout"] = False
        info["final_value"] = float("nan")
        return None, info

    
    sol = best_rollout.to_solution()
    assert sol is not None and sol.is_valid(), "Best rollout is not a valid solution."

    
    info["used_best_rollout"] = True
    info["iterations"] = info.get("iterations", 0)
    info["time"] = time.time() - start

    return sol, info







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
    prob: ShareARideProblem = read_instance()


    
    def v_func(
        parsol: PartialSolution,
    ) -> float:
        
        cost = parsol_scorer(parsol)
        return -cost    


    def stochastic_select_policy(
        _ps: PartialSolution,
        actions: List[Action],
    ) -> Optional[Action]:
        
        rng = random.Random()
        if not actions:
            return None
        increments = [float(action[3]) for action in actions]
        weights = softmax_weighter(increments, T=0.1)
        chosen_idx = sample_from_weight(rng, weights)
        return actions[chosen_idx]


    def sim_policy(
        ps: PartialSolution,
    ) -> Optional[PartialSolution]:
        
        sim_solution, _ = greedy_balanced_solver(
            ps.problem,
            premature_routes=[r.copy() for r in ps.routes],
            verbose=False,
        )
        return ps if sim_solution is None else PartialSolution.from_solution(sim_solution)


    def def_policy(
        ps: PartialSolution,
    ) -> Optional[Solution]:
        
        beam_solution, _ = beam_search_solver(
            ps.problem,
            cost_function=parsol_scorer,
            initial=PartialSolutionSwarm([ps]),
        )

        return beam_solution


    sol, _ = mcts_solver(
        problem=prob,
        partial=None,
        value_function=v_func,
        selection_policy=stochastic_select_policy,
        simulation_policy=sim_policy,
        defense_policy=def_policy,

        width=3,
        uct_c=5.0,
        max_iters=100000,

        seed=42,
        time_limit=200.0,
        verbose=verbose,
    )

    assert sol, "No solution found by MCTS."
    if verbose:
        print()
        print(f"Cost after MCTS: {sol.max_cost:.2f}")
        print("===============================")


    
    st1 = time.time()
    par = PartialSolution.from_solution(sol)
    new_par, modified, n_relocates = relocate_operator(
        partial=par,
        steps=None,
        mode='first',
        seed=111,
        verbose=verbose
    )
    sol = new_par.to_solution()
    assert sol, "No solution found after relocate."
    if verbose:
        print()
        print(f"Total relocate performed: {n_relocates}")
        print(f"Cost after relocate: {sol.max_cost:.2f}")
        print(f"Time for relocate: {time.time() - st1:.2f} seconds")
        print("===============================")

    sol.stdin_print(verbose=verbose)


if __name__ == "__main__":
    main(verbose=False)


