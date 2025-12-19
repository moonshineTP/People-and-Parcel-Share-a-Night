
import sys
import random
import time
import math
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass

Action = Tuple[int, str, int, int]

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
        self.num_actions = 2 * N + 2 * M + K    

        
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
        
        if len(self.q) != self.M:
            return False
        if len(self.Q) != self.K:
            return False
        if len(self.D) != self.num_nodes:
            return False
        if not all(len(row) == self.num_nodes for row in self.D):
            return False
        if self.coords is not None and len(self.coords) != self.num_nodes:
            return False

        return True


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
        self.num_actions = problem.num_actions
        self.max_cost = max(new_route_costs) if new_route_costs else 0


    def is_valid(self) -> bool:
        

        prob = self.problem
        K = prob.K      
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

                    
                    
                    parcel_load -= prob.q[jd - 1]
                    parcel_onboard.remove(jd)

            
            if passenger_onboard:
                return False
            if parcel_load != 0:
                return False

        return True


    def stdin_print(self, verbose: bool = False):
        
        

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

        self.num_actions = sum(len(route) - 1 for route in self.routes)


    def _init_routes(self, routes):
        K = self.problem.K      

        
        if not routes:
            return [[0] for _ in range(K)]
        if len(routes) != K:
            raise ValueError(f"Expected {K} routes, got {len(routes)}.")
        for route in routes:
            if route is None:
                raise ValueError("One route cannot be null")
            elif not route or route[0] != 0:
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
                "parcels": onboard_parcels,
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
        if self.num_actions != total_actions:
            return False

        return True


    def is_identical(self, other: "PartialSolution") -> bool:
        
        if self is other:
            return True

        
        if self.problem is not other.problem:
            return False
        if self.num_actions != other.num_actions:
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


    def check_action(self, t_idx: int, kind: str, node_idx: int) -> bool:
        
        state = self.route_states[t_idx]
        prob = self.problem

        
        if state["ended"]:
            return False

        
        if kind == "pickP":
            return state["passenger"] == 0 and node_idx in self.remaining_pass_pick
        if kind == "dropP":
            return state["passenger"] == node_idx
        if kind == "pickL":
            return node_idx in self.remaining_parc_pick \
                and state["load"] + prob.q[node_idx - 1] <= prob.Q[t_idx]
        if kind == "dropL":
            return node_idx in state["parcels"]

        
        raise ValueError(f"Unknown action kind: {kind}")


    def check_return(self, t_idx: int) -> bool:
        
        state = self.route_states[t_idx]

        return not (state["ended"] or state["passenger"] != 0 or state["parcels"])


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
        self.num_actions += 1


    def apply_return(self, t_idx: int) -> None:
        
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
        self.num_actions += 1


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
        self.num_actions -= 1


    def is_complete(self) -> bool:
        
        return self.num_actions == self.problem.num_actions \
            and all(state["ended"] for state in self.route_states)
        


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
            solutions: Optional[List[PartialSolution]],
        ):
        

        if not solutions:
            raise ValueError("Solutions list cannot be empty.")

        self.problem = solutions[0].problem
        self.num_partials = len(solutions)
        self.partial_lists = solutions
        self.partial_num_actions = [sol.num_actions for sol in solutions]
        self.costs = [sol.max_cost for sol in solutions]
        self.min_cost = min(self.costs)
        self.max_cost = max(self.costs)
        self.avg_cost = sum(sol.max_cost for sol in solutions) / len(solutions)
        self.best_partial = min(solutions, key=lambda s: s.max_cost)


    def apply_action_one(
        self, sol_idx: int, t_idx: int, kind: str, node_idx: int, inc: int
    ):
        
        partials = self.partial_lists[sol_idx]
        partials.apply_action(t_idx, kind, node_idx, inc)

        
        self.partial_num_actions[sol_idx] = partials.num_actions
        self.costs[sol_idx] = partials.max_cost

        self.min_cost = min(self.costs)
        self.max_cost = max(self.costs)
        self.avg_cost = sum(self.costs) / len(self.costs)
        if partials.max_cost == self.min_cost:
            self.best_partial = partials


    def apply_return_to_depot_one(self, sol_idx: int, t_idx: int):
        
        parsol = self.partial_lists[sol_idx]
        parsol.apply_return(t_idx)

        
        self.partial_num_actions[sol_idx] = parsol.num_actions
        self.costs[sol_idx] = parsol.max_cost

        self.min_cost = min(self.costs)
        self.max_cost = max(self.costs)
        self.avg_cost = sum(self.costs) / len(self.costs)
        if parsol.max_cost == self.min_cost:
            self.best_partial = parsol


    def copy(self):
        
        copied_solutions = [sol.copy() for sol in self.partial_lists]
        return PartialSolutionSwarm(solutions=copied_solutions)


    def opt(self) -> Optional[Solution]:
        
        opt_cost = 10**18
        opt_sol = None
        for par in self.partial_lists:
            if par.is_complete():
                sol = par.to_solution()
                if sol and sol.max_cost < opt_cost:
                    opt_cost = sol.max_cost
                    opt_sol = sol

        return opt_sol




from typing import Sequence

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




def repair_one_route(
        partial: PartialSolution,
        route_idx: int,
        steps: int,
        T: float = 1.0,
        seed: Optional[int] = 42,
        verbose: bool = False
    ) -> Tuple[PartialSolution, List[bool], int]:
    

    
    

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
            partial.apply_return(route_idx)
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


def repair_operator(
    partial: PartialSolution,
    repair_proba: float,
    steps: int,
    T: float = 1.0,
    seed: Optional[int] = 42,
    verbose: bool = False
) -> Tuple[PartialSolution, List[bool], int]:
    

    
    rng = random.Random(seed)

    
    routes = [r_idx for r_idx in range(partial.problem.K)]
    num_routes = partial.problem.K
    approx_repair_count = round(repair_proba * num_routes + 0.5)
    repair_count = min(num_routes, max(1, approx_repair_count))

    
    selected_routes = rng.sample(routes, repair_count)
    total_added_nodes = 0
    modified = [False] * num_routes
    for r_idx in selected_routes:
        partial, modified, added_nodes = repair_one_route(
            partial=partial,
            route_idx=r_idx,
            steps=steps,
            T=T,
            seed=rng.randint(0, 1_000_000),
            verbose=verbose
        )

        total_added_nodes += added_nodes
        modified[r_idx] = True

        if verbose:
            print(f"[Repair]: Repairing route {r_idx} with up to {steps} steps.")


    
    if verbose:
        print()
        print("[Repair] Operator completed.")
        print(f"Total routes repaired: {repair_count};")
        print(f"Total nodes added: {total_added_nodes}.")
        print("------------------------------")
        print()

    return partial, modified, total_added_nodes






from typing import List, Tuple, Optional






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
        print(f"[Destroy] Route {route_idx}: removed last {remove_len} nodes.")

    return destroyed_route


def destroy_operator(
        sol: Solution,
        destroy_proba: float,
        destroy_steps: int,
        seed: Optional[int] = None,
        t: float = 1.0,
        verbose: bool = False
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


    
    
    weights = softmax_weighter(costs, t=t)

    
    selected_ids: List[int] = []
    available_ids = list(range(sol.problem.K))
    available_weights = weights[:]
    for _ in range(destroyed_count):
        if not available_ids:
            break

        
        selected_idx = sample_from_weight(rng, available_weights)
        selected_ids.append(available_ids[selected_idx])

        
        available_ids.pop(selected_idx)
        available_weights.pop(selected_idx)


    
    for idx in selected_ids:
        route = routes[idx]

        
        if len(route) <= 2:
            continue

        
        reduced = destroy_one_route(
            route, idx, steps=destroy_steps, verbose=verbose
        )
        removed = max(0, len(route) - len(reduced))

        
        if removed > 0:
            routes[idx] = reduced
            flags[idx] = True
            num_removed += removed

    partial_sol = PartialSolution(problem=sol.problem, routes=routes)

    
    if verbose:
        print()
        print("[Destroy] Operation complete.")
        print(f"[Destroy] Destroyed {len(selected_ids)} routes, removed {num_removed} nodes total.")
        print("------------------------------")
        print()

    return partial_sol, flags, num_removed





def greedy_balanced_solver(
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    

    if partial is None:
        partial = PartialSolution(problem=problem, routes=[])

    start_time = time.time()
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
            partial.apply_return(argmin_t_idx)
            continue

        
        kind, idx, inc = min(actions, key=lambda x: x[2])
        partial.apply_action(argmin_t_idx, kind, idx, inc)

        if verbose:
            print(f"[Greedy] Taxi {argmin_t_idx} extended route with {kind} {idx} (inc {inc})")


    
    for t_idx, t_state in enumerate(taxi_states):
        if not t_state["ended"]:
            partial.apply_return(t_idx)

    
    sol = partial.to_solution()

    
    elapsed = time.time() - start_time
    info = {
        "iterations": stats["iterations"],
        "actions_evaluated": stats["actions_evaluated"],
        "time": elapsed
    }

    
    if sol and not sol.is_valid():
        sol = None
    

    
    if verbose:
        print("[Greedy] All tasks completed.")
        print(f"[Greedy] Solution max cost: {sol.max_cost if sol else 'N/A'}")
        print(f"[Greedy] Time taken: {elapsed:.4f} seconds")

    return sol, info


def iterative_greedy_balanced_solver(
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        iterations: int = 10,
        destroy_proba: float = 0.4,
        destroy_steps: int = 15,
        destroy_t: float = 1.0,
        rebuild_proba: float = 0.3,
        rebuild_steps: int = 5,
        rebuild_t: float = 1.0,
        time_limit: float = 10.0,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    

    
    
    

    if partial is None:
        partial = PartialSolution(problem=problem, routes=[])

    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + time_limit if time_limit is not None else None

    
    best_sol, base_info = greedy_balanced_solver(problem, partial=partial, verbose=False)
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


        
        destroy_seed = None if seed is None else 2 * seed + it
        partial_sol, destroyed_flags, removed = destroy_operator(
            best_sol,
            destroy_proba,
            destroy_steps,
            seed=destroy_seed,
            t=destroy_t
        )
        nodes_destroyed += removed


        
        for r_idx, was_destroyed in enumerate(destroyed_flags):
            if not was_destroyed or len(partial_sol.routes[r_idx]) <= 2:
                continue
            if rng.random() > rebuild_proba:
                continue

            partial_sol, _, new_nodes_count = repair_one_route(
                partial_sol,
                route_idx=r_idx,
                steps=rebuild_steps,
                T=rebuild_t,
                seed=(destroy_seed + r_idx) if destroy_seed is not None else None,
                verbose=False
            )
            nodes_rebuilt += new_nodes_count


        
        sol_cand, info_cand = greedy_balanced_solver(
            problem,
            partial=partial_sol,
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



def balanced_scorer(
        parsol: Union[PartialSolution, Solution],
        sample_size: int = 15,
        w_std: float = 0.15,
        seed: Optional[int] = None,
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


def check_general_action(partial: PartialSolution, action: Action) -> bool:
    
    taxi, kind, node_idx, _inc = action
    if kind == "return":
        return partial.check_return(taxi)
    return partial.check_action(taxi, kind, node_idx)


def apply_general_action(partial: PartialSolution, action: Action) -> None:
    
    taxi, kind, node_idx, inc = action
    if kind == "return":
        partial.apply_return(taxi)
    else:
        partial.apply_action(taxi, kind, node_idx, inc)


def enumerate_actions_greedily(
        partial: PartialSolution,
        width: Optional[int] = None,
        assymetric: bool = True,
    ) -> List[Action]:
    
    if width is None:
        width = 10**9    

    problem = partial.problem


    
    
    active_taxis = [
        idx for idx, state in enumerate(partial.route_states)
        if not state["ended"]
    ]
    if not active_taxis:
        return []

    taxi_order = sorted(active_taxis, key=lambda idx: partial.route_states[idx]["cost"])
    num_taxis = len(taxi_order)


    
    if assymetric:
        seen_configs: set = set()
        filtered: List[int] = []
        for t_idx in taxi_order:
            route_config = tuple(partial.route_states[t_idx]["route"])
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
            
            assigned_actions = partial.possible_actions(t_idx)
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
    prioritized: List[Action] = []
    secondary: List[Action] = []
    for t_idx, kind, node_idx, inc in expansions:
        if partial.route_costs[t_idx] + inc <= current_max:
            prioritized.append((t_idx, kind, node_idx, inc))
        else:
            secondary.append((t_idx, kind, node_idx, inc))

    prioritized.sort(key=lambda item: item[3])
    secondary.sort(key=lambda item: item[3])


    
    final_actions: List[Action] = prioritized + secondary

    
    if not final_actions:
        
        if partial.num_actions < problem.num_nodes - 1:
            print("[Warning] No feasible actions found before closing depth")
            raise RuntimeError("Premature routes not covering all nodes.")

        
        return_actions: List[Action] = []
        for t_idx in taxi_order:
            state = partial.route_states[t_idx]
            if state["pos"] == 0:
                continue
            if partial.check_return(t_idx):
                inc_back = problem.D[state["pos"]][0]
                return_actions.append((t_idx, "return", 0, inc_back))

        return return_actions[:width]

    
    return final_actions








from typing import Iterator

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
        seed: Optional[int] = None, 
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
                print(f"[Relocate] [{from_route_idx}->{to_route_idx}] moved request "
                      f"(P:{pf},D:{qf}) to ({pt},{qt}). Decrement={dec}"
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
        seed: Optional[int] = None,         
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

    
    if verbose:
        print()
        print("[Relocate] Operator completed. ")
        print(f"Total relocations = {total_moves}; ")
        print(f"Decrement = {partial.max_cost - current_par.max_cost}; ")
        print(f"New max cost = {current_par.max_cost}.")
        print("------------------------------")
        print()

    return current_par, modified_total, total_moves


DefensePolicy = Callable[PartialSolution, Optional[Solution]]
FinalizePolicy = Callable[PartialSolution, Optional[Solution]]





def _default_defense_policy(
        partial: PartialSolution,
        seed: Optional[int] = None,     
    ) -> Optional[Solution]:
    
    sol, _info = sol, _info = greedy_balanced_solver(partial.problem, partial, False)

    return sol


def _default_finalize_policy(
        partial: PartialSolution,
        seed: Optional[int] = None,
    ) -> Optional[Solution]:

    sol, _info = iterative_greedy_balanced_solver(
        partial.problem,
        partial,
        iterations=5000,
        time_limit=5.0,
        seed=seed,
        verbose=False
    )

    if not sol:
        return None

    raw_sol = PartialSolution.from_solution(sol)
    refined_sol, _modified, _cnt = relocate_operator(
        partial=raw_sol,
        seed=seed,
        verbose=False
    )

    return refined_sol.to_solution()





@dataclass
class SolutionTracker:
    
    best_solution: Optional[Solution] = None
    best_cost: int = 10**18
    worst_cost: int = -1
    total_cost: int = 0
    count: int = 0


    def update(
            self,
            source: Union[Solution, PartialSolutionSwarm, List[Optional[Solution]]]
        ) -> None:
        
        if isinstance(source, Solution):
            self._update_from_solution(source)
        elif isinstance(source, PartialSolutionSwarm):
            self._update_from_swarm(source)
        elif isinstance(source, list):
            self._update_from_list(source)


    def _update_from_solution(self, solution: Solution) -> None:
        
        cost = solution.max_cost

        self.count += 1
        self.total_cost += cost
        self.worst_cost = max(self.worst_cost, cost)

        if cost < self.best_cost:
            self.best_cost = cost
            self.best_solution = solution


    def _update_from_swarm(self, swarm: PartialSolutionSwarm) -> None:
        
        for partial in swarm.partial_lists:
            if not partial.is_complete():
                continue

            sol = partial.to_solution()
            if not sol:
                continue

            self._update_from_solution(sol)


    def _update_from_list(self, solutions: List[Optional[Solution]]) -> None:
        
        for sol in solutions:
            if sol is None:
                continue

            self._update_from_solution(sol)


    def stats(self) -> Dict[str, float]:
        
        avg_cost = self.total_cost / self.count if self.count > 0 else 0.0
        return {
            "best_cost": self.best_cost,
            "worst_cost": self.worst_cost,
            "avg_cost": avg_cost,
            "count": self.count,
        }


    def opt(self) -> Optional[Solution]:
        
        return self.best_solution


class PheromoneMatrix:
    

    def __init__(
            self,
            problem: ShareARideProblem,
            sigma: int,
            rho: float,
            init_cost: int
        ) -> None:

        

        self.size = problem.num_nodes
        self.sigma = sigma
        self.rho = rho
        self.tau_0 = 1.0 / (rho * init_cost)
        self.tau_max = 2 * self.tau_0
        self.tau_min = self.tau_0 / 10.0
        self.tau = [
            [self.tau_0 for _ in range(self.size)]
            for _ in range(self.size)
        ]

        
        


    def _clamp(self, phe: float) -> float:
        
        return min(self.tau_max, max(self.tau_min, phe))


    def get(self, i: int, j: int) -> float:
        
        return self.tau[i][j]


    def set(self, i: int, j: int, phe: float) -> None:
        
        self.tau[i][j] = self._clamp(phe)


    def update(
            self,
            swarm: PartialSolutionSwarm,
            opt: Optional[Solution],
        ) -> None:
        
        
        def extract_edges(partial: PartialSolution) -> List[Tuple[int, int]]:
            
            edges: List[Tuple[int, int]] = []
            for route in partial.routes:
                for idx in range(len(route) - 1):
                    edges.append((route[idx], route[idx + 1]))
            return edges

        
        ranked_partials = sorted(
            [(par, float(par.max_cost)) for par in swarm.partial_lists],
            key=lambda x: x[1]
        )[:self.sigma - 1]

        
        delta: List[List[float]] = [
            [0.0 for _ in range(self.size)]
            for _ in range(self.size)
        ]

        
        for rank, (partial, cost) in enumerate(ranked_partials, start=1):
            weight = (self.sigma - rank) / cost

            
            for (i, j) in extract_edges(partial):
                if 0 <= i < self.size and 0 <= j < self.size:
                    delta[i][j] += weight

        
        if opt is not None and opt.max_cost > 0:
            best_weight = self.sigma / opt.max_cost

            
            best_partial = PartialSolution.from_solution(opt)

            
            for (i, j) in extract_edges(best_partial):
                if 0 <= i < self.size and 0 <= j < self.size:
                    delta[i][j] += best_weight


        
        for i in range(self.size):
            for j in range(self.size):
                new_phe = self.rho * self.tau[i][j] + delta[i][j]
                self.tau[i][j] = self._clamp(new_phe)


class DesirabilityMatrix:
    

    def __init__(
            self,
            problem: ShareARideProblem,
            phi: float,
            chi: float,
            gamma: float,
            kappa: float
        ) -> None:
        
        self.size = problem.num_nodes
        self.problem = problem
        self.phi = phi
        self.chi = chi
        self.gamma = gamma
        self.kappa = kappa

        
        self.eta_dist: List[List[float]] = []
        D = self.problem.D      
        for i in range(self.size):
            row = []

            for j in range(self.size):
                if i == j:
                    row.append(0)   
                else:
                    
                    slack_ij = max(D[0][i] + D[j][0] - D[i][j], 0)
                    
                    saving_term = (1 + slack_ij) ** self.phi
                    
                    distance_term = (1 + D[i][j]) ** self.chi
                    
                    eta_dist_ij = saving_term / distance_term

                    row.append(eta_dist_ij)

            self.eta_dist.append(row)


    def get(self, i: int, j: int, partial: PartialSolution, action: Action) -> float:
        
        
        taxi_idx, kind, node, _inc = action

        
        Q = self.problem.Q[taxi_idx]        
        state = partial.route_states[taxi_idx]
        new_cap = state["load"]
        if kind == "pickL":
            new_cap += self.problem.q[node - 1]
        if kind == "dropL":
            new_cap -= self.problem.q[node - 1]

        people_term = 2 - int(kind == "pickP")
        parcel_term = (1 + self.gamma * (Q - new_cap) / Q) * self.kappa

        return self.eta_dist[i][j] * people_term * parcel_term


class NearestExpansionCache:
    

    def __init__(
            self,
            problem: ShareARideProblem,
            num_nearest: int = 3
        ) -> None:
        

        self.nearest_actions: List[List[Tuple[str, int, int]]] = []
        for node_idx in range(problem.num_nodes):
            
            
            if node_idx == 0:
                routes = [[0] for _ in range(problem.K)]
            else:
                routes = [[0, node_idx]] + [[0] for _ in range(problem.K - 1)]

            partial = PartialSolution(problem, routes=routes)
            t_acts = partial.possible_actions(0)
            t_acts = sorted(t_acts, key=lambda item: item[2])[:num_nearest]
            self.nearest_actions.append(t_acts)


    def query(self, partial: PartialSolution, num_queried: int) -> List[Action]:
        
        current_max = partial.max_cost
        prioritized: List[Action] = []
        secondary: List[Action] = []

        
        for taxi_idx, state in enumerate(partial.route_states):
            if state["ended"]:
                continue
            pos = state["pos"]
            cached: List[Tuple[str, int, int]] = self.nearest_actions[pos]

            
            for unassigned_action in cached:
                
                kind, node_idx, inc = unassigned_action

                
                if not partial.check_action(taxi_idx, kind, node_idx):
                    continue

                
                action: Action = (taxi_idx, kind, node_idx, inc)

                
                if partial.route_costs[taxi_idx] + inc <= current_max:
                    prioritized.append(action)
                else:
                    secondary.append(action)


        return (prioritized + secondary)[:num_queried]


class Ant:
    

    class ProbaExpandSampler:
        
        partial: PartialSolution
        cache: "NearestExpansionCache"
        alpha: float
        beta: float
        q_prob: float
        width: int

        def __init__(
                self,
                partial: PartialSolution,
                cache: "NearestExpansionCache",
                alpha: float,
                beta: float,
                q_prob: float,
                width: int
            ) -> None:
            self.partial = partial
            self.cache = cache
            self.alpha = alpha
            self.beta = beta
            self.q_prob = q_prob
            self.width = width


        def _get_to_node(
                self,
                action: Tuple[int, str, int, int]
            ) -> int:
            
            _taxi_idx, kind, req_idx, _ = action
            prob = self.partial.problem
            if kind == "pickP":
                return prob.ppick(req_idx)
            elif kind == "dropP":
                return prob.pdrop(req_idx)
            elif kind == "pickL":
                return prob.parc_pick(req_idx)
            elif kind == "dropL":
                return prob.parc_drop(req_idx)
            else:
                return 0


        def _compute_log_proba(
                self,
                tau: PheromoneMatrix,
                eta: DesirabilityMatrix,
                action: Tuple[int, str, int, int],
            ) -> float:
            
            
            taxi_idx = action[0]
            state = self.partial.route_states[taxi_idx]
            from_node = state["pos"]
            to_node = self._get_to_node(action)

            tau_val = tau.get(from_node, to_node)
            eta_val = eta.get(from_node, to_node, self.partial, action)

            
            tau_val = max(tau_val, 1e-300)
            eta_val = max(eta_val, 1e-300)

            log_proba = self.alpha * math.log(tau_val) + self.beta * math.log(eta_val)

            return log_proba


        def _collect_actions(self) -> List[Action]:
            
            partial = self.partial
            width = self.width

            
            
            actions = self.cache.query(partial, width)
            if actions:
                return actions[:width]

            
            return enumerate_actions_greedily(
                partial,
                width=width,
                assymetric=True
            )[:width]


        def sample_action(
                self,
                tau: PheromoneMatrix,
                eta: DesirabilityMatrix,
                rng: random.Random
            ) -> Optional[Action]:
            

            
            actions = self._collect_actions()
            if not actions:
                return None

            
            log_probas: List[float] = []
            for action in actions:
                log_proba = self._compute_log_proba(tau, eta, action)
                log_probas.append(log_proba)

            
            
            max_log = max(log_probas)
            exp_shifted = [math.exp(lp - max_log) for lp in log_probas]
            total = sum(exp_shifted)
            probas = [e / total for e in exp_shifted]


            
            select_idx: int
            if rng.random() < self.q_prob:
                
                
                select_idx = min(range(len(actions)), key=lambda i: actions[i][3])
            else:
                
                select_idx = sample_from_weight(rng, probas)

            return actions[select_idx]


    def __init__(
            self,
            partial: PartialSolution,
            cache: "NearestExpansionCache",
            tau: PheromoneMatrix,
            eta: DesirabilityMatrix,
            alpha: float,
            beta: float,
            q_prob: float,
            width: int,
            rng: random.Random,
        ) -> None:

        
        self.problem = partial.problem
        self.partial = partial
        self.cache = cache
        self.tau = tau
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.q_prob = q_prob
        self.width = width
        self.rng = rng

        
        self.sampler = Ant.ProbaExpandSampler(
            partial=self.partial,
            cache=cache,
            alpha=alpha,
            beta=beta,
            q_prob=q_prob,
            width=width
        )


    def expand(self) -> bool:
        
        if self.partial.is_complete():
            return False

        
        sampled_action = self.sampler.sample_action(
            self.tau, self.eta, self.rng
        )
        if not sampled_action:
            return False

        
        apply_general_action(self.partial, sampled_action)

        return True


class AntPopulation:
    

    def __init__(
            self,
            initial_swarm: PartialSolutionSwarm,

            
            cache: NearestExpansionCache,
            tau: PheromoneMatrix,
            eta: DesirabilityMatrix,
            lfunc: SolutionTracker,

            
            alpha: float,
            beta: float,
            q_prob: float,
            width: int,

            
            iterations: int,
            time_limit: float,
            seed: Optional[int],
            verbose: bool
        ) -> None:
        self.swarm = initial_swarm.copy()
        self.completed = [par.is_complete() for par in self.swarm.partial_lists]

        self.cache = cache
        self.tau = tau
        self.eta = eta
        self.lfunc = lfunc

        self.iterations = iterations
        self.time_limit = time_limit
        self.seed = seed
        self.verbose = verbose

        
        self.ants: List[Ant] = []
        for idx, partial in enumerate(self.swarm.partial_lists):
            ant = Ant(
                partial=partial,
                cache=cache,
                tau=self.tau,
                eta=self.eta,
                alpha=alpha,
                beta=beta,
                q_prob=q_prob,
                width=width,
                rng=random.Random(hash(seed + 100 * idx) if seed else None),
            )
            self.ants.append(ant)

        
        self.num_ants = len(self.ants)
        self.max_actions = initial_swarm.problem.num_actions
        self.start_time = time.time()
        self.end_time = self.start_time + self.time_limit
        self.tle = lambda: time.time() > self.end_time


    def expand(self) -> bool:
        
        is_expanded = False
        for idx, ant in enumerate(self.ants):
            if self.completed[idx]:
                continue

            if ant.expand():
                is_expanded = True
            elif self.verbose:
                print(
                    f"[ACO] [Depth {ant.partial.num_actions}] "
                    f"[Warning] Ant {idx + 1} cannot expand, "
                    "further diagnosis needed."
                )

            self.completed[idx] = ant.partial.is_complete()

        return is_expanded


    def update(self) -> None:
        
        self.lfunc.update(
            source=self.swarm,  
        )

        self.tau.update(
            swarm=self.swarm,
            opt=self.lfunc.opt(),
        )


    def run(self) -> PartialSolutionSwarm:
        
        anchors = [
            self.max_actions // 5,
            self.max_actions // 2,
            self.max_actions * 9 // 10
        ]

        
        for ite in range(self.iterations):
            if self.tle():
                if self.verbose:
                    print("[ACO] Time limit reached, skipping iteration.")
                return self.swarm

            
            if self.verbose:
                if ite in anchors or ite % 100 == 0:
                    costs = [par.max_cost for par in self.swarm.partial_lists]
                    depths = [par.num_actions for par in self.swarm.partial_lists]
                    print(
                        f"[ACO] [Iteration {ite}] "
                        f"Partial cost range: {min(costs):.3f} - {max(costs):.3f}, "
                        f"Depth range: {min(depths)} - {max(depths)}, "
                        f"Time_elapsed={time.time() - self.start_time:.2f}s."
                    )

            
            if not self.expand():
                if self.verbose:
                    print("[ACO] All ants have completed their solutions.")
                break

            self.update()


        
        if self.verbose:
            num_sol = sum(1 for par in self.swarm.partial_lists if par.is_complete())
            run_opt = self.swarm.opt()
            global_opt = self.lfunc.opt()
            print(
                f"[ACO] Finished all iterations.\n"
                f"Complete solutions found: {num_sol}/{self.num_ants}.\n"
                f"Run best cost: {run_opt.max_cost if run_opt else 'N/A'}, "
                f"Opt cost: {global_opt.max_cost if global_opt else 'N/A'}."
            )

        return self.swarm   


class SwarmTracker:
    
    def __init__(
            self,
            initial_swarm: PartialSolutionSwarm,
            defense_policy: DefensePolicy,
            finalize_policy: FinalizePolicy,
            seed: Optional[int] = None,
        ) -> None:

        
        self.seed = seed

        
        self.frontier_swarm: List[PartialSolution] = [
            partial.copy() for partial in initial_swarm.partial_lists
        ]
        self.num_partials = initial_swarm.num_partials

        
        self.frontier_potential = [
            defense_policy(partial)
            for partial in self.frontier_swarm
        ]
        self.frontier_potential_costs = [
            sol.max_cost if sol is not None else 10**18
            for sol in self.frontier_potential
        ]

        
        self.finals: List[Solution]
        self.is_finalized: bool = False

        
        self.defense_policy = defense_policy
        self.finalize_policy = finalize_policy


    def update(self, source: PartialSolutionSwarm) -> List[Optional[Solution]]:
        
        

        for idx, partial in enumerate(source.partial_lists):
            sol = self.defense_policy(
                partial, seed=self.seed + 10 * idx if self.seed else None
            )
            if not sol:
                continue

            
            cost = sol.max_cost
            if cost < self.frontier_potential_costs[idx]:
                self.frontier_swarm[idx] = partial.copy()
                self.frontier_potential[idx] = sol
                self.frontier_potential_costs[idx] = cost

        return self.frontier_potential


    def finalize(self, cutoff: Optional[int], time_limit: Optional[float]) -> List[Solution]:
        
        if time_limit is None:
            time_limit = float("inf")
        start_time = time.time()
        end_time = start_time + time_limit


        
        sorted_partials = sorted(
            zip(
                self.frontier_swarm, self.frontier_potential, self.frontier_potential_costs
            ),
            key=lambda x: x[2]
        )
        chosen_partials = sorted_partials[:cutoff] if cutoff else sorted_partials


        
        finalized: List[Solution] = []
        remaining_potentials: List[Solution] = []
        for idx, (par, _potential, _cost) in enumerate(chosen_partials):
            
            if time.time() >= end_time:
                
                for _, _pot, _ in sorted_partials[idx:]:
                    if _pot is not None:
                        remaining_potentials.append(_pot)
                break

            sol = self.finalize_policy(
                par, seed=self.seed + 20 * idx if self.seed else None
            )
            if sol:
                finalized.append(sol)


        
        finalized.extend(remaining_potentials)
        finalized.sort(key=lambda s: s.max_cost)
        finalized = finalized[:cutoff] if cutoff else finalized

        
        self.is_finalized = True
        self.finals = finalized

        
        return finalized


    def top(
            self,
            k: int,
            cutoff: Optional[int] = None,
            time_limit: Optional[float] = None
        ) -> List[Solution]:
        
        if cutoff is None:
            cutoff = k

        if not self.is_finalized:
            self.finalize(cutoff, time_limit)

        return self.finals[:k]


    def opt(
            self,
            cutoff: Optional[int] = None,
            time_limit: Optional[float] = None
        ) -> Solution:
        
        if not self.is_finalized:
            self.finalize(cutoff, time_limit)

        

        return self.finals[0]  





def _run_aco(
    problem: ShareARideProblem,
    swarm: PartialSolutionSwarm,

    
    runs: int,
    iterations: Optional[int],
    width: int,
    cutoff: Optional[int],

    
    
    q_prob: float,
    alpha: float,
    beta: float,
    
    phi: float,
    chi: float,
    gamma: float,
    kappa: float,
    
    sigma: int,
    rho: float,

    
    defense_policy: DefensePolicy,
    finalize_policy: FinalizePolicy,

    
    seed: Optional[int],
    time_limit: float,
    verbose: bool,
) -> Tuple[SwarmTracker, Dict[str, Any]]:
    
    start = time.time()
    end = start + time_limit

    if iterations is None:
        iterations = problem.num_actions


    
    if verbose:
        print("[ACO] [Init] Estimating costs from initial greedy solver...")

    init_sol, _info = greedy_balanced_solver(problem, None, False)
    
    init_cost = init_sol.max_cost

    if verbose:
        print(f"[ACO] [Init] Greedy solution cost: {init_cost:.3f}")


    
    if verbose:
        print("[ACO] [Init] Initializing nearest expansion cache...")

    cache = NearestExpansionCache(problem, num_nearest=20)


    
    if verbose:
        print("[ACO] [Init] Initializing matrices...")

    tau = PheromoneMatrix(problem, sigma=sigma, rho=rho, init_cost=init_cost)
    eta = DesirabilityMatrix(problem, phi, chi, gamma, kappa)


    
    if verbose:
        print("[ACO] [Init] Initializing trackers...")
    lfunc = SolutionTracker()
    lfunc.update(init_sol)


    
    tracker = SwarmTracker(
        initial_swarm=swarm,
        defense_policy=defense_policy,
        finalize_policy=finalize_policy,
    )


    
    runs_completed = runs
    for run in range(runs):
        if time.time() - start >= 0.8 * time_limit:
            runs_completed = run

            if verbose:
                print(f"[ACO] Time limit approaching, stopping at run {run + 1}/{runs}.")

            break

        if verbose:
            print(f"[ACO] [Run {run + 1}/{runs}] Starting the population run...")

        
        population = AntPopulation(
            initial_swarm=swarm,
            cache=cache,
            tau=tau,
            eta=eta,
            lfunc=lfunc,
            alpha=alpha,
            beta=beta,
            q_prob=q_prob,
            width=width,
            iterations=iterations,
            time_limit=time_limit,
            seed=hash(seed + 10 * run) if seed else None,
            verbose=verbose,
        )

        
        if verbose:
            print(f"[ACO] [Run {run + 1}/{runs}] Running the ant population")
        updated_swarm = population.run()

        
        if verbose:
            print(f"[ACO] [Run {run + 1}/{runs}] Updating swarm tracker")
        potential_swarm = tracker.update(updated_swarm)
        lfunc.update(potential_swarm)

        if verbose:
            print()


    
    if verbose:
        print(f"[ACO] Finalizing top {cutoff} partial into solutions...")
    tracker.finalize(cutoff, max(0, end - time.time()))

    
    elapsed = time.time() - start
    best_sol = tracker.opt(cutoff=cutoff, time_limit=max(0, end - time.time()))
    best_cost = best_sol.max_cost if best_sol else float("inf")
    info: Dict[str, Any] = {
        "runs_completed": runs_completed,
        "time": elapsed,
        "best_cost": best_cost,
        "elitists_count": tracker.num_partials,
    }

    
    if verbose:
        print(
            f"[ACO] The run finished. "
            f"Runs_completed={info['runs_completed']}, "
            f"Best_cost={info['best_cost']:.3f}, "
            f"Time={info['time']:.3f}s."
        )

    return tracker, info



def aco_solver(
    problem: ShareARideProblem,
    initial_swarm: Optional[PartialSolutionSwarm] = None,

    
    cutoff: int = 5,
    num_ants: int = 10,
    runs: int = 10,
    iterations: Optional[int] = None,
    width: int = 10,

    
    q_prob: float = 0.75,
    alpha: float = 1.2,
    beta: float = 1.4,
    phi: float = 0.5,
    chi: float = 1.5,
    gamma: float = 0.4,
    kappa: float = 2.0,
    sigma: int = 10,
    rho: float = 0.55,

    
    defense_policy: DefensePolicy = _default_defense_policy,
    finalize_policy: FinalizePolicy = _default_finalize_policy,

    
    seed: Optional[int] = None,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> Tuple[Optional[Solution], Dict[str, Any]]:
    
    if initial_swarm is None:
        initial_partials = [
            PartialSolution(problem=problem, routes=[]) for _ in range(num_ants)
        ]
        initial_swarm = PartialSolutionSwarm(solutions=initial_partials)

    
    tracker, run_info = _run_aco(
        problem=problem,
        swarm=initial_swarm,
        runs=runs,
        iterations=iterations,
        width=width,
        q_prob=q_prob,
        alpha=alpha,
        beta=beta,
        phi=phi,
        chi=chi,
        gamma=gamma,
        kappa=kappa,
        sigma=sigma,
        rho=rho,
        defense_policy=defense_policy,
        finalize_policy=finalize_policy,
        cutoff=cutoff,
        seed=seed,
        time_limit=time_limit,
        verbose=verbose,
    )


    best_solution = tracker.opt()

    info: Dict[str, Any] = {
        "runs_completed": run_info["runs_completed"],
        "time": run_info["time"],
        "best_cost": run_info["best_cost"],
        "elitists_count": run_info["elitists_count"],
    }

    if verbose:
        print()
        print("[ACO] Solver complete.")
        if best_solution is not None:
            print(f"[ACO] Best solution cost: {best_solution.max_cost}")
        else:
            print("[ACO] No valid solution found.")
        print(f"[ACO] Total time: {info['time']:.3f}s")
        print("------------------------------")
        print()

    return best_solution, info




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

    
    sol, _ = aco_solver(
        prob,
        cutoff=10,
        num_ants=(
            500 if prob.num_nodes <= 100
            else 150 if prob.num_nodes <= 250
            else 50 if prob.num_nodes <= 500
            else 25 if prob.num_nodes <= 1000
            else 10
        ),
        runs=(
            100 if prob.num_nodes <= 100
            else 75 if prob.num_nodes <= 250
            else 50 if prob.num_nodes <= 500
            else 20 if prob.num_nodes <= 1000
            else 10
        ),
        width=(
            10 if prob.num_nodes <= 100
            else 8 if prob.num_nodes <= 250
            else 6 if prob.num_nodes <= 500
            else 4 if prob.num_nodes <= 1000
            else 2
        ),

        seed=42,
        time_limit=250.0,
        verbose=verbose,
    )


    sol.stdin_print(verbose=verbose)


if __name__ == "__main__":
    main(verbose=True)

