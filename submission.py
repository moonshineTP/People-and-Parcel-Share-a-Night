import sys
import math
import random
import time
import bisect
import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence, Iterator, Set

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
        self.partial_costs = [sol.max_cost for sol in solutions]
        self.min_cost = min(self.partial_costs)
        self.max_cost = max(self.partial_costs)
        self.avg_cost = sum(sol.max_cost for sol in solutions) / len(solutions)


    def update(self) -> None:
         
        self.partial_num_actions = [sol.num_actions for sol in self.partial_lists]
        self.partial_costs = [sol.max_cost for sol in self.partial_lists]
        self.min_cost = min(self.partial_costs)
        self.max_cost = max(self.partial_costs)
        self.avg_cost = sum(sol.max_cost for sol in self.partial_lists) / len(self.partial_lists)


    def opt(self) -> Optional[Solution]:
         
        self.update()

        opt_cost = 10**18
        opt_sol = None
        for par in self.partial_lists:
            if par.is_completed():
                sol = par.to_solution()
                if sol and sol.max_cost < opt_cost:
                    opt_cost = sol.max_cost
                    opt_sol = sol

        return opt_sol


    def stats(self) -> Dict[str, float]:
         
        self.update()

        return {
            "num_partials": self.num_partials,
            "min_cost": self.min_cost,
            "max_cost": self.max_cost,
            "avg_cost": self.avg_cost
        }


    def copy(self):
         
        copied_solutions = [sol.copy() for sol in self.partial_lists]
        return PartialSolutionSwarm(solutions=copied_solutions)



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




class TreeSegment:
    def __init__(
        self,
        data: Sequence[Union[int, float]],
        op: Callable[[Union[int, float], Union[int, float]], Union[int, float]],
        identity: Union[int, float],
        sum_like: bool = True,
        add_neutral: Union[int, float] = 0,
    ):
        self.num_elements = len(data)
        self.op = op
        self.identity = identity
        self.sum_like = sum_like

        self.num_leaves = 1
        while self.num_leaves < self.num_elements:
            self.num_leaves *= 2

        self.data = [self.identity] * (2 * self.num_leaves)
        self.lazy = [add_neutral] * (2 * self.num_leaves)

        for i in range(self.num_elements):
            self.data[self.num_leaves + i] = data[i]
        for i in range(self.num_leaves - 1, 0, -1):
            self.data[i] = self.op(self.data[2 * i], self.data[2 * i + 1])


    def _apply(self, x: int, val: Union[int, float], length: int):
        if self.sum_like:
            self.data[x] += val * length
        else:
            self.data[x] += val
        if x < self.num_leaves:
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
        self._update(l, r, val, 1, 0, self.num_leaves)


    def query(self, l: int, r: int) -> Union[int, float]:
        return self._query(l, r, 1, 0, self.num_leaves)




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




def cost_decrement_intra_swap(
        partial: PartialSolution,
        route_idx: int,
        a_idx: int, len_a: int,
        b_idx: int, len_b: int
    ) -> int:

    if a_idx > b_idx:
        a_idx, b_idx = b_idx, a_idx
        len_a, len_b = len_b, len_a

    D = partial.problem.D   

    route = partial.routes[route_idx]
    route_len = len(route)

    prev_a = route[a_idx - 1]
    start_a = route[a_idx]
    end_a = route[a_idx + len_a - 1]
    next_a = route[a_idx + len_a] if a_idx + len_a < route_len else None
    prev_b = route[b_idx - 1]
    start_b = route[b_idx]
    end_b = route[b_idx + len_b - 1]
    next_b = route[b_idx + len_b] if b_idx + len_b < route_len else None

    
    cost_before = 0
    cost_after = 0
    if a_idx + len_a == b_idx:

        cost_before += D[prev_a][start_a]
        cost_before += D[end_a][start_b]
        if next_b is not None:
            cost_before += D[end_b][next_b]

        
        cost_after += D[prev_a][start_b]
        cost_after += D[end_b][start_a]
        if next_b is not None:
            cost_after += D[end_a][next_b]
    else:
        
        cost_before += D[prev_a][start_a]
        if next_a is not None:
            cost_before += D[end_a][next_a]
        cost_before += D[prev_b][start_b]
        if next_b is not None:
            cost_before += D[end_b][next_b]

        
        cost_after += D[prev_a][start_b]
        if next_a is not None:
            cost_after += D[end_b][next_a]
        cost_after += D[prev_b][start_a]
        if next_b is not None:
            cost_after += D[end_a][next_b]

    return cost_before - cost_after




def intra_swap_one_route_operator(
        partial: PartialSolution,           
        route_idx: int,                     
        steps: Optional[int] = None,        
        mode: str = 'first',                
        uplift: int = 1,                    
        seed: Optional[int] = None,         
        verbose: bool = False               
    ) -> Tuple[PartialSolution, List[bool], int]:

    rng = random.Random(seed)

    partial = partial.copy()
    prob = partial.problem
    route = partial.routes[route_idx]
    k_vehicles = prob.K                      
    n_nodes = len(route)

    if n_nodes < 5:
        return partial, [False] * k_vehicles, 0
    if steps is None:
        steps = n_nodes ** 2

    pos = {}
    actions: List[List[int]] = []        
    action_deltas: List[int] = []
    node_to_action = {}


    def build_structures():
        nonlocal pos, actions, action_deltas, node_to_action

        pos = {node: idx for idx, node in enumerate(route)}
        cur_idx = 1
        action_idx = 0
        while cur_idx < n_nodes - 1:
            node = route[cur_idx]
            length = 0
            delta = 0

            if prob.is_ppick(node):
                if cur_idx + 2 > n_nodes - 1:
                    cur_idx += 1
                    continue
                length = 2
                delta = 0
            elif prob.is_lpick(node):
                length = 1
                lid = prob.rev_lpick(node)
                delta = prob.q[lid - 1]
            elif prob.is_ldrop(node):
                length = 1
                lid = prob.rev_ldrop(node)
                delta = -prob.q[lid - 1]
            else:
                cur_idx += 1
                continue

            actions.append([cur_idx, length])
            action_deltas.append(delta)
            for k in range(length):
                node_to_action[cur_idx + k] = action_idx

            cur_idx += length
            action_idx += 1

    build_structures()

    n_actions = len(actions)
    partners = [-1] * n_actions
    for i in range(n_actions):
        node_idx, _length = actions[i]
        node = route[node_idx]
        if prob.is_lpick(node):
            lid = prob.rev_lpick(node)
            drop_node = prob.ldrop(lid)
            if drop_node in pos:
                drop_idx = pos[drop_node]
                if drop_idx in node_to_action:
                    p_idx = node_to_action[drop_idx]
                    partners[i] = p_idx
                    partners[p_idx] = i

    
    action_loads = [0] * n_actions
    curr = 0
    for i in range(n_actions):
        curr += action_deltas[i]
        action_loads[i] = curr

    min_load_segment = TreeSegment(action_loads, min, 10**18, False)
    max_load_segment = TreeSegment(action_loads, max, 0, False)


    def check_precedence(i: int, j: int) -> bool:
        if action_deltas[i] > 0 and partners[i] <= j:
            return False

        if action_deltas[j] < 0 and partners[j] >= i:
            return False
        return True

    
    def check_load(i: int, j: int) -> bool:
        delta_i = action_deltas[i]
        delta_j = action_deltas[j]
        diff = delta_j - delta_i

        if diff > 0:
            if max_load_segment.query(i, j) + diff > prob.Q[route_idx]:
                return False
        elif diff < 0:
            if min_load_segment.query(i, j) + diff < 0:
                return False

        return True

    
    def check_swap(i: int, j: int) -> Tuple[bool, int]:
        if not (check_precedence(i, j) and check_load(i, j)):
            return False, 0

        idx_a, len_a = actions[i]
        idx_b, len_b = actions[j]
        dec = cost_decrement_intra_swap(partial, route_idx, idx_a, len_a, idx_b, len_b)
        return True, dec


    
    def find_candidates():
        for i in range(n_actions):
            for j in range(i + 1, n_actions):
                feasible, dec = check_swap(i, j)
                if not feasible or dec < uplift:
                    continue

                yield (i, j, dec)
                if mode == 'first':
                    return

    def select_candidate():
        cand_list = list(find_candidates())
        if not cand_list:
            return None
        if mode == 'first':
            return cand_list[0]
        elif mode == 'best':
            return max(cand_list, key=lambda x: x[2])
        elif mode == 'stochastic':
            return rng.choice(cand_list)


    
    swaps_done = 0
    modified = [False] * k_vehicles

    
    def update_partial(action):
        nonlocal route, partial
        i, j, dec = action
        idx_a, len_a = actions[i]
        idx_b, len_b = actions[j]

        
        new_route = (
            route[:idx_a]
            + route[idx_b : idx_b + len_b]
            + route[idx_a + len_a : idx_b]
            + route[idx_a : idx_a + len_a]
            + route[idx_b + len_b:]
        )

        
        route[:] = new_route
        partial.decrease_cost(route_idx, dec)


    
    def update_ds(action):
        nonlocal action_deltas, partners, actions, pos, node_to_action
        i, j, _ = action

        diff = action_deltas[j] - action_deltas[i]
        if diff:
            min_load_segment.update(i, j, diff)
            max_load_segment.update(i, j, diff)
        action_deltas[i], action_deltas[j] = action_deltas[j], action_deltas[i]

        
        if partners[i] != -1:
            partners[partners[i]] = j
        if partners[j] != -1:
            partners[partners[j]] = i
        partners[i], partners[j] = partners[j], partners[i]

        
        idx_a, len_a = actions[i]
        idx_b, len_b = actions[j]
        shift = len_b - len_a

        actions[i], actions[j] = actions[j], actions[i]
        actions[i][0], actions[j][0] = idx_a, idx_b + shift

        if shift:
            for k in range(i + 1, j):
                actions[k][0] += shift

        
        pos = {node: idx for idx, node in enumerate(route)}
        node_to_action = {
            actions[k][0] + l: k
            for k in range(n_actions)
            for l in range(actions[k][1])
        }

    while True:
        if steps is not None and swaps_done >= steps:
            break

        action = select_candidate()
        if action is None:
            break

        update_partial(action)
        update_ds(action)

        swaps_done += 1
        modified[route_idx] = True

    return partial, modified, swaps_done




def intra_swap_operator(
        partial: PartialSolution,           
        steps: Optional[int] = None,        
        mode: str = 'first',                
        uplift: int = 1,                    
        seed: Optional[int] = None,         
        verbose: bool = False
    ) -> Tuple[PartialSolution, List[bool], int]:
    if steps is None:
        steps = 10**9

    total_swaps = 0
    K = partial.problem.K       
    modified: List[bool] = [False] * K
    current_par: PartialSolution = partial.copy()
    for route_idx in range(K):
        tmp_par, modified_one, n_swaps_one = intra_swap_one_route_operator(
            current_par,
            route_idx=route_idx,
            steps=(steps - total_swaps),
            mode=mode,
            uplift=uplift,
            seed=seed,
            verbose=verbose
        )

        current_par = tmp_par
        total_swaps += n_swaps_one
        if modified_one[route_idx]:
            modified[route_idx] = True


    if current_par.is_valid(verbose=verbose) is False:
        current_par.stdin_print()
        raise ValueError("Intra-swap operator produced invalid solution.")

    return current_par, modified, total_swaps




def cost_decrement_inter_swap(
        partial: PartialSolution,
        raidx: int,
        rbidx: int,
        paidx: int,
        qaidx: int,
        pbidx: int,
        qbidx: int,
    ) -> Tuple[int, int, int]:
    
    D = partial.problem.D   

    
    route_a = partial.routes[raidx]
    route_b = partial.routes[rbidx]
    route_a_len = len(route_a)
    route_b_len = len(route_b)
    assert route_a[paidx] != 0 and route_b[pbidx] != 0, "Cannot swap depot nodes."

    
    route_a_cost = partial.route_costs[raidx]
    route_b_cost = partial.route_costs[rbidx]
    max_cost_before = partial.max_cost


    
    
    paprev = route_a[paidx - 1]
    pa = route_a[paidx]
    pasucc = route_a[paidx + 1]
    qaprev = route_a[qaidx - 1]
    qa = route_a[qaidx]
    qasucc = None
    if qaidx + 1 < route_a_len:
        qasucc = route_a[qaidx + 1]

    pbprev = route_b[pbidx - 1]
    pb = route_b[pbidx]
    pbsucc = route_b[pbidx + 1]
    qbprev = route_b[qbidx - 1]
    qb = route_b[qbidx]
    qbsucc = None
    if qbidx + 1 < route_b_len:
        qbsucc = route_b[qbidx + 1]

    
    cost_delta_a = 0
    if paidx + 1 == qaidx:      
        cost_delta_a -= D[paprev][pa] + D[pa][qa]
        if qasucc is not None:
            cost_delta_a -= D[qa][qasucc]

        cost_delta_a += D[paprev][pb] + D[pb][qb]
        if qasucc is not None:
            cost_delta_a += D[qb][qasucc]
    else:                       
        cost_delta_a -= D[paprev][pa] + D[pa][pasucc] + D[qaprev][qa]
        if qasucc is not None:
            cost_delta_a -= D[qa][qasucc]

        cost_delta_a += D[paprev][pb] + D[pb][pasucc] + D[qaprev][qb]
        if qasucc is not None:
            cost_delta_a += D[qb][qasucc]

    
    cost_delta_b = 0
    if pbidx + 1 == qbidx:      
        cost_delta_b -= D[pbprev][pb] + D[pb][qb]
        if qbsucc is not None:
            cost_delta_b -= D[qb][qbsucc]

        cost_delta_b += D[pbprev][pa] + D[pa][qa]
        if qbsucc is not None:
            cost_delta_b += D[qa][qbsucc]
    else:                       
        cost_delta_b -= D[pbprev][pb] + D[pb][pbsucc] + D[qbprev][qb]
        if qbsucc is not None:
            cost_delta_b -= D[qb][qbsucc]

        cost_delta_b += D[pbprev][pa] + D[pa][pbsucc] + D[qbprev][qa]
        if qbsucc is not None:
            cost_delta_b += D[qa][qbsucc]


    
    racost_after = route_a_cost + cost_delta_a
    rbcost_after = route_b_cost + cost_delta_b
    remaining_costs = [
        partial.route_costs[i] for i in range(partial.problem.K)
        if i != raidx and i != rbidx
    ]

    max_cost_after = max(racost_after, rbcost_after, *remaining_costs)
    cost_dec = max_cost_before - max_cost_after

    return racost_after, rbcost_after, cost_dec




def inter_swap_route_pair_operator (
        partial: PartialSolution,           
        route_a_idx: int,                   
        route_b_idx: int,                   
        steps: Optional[int] = None,        
        mode: str = 'first',                
        uplift: int = 1,                    
        seed: Optional[int] = None,         
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

    
    def build_ds(route: List[int]):
        n_nodes = len(route)
        cum_loads = [0] * n_nodes
        load_delta = [0] * n_nodes
        load = 0
        for i, node in enumerate(route):
            dq = 0

            if prob.is_ppick(node) or prob.is_pdrop(node):
                pass
            elif prob.is_lpick(node):
                lid = prob.rev_lpick(node)
                dq = prob.q[lid - 1]
            elif prob.is_ldrop(node):
                lid = prob.rev_ldrop(node)
                dq = -prob.q[lid - 1]

            load += dq
            cum_loads[i] = load
            load_delta[i] = dq

        
        min_load_segment = TreeSegment(
            data=cum_loads, op=min, identity=10**18, sum_like=False
        )
        max_load_segment = TreeSegment(
            data=cum_loads, op=max, identity=0, sum_like=False
        )
        pos = {node: i for i, node in enumerate(route)}

        return pos, load_delta, min_load_segment, max_load_segment

    
    pos_a, load_delta_a, min_load_segment_a, max_load_segment_a = build_ds(route_a)
    pos_b, load_delta_b, min_load_segment_b, max_load_segment_b = build_ds(route_b)

    cap_a = prob.Q[route_a_idx]
    cap_b = prob.Q[route_b_idx]


    
    
    
    def check_load(req_a: Tuple[int, int, str], req_b: Tuple[int, int, str]) -> bool:
        paidx, qaidx, _kind_a = req_a
        pbidx, qbidx, _kind_b = req_b
        
        delta_a = load_delta_b[pbidx] - load_delta_a[paidx]
        
        delta_b = load_delta_a[paidx] - load_delta_b[pbidx]

        
        if delta_a != 0:
            min_a = min_load_segment_a.query(paidx, qaidx)
            max_a = max_load_segment_a.query(paidx, qaidx)
            if min_a + delta_a < 0 or max_a + delta_a > cap_a:
                return False

        
        if delta_b != 0:
            min_b = min_load_segment_b.query(pbidx, qbidx)
            max_b = max_load_segment_b.query(pbidx, qbidx)
            if min_b + delta_b < 0 or max_b + delta_b > cap_b:
                return False

        return True


    def check_consecutivity(req_a: Tuple[int, int, str], req_b: Tuple[int, int, str]) -> bool:
        paidx, qaidx, kind_a = req_a
        pbidx, qbidx, kind_b = req_b

        if kind_a == 'serveP':
            if qbidx != pbidx + 1:
                return False
        if kind_b == 'serveP':
            if qaidx != paidx + 1:
                return False

        return True


    def check_swap(
            req_a: SwapRequest,
            req_b: SwapRequest
        ) -> Tuple[bool, int, int, int]:
        paidx, qaidx, _ = req_a
        pbidx, qbidx, _ = req_b

        if not check_consecutivity(req_a, req_b):
            return False, 0, 0, 0
        if not check_load(req_a, req_b):
            return False, 0, 0, 0

        
        after_cost_a, after_cost_b, dec = cost_decrement_inter_swap(
            current_par,
            route_a_idx, route_b_idx,
            paidx, qaidx,
            pbidx, qbidx,
        )
        return True, after_cost_a, after_cost_b, dec


    
    def find_candidates():
        pickup_indices_a = [
            actid for actid in range(n_a)
            if prob.is_ppick(route_a[actid]) or prob.is_lpick(route_a[actid])
        ]
        pickup_indices_b = [
            actid for actid in range(n_b)
            if prob.is_ppick(route_b[actid]) or prob.is_lpick(route_b[actid])
        ]

        def form_request(p_idx: int, route: List[int], pos: Dict[int, int]) -> Optional[SwapRequest]:
            p_node = route[p_idx]
            kind = ''
            if prob.is_ppick(p_node):
                q_idx = p_idx + 1
                kind = 'serveP'
            else:
                lid = prob.rev_lpick(p_node)
                q_node = prob.ldrop(lid)
                if q_node not in pos:
                    return None

                q_idx = pos[q_node]
                kind = 'serveL'

            return (p_idx, q_idx, kind)


        for paidx in pickup_indices_a:
            req_a = form_request(paidx, route_a, pos_a)
            if req_a is None:
                continue

            for pbidx in pickup_indices_b:
                req_b = form_request(pbidx, route_b, pos_b)
                if req_b is None:
                    continue

                feasible, after_cost_a, after_cost_b, dec = check_swap(
                    req_a, req_b
                )
                if not feasible or dec < uplift:
                    continue

                yield (req_a, req_b, after_cost_a, after_cost_b, dec)
                if mode == 'first':
                    return


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
        steps = 10**9   

    
    def update_partial(action: Tuple[SwapRequest, SwapRequest, int, int, int]):
        nonlocal route_a, route_b, current_par
        nonlocal pos_a, pos_b

        
        req_a, req_b, new_cost_a, new_cost_b, decrement = action
        paidx, qaidx, _ = req_a
        pbidx, qbidx, _ = req_b
        pa, qa = route_a[paidx], route_a[qaidx]
        pb, qb = route_b[pbidx], route_b[qbidx]

        
        del pos_a[pa]
        del pos_a[qa]
        pos_a[pb] = paidx
        pos_a[qb] = qaidx

        del pos_b[pb]
        del pos_b[qb]
        pos_b[pa] = pbidx
        pos_b[qa] = qbidx

        
        route_a[paidx], route_a[qaidx] = pb, qb
        route_b[pbidx], route_b[qbidx] = pa, qa

        
        current_par.node_assignment[pa] = route_b_idx
        current_par.node_assignment[qa] = route_b_idx
        current_par.node_assignment[pb] = route_a_idx
        current_par.node_assignment[qb] = route_a_idx

        
        current_par.route_costs[route_a_idx] = new_cost_a
        current_par.route_costs[route_b_idx] = new_cost_b
        current_par.max_cost -= decrement


    
    def update_segment(action: Tuple[SwapRequest, SwapRequest, int, int, int]):
        nonlocal pos_a, pos_b
        nonlocal load_delta_a, load_delta_b
        nonlocal min_load_segment_a, max_load_segment_a
        nonlocal min_load_segment_b, max_load_segment_b

        req_a, req_b, _, __, ___ = action
        pa, qa, _ = req_a
        pb, qb, _ = req_b


        
        
        dparc_a = load_delta_b[pb] - load_delta_a[pa]
        if dparc_a != 0:
            min_load_segment_a.update(pa, qa, dparc_a)
            max_load_segment_a.update(pa, qa, dparc_a)
        dparc_b = load_delta_a[pa] - load_delta_b[pb]
        if dparc_b != 0:
            min_load_segment_b.update(pb, qb, dparc_b)
            max_load_segment_b.update(pb, qb, dparc_b)

        
        load_delta_a[pa], load_delta_b[pb] = load_delta_b[pb], load_delta_a[pa]
        load_delta_a[qa], load_delta_b[qb] = load_delta_b[qb], load_delta_a[qa]


    
    def swap_until_convergence():
        nonlocal swaps_done, modified, best_improvement
        while swaps_done < steps:
            action = select_candidate()
            if action is None:
                break

            update_segment(action)
            update_partial(action)

            best_improvement += action[4]
            modified[route_a_idx] = True
            modified[route_b_idx] = True
            swaps_done += 1

    swap_until_convergence()
    return current_par, modified, swaps_done




def inter_swap_operator(
        partial: PartialSolution,           
        steps: Optional[int] = None,        
        mode: str = 'first',                
        uplift: int = 1,                    
        seed: Optional[int] = None,         
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
            min_tuple = pop_valid_min(exclude_idx=max_idx)
            if min_tuple is None:
                break
            _, min_idx = min_tuple
            popped_mins.append(min_tuple)

            
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
                
                push_idx(max_idx)
                push_idx(min_idx)

                
                current_par = next_par

                
                total_swaps += n_swaps_pair

                
                if modified_pair[max_idx]:
                    modified[max_idx] = True
                if modified_pair[min_idx]:
                    modified[min_idx] = True

                improved = True
                break

        
        for c, idx in popped_mins:
            heapq.heappush(min_heap, (c, idx))

        
        if not improved:
            push_idx(max_idx)
            break

    if current_par.is_valid(verbose=verbose) is False:
        raise ValueError("Inter-swap operator produced invalid solution.")

    return current_par, modified, total_swaps



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





def _default_vfunc(
        partial: PartialSolution,
        sample_size=6,
        w_std=0.15,
        seed: Optional[int] = None
    ) -> float:
    """Default value function: negative max route cost."""
    return -balanced_scorer(
        partial,
        sample_size=sample_size,
        w_std=w_std,
        seed=seed
    )


def _default_selpolicy(
        actions: List[Action],
        seed: Optional[int] = None,
        t: float = 0.1
    ) -> Optional[Action]:
    """Default selection policy: choose action with minimal incremental cost."""
    rng = random.Random(seed)

    if not actions:
        return None

    weights = softmax_weighter([action_weight(a) for a in actions], t=t)
    chosen_idx = sample_from_weight(rng, weights)

    return actions[chosen_idx]


def _default_simpolicy(
        partial: PartialSolution,
        seed: Optional[int] = None
    ) -> Optional[PartialSolution]:
    """Default simulation policy: greedy balanced solver."""
    sim_solution, _ = greedy_solver(
        partial.problem,
        partial=partial,
    )
    assert sim_solution is not None, "Greedy solver failed in simulation policy."
    seed= 107 * seed + 108 if seed is not None else None

    return PartialSolution.from_solution(sim_solution)


def _default_defpolicy(
        partial: PartialSolution,
        verbose: bool=False,
        seed: Optional[int] = None      
    ) -> Optional[Solution]:
    """Default defense policy: beam search solver."""
    def_sol, _ = iterative_greedy_solver(
        partial.problem,
        partial=partial,
        iterations=2000,
        time_limit=20.0,
        verbose=verbose,
    )

    return def_sol





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


    
    def reward_from_value(self, value: float, reward_pow: float = 1.0) -> float:
        if not math.isfinite(value):
            return 0.0
        if self.visits == 0:
            return 0.5 ** reward_pow
        if self.max_value == self.min_value:
            return 0.5 ** reward_pow

        
        span = self.max_value - self.min_value
        norm = (value - self.min_value) / span

        
        norm = norm ** reward_pow

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
        self.untried_actions = enumerate_actions_greedily(self.partial, self.width)


    
    @property
    def is_terminal(self) -> bool:
        return self.partial.is_completed()


    
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


    
    def update(self, cost: int, reward: float) -> None:

        self.visits += 1
        self.total_cost += cost
        self.total_reward += reward




def _select(root: MCTSNode, exploration: float) -> List[MCTSNode]:

    path = [root]
    current = root
    while True:
        if current.untried_actions:
            return path
        if not current.children:
            return path

        
        current = max(
            current.children,
            key=lambda child: child.uct_score(exploration)
        )

        path.append(current)




def _expand(
    node: MCTSNode,
    selection_policy: Callable,
    width: Optional[int]
) -> Optional[MCTSNode]:

    if not node.untried_actions:
        return None

    
    action = selection_policy(node.untried_actions)
    if action is None:
        return None

    
    try:
        node.untried_actions.remove(action)
    except ValueError:
        pass

    
    child_partial = node.partial.copy()
    apply_general_action(child_partial, action)

    
    child = MCTSNode(child_partial, parent=node, action=action, width=width)
    node.children.append(child)

    
    return child




def _backpropagate(path: List[MCTSNode], cost: int, reward: float) -> None:
    for node in reversed(path):
        node.update(cost, reward)




def _gather_leaves(
    node: MCTSNode,
    value_function: Callable,
    limit: Optional[int] = None,
) -> List[MCTSNode]:
    if limit is None:
        limit = 10**9  

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

    
    width: Optional[int],

    
    uct_c: float,
    cutoff_depth: int,
    cutoff_depth_inc: int,
    cutoff_iter: int,
    reward_pow: float,

    
    value_function: Callable,
    selection_policy: Callable,
    simulation_policy: Callable,
    defense_policy: Callable,

    
    seed: Optional[int],
    time_limit: float,
    verbose: bool,
) -> Tuple[MCTSNode, PartialSolution, Optional[Solution], Dict[str, Any]]:
    start = time.time()
    end = start + time_limit
    reward_function = RewardFunction()

    if seed is not None:
        random.seed(seed)
    if partial is None:
        partial = PartialSolution(problem=problem, routes=[])

    
    base_partial = partial or PartialSolution(problem=problem, routes=[])
    root = MCTSNode(base_partial, width=width)


    
    iterations = 0
    best_leaf: Optional[PartialSolution] = None
    best_solution: Optional[Solution] = None
    best_solution_cost = 10**18
    max_abs_depth = 0  
    next_cutoff_depth = cutoff_depth
    cutoff_cnt = 0
    status = "done"
    while True:
        
        if time.time() >= end:
            status = "overtime"
            if verbose:
                print(f"[MCTS] Reached time limit: {time_limit:.2f}s")
            break

        iterations += 1

        
        path = _select(root, uct_c)
        leaf = path[-1]
        abs_depth = leaf.partial.num_actions  

        if abs_depth > max_abs_depth:
            max_abs_depth = abs_depth
            if abs_depth > 0 and abs_depth == next_cutoff_depth:
                cutoff_cnt += 1
                next_cutoff_depth += cutoff_depth + cutoff_depth_inc * cutoff_cnt
                root = MCTSNode(leaf.partial, width=width)
                reward_function = RewardFunction()
                if verbose:
                    print(
                        f"[MCTS] Cutoff at iter {iterations}, " 
                        f"abs_depth {abs_depth}, new root set."
                    )

                continue  

        
        if iterations > 0 and iterations % cutoff_iter == 0:
            root = MCTSNode(leaf.partial, width=width)
            reward_function = RewardFunction()
            if verbose:
                print(
                    f"[MCTS] Cutoff at iteration {iterations}, "
                    f"depth {abs_depth}, new root set."
                )

            continue  


        
        if not leaf.is_terminal:
            child = _expand(leaf, selection_policy, width)
            if child is not None:
                path.append(child)
                working = child
            else:
                working = leaf
        else:
            break   


        
        rollout_result = simulation_policy(
            working.partial.copy(), seed=12*seed if seed is not None else None
        )
        if rollout_result is None or not rollout_result.is_completed(): 
            if verbose:
                print(f"[MCTS] Rollout failed or incomplete at iteration {iterations}.")

            reward_function.update(float('-inf') )
            continue

        solution: Optional[Solution] = rollout_result.to_solution()
        assert solution is not None, "Conversion from rollout to solution failed."

        
        
        
        solution_cost = solution.max_cost
        if solution_cost <= best_solution_cost:
            if best_leaf is None or working.partial.num_actions > best_leaf.num_actions:
                best_leaf = working.partial.copy()
                best_solution_cost = solution_cost
                best_solution = solution

        
        def_value = value_function(rollout_result)
        reward_function.update(def_value)
        reward = reward_function.reward_from_value(def_value, reward_pow=reward_pow)


        
        _backpropagate(path, solution_cost, reward)

        
        if verbose:
            
            magnitude = 10 ** (len(str(iterations)) - 1)
            if iterations % magnitude == 0 or iterations % 1000 == 0:
                elapsed = time.time() - start
                print(
                    f"[MCTS] [Iteration {iterations}] "
                    f"Cost: {best_solution_cost:.3f}, "
                    f"Value range: {reward_function.min_value:.3f} "
                    f"- {reward_function.max_value:.3f}, "
                    f"Depth: {abs_depth}, "
                    f"Max depth: {max_abs_depth}, "
                    f"Time: {elapsed:.2f}s.",
                )


    
    stats = {
        "iterations": iterations,
        "time": time.time() - start,
        "best_rollout_cost": best_solution_cost,
        "status": status,
    }

    
    if verbose:
        print(
            f"[MCTS] Iterations count: {iterations}, "
            f"Max absolute depth reached: {max_abs_depth}, "
            f"Time={stats['time']:.3f}s."
        )
        print(
            f"[MCTS] Best leaf depth: {best_leaf.num_actions if best_leaf else 'N/A'} "
            f"with rollout cost: {best_solution_cost:.3f}."
        )

    if best_leaf is None:
        best_leaf = root.partial

    
    if best_leaf is not None and best_leaf.is_pending():
        if verbose:
            print(f"[MCTS] Applying defense policy on best leaf...")
        def_sol = defense_policy(
            best_leaf, verbose=verbose, seed=24 * seed if seed is not None else None
        )
        if def_sol is not None:
            cost = def_sol.max_cost
            if best_solution is None or cost < best_solution_cost:
                if verbose:
                    print(f"[MCTS] Defense policy improved solution: {best_solution_cost:.3f} -> {cost:.3f}")
                best_solution = def_sol
                best_solution_cost = cost
                stats["best_rollout_cost"] = best_solution_cost


    return root, best_leaf, best_solution, stats





def mcts_enumerator(
    problem: ShareARideProblem,
    partial: Optional[PartialSolution] = None,

    
    n_return: int = 5,

    
    width: Optional[int] = 3,
    uct_c: float = 0.58,
    cutoff_depth: int = 9,
    cutoff_depth_inc: int = 4,
    cutoff_iter: int = 11300,
    reward_pow: float = 1.69,
    
    value_function: Callable = _default_vfunc,
    selection_policy: Callable = _default_selpolicy,
    simulation_policy: Callable = _default_simpolicy,
    defense_policy: Callable = _default_defpolicy,

    
    seed: Optional[int] = None,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> Tuple[List[PartialSolution], Dict[str, Any]]:

    tree, _, _, info = _run_mcts(
        problem,
        partial,
        width=width,
        uct_c=uct_c,
        cutoff_depth=cutoff_depth,
        cutoff_depth_inc=cutoff_depth_inc,
        cutoff_iter=cutoff_iter,
        reward_pow=reward_pow,
        value_function=value_function,
        selection_policy=selection_policy,
        simulation_policy=simulation_policy,
        defense_policy=defense_policy,
        time_limit=time_limit,
        seed=seed,
        verbose=verbose,
    )

    top_leaves = _gather_leaves(
        tree,
        value_function=value_function,
        limit=max(1, n_return),
    )
    
    top = [leaf.partial.copy() for leaf in top_leaves]

    return top, info




def mcts_solver(
    problem: ShareARideProblem,
    partial: Optional[PartialSolution] = None,

    width: Optional[int] = 3,
    uct_c: float = 0.58,
    cutoff_depth: int = 9,
    cutoff_depth_inc: int = 4,
    cutoff_iter: int = 11300,
    reward_pow: float = 1.69,
    value_function: Callable = _default_vfunc,
    selection_policy: Callable = _default_selpolicy,
    simulation_policy: Callable = _default_simpolicy,
    defense_policy: Callable = _default_defpolicy,

    seed: Optional[int] = None,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> Tuple[Optional[Solution], Dict[str, Any]]:
    start = time.time()

    
    _, best_leaf, sol, info = _run_mcts(
        problem=problem,
        partial=partial,
        value_function=value_function,
        selection_policy=selection_policy,
        simulation_policy=simulation_policy,
        defense_policy=defense_policy,
        width=width,
        uct_c=uct_c,
        cutoff_depth=cutoff_depth,
        cutoff_depth_inc=cutoff_depth_inc,
        cutoff_iter=cutoff_iter,
        reward_pow=reward_pow,
        seed=seed,
        time_limit=time_limit,
        verbose=verbose,
    )
    assert sol

    
    if verbose:
        print(f"[MCTS] Applying relocate operator to final solution...")
    best_partial = PartialSolution.from_solution(sol)
    refined_partial, _, _ = relocate_operator(
        best_partial,
        mode='first',
        seed=None if seed is None else 4 * seed + 123
    )
    sol = refined_partial.to_solution();  assert sol
    best_cost = sol.max_cost
    if verbose:
        print(
            f"[MCTS] After relocate, final solution cost: {best_cost}"
        )

    
    info["used_best_rollout"] = True
    info["iterations"] = info.get("iterations", 0)
    info['time'] = time.time() - start

    
    if verbose:
        if sol is not None:
            print()
            print(
                f"[MCTS] Final solution cost: {sol.max_cost:.3f} "
                f"after {info['iterations']} iterations "
                f"in {info['time']:.2f}s."
            )
            print("------------------------------")
            print()
        else:
            print()
            print(
                f"[MCTS] No solution found after "
                f"{info['iterations']} iterations "
                f"in {info['time']:.2f}s."
            )
            print("------------------------------")
            print()

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

    best_solution, stats = mcts_solver(
        problem,
        width=width,
        uct_c=uct_c,
        cutoff_depth=cutoff_depth,
        cutoff_depth_inc=cutoff_depth_inc,
        cutoff_iter=cutoff_iter,
        seed=42,
        time_limit=250.0,
        verbose=verbose
    )
    assert best_solution
    best_solution.stdin_print()




if __name__ == "__main__":
    main(verbose=False)
