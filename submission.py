import sys
import math
import random
import time
import bisect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence, Iterator, Set

Request = Tuple[int, int, str]
ActionNode = Tuple[int, int]
ValueFunction = Callable
FinalizePolicy = Callable
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

        return sorted(tuple(r) for r in self.routes) == sorted(tuple(r) for r in other.routes)


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
        if state["pos"] == 0 and state["actions"] > 0:
            state["ended"] = True
            return
        if state["parcels"]:
            raise ValueError(f"Taxi {t_idx} must drop all loads before returning to depot.")

         
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
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
     
    start = time.time()

     
    if partial is None:
        partial = PartialSolution(problem=problem)


     
    iterations = 0
    pre_depth = partial.num_actions
    while partial.is_pending():
        iterations += 1

        actions = enumerate_actions_greedily(partial, 1)
        if not actions:

               

            return None, {
                "iterations": iterations,
                "time": time.time() - start,
                "actions_done": partial.num_actions - pre_depth,
                "status": "error",
            }

         
        action = actions[0]
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

     
   
     
    if partial is None:
        partial = PartialSolution(problem=problem, routes=[])


     
    best_sol, greedy_info = greedy_solver(problem, partial=partial, verbose=verbose)
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
    """
    Compute the decrement in max_cost of relocating a full request (pickup and drop)
    from one route to another at specified insertion indices, as Cost_before - Cost_after.
    This assumes the pickup/drop indices are correct with some defensive assertions.

    Parameters:
    - partial: PartialSolution object representing the current solution.
    - from_route_idx: Index of the route from which the request is relocated.
    - to_route_idx: Index of the route to which the request is relocated.
    - pfidx: Index of the pickup node in the from_route.
    - qfidx: Index of the drop node in the from_route.
    - ptidx: Index in the to_route where the pickup node will be inserted.
    - qtidx: Index in the to_route where the drop node will be inserted.

    Returns: A tuple containing:
    - from_route_next_cost: Cost of the from_route after relocation.
    - to_route_next_cost: Cost of the to_route after relocation.
    - cost_decrement: Decrement in max_cost due to the relocation.
    """
    D = partial.problem.D       # pylint: disable=invalid-name
    cur_cost = partial.max_cost

    routef = partial.routes[rfidx]
    routet = partial.routes[rtidx]
    rfcost = partial.route_costs[rfidx]
    rtcost = partial.route_costs[rtidx]
    pf = routef[pfidx]
    qf = routef[qfidx]

    # Basic bound assertions




    # //// Cost change for from_route: remove both pickup and drop
    pprev = routef[pfidx - 1]
    pnext = routef[pfidx + 1]
    qprev = routef[qfidx - 1]
    qnext = routef[qfidx + 1]
    cost_delta_f = 0

    if pfidx + 1 == qfidx:      # Adjacent
        cost_delta_f -= D[pprev][pf] + D[pf][qf] + D[qf][qnext]
        cost_delta_f += D[pprev][qnext]
    else:                       # Non-adjacent
        cost_delta_f -= D[pprev][pf] + D[pf][pnext] + D[qprev][qf] + D[qf][qnext]
        cost_delta_f += D[pprev][pnext] + D[qprev][qnext]

    rfcost_after = rfcost + cost_delta_f


    # ---------- Cost change for to_route: insert pickup then drop ----------
    # Stitch neighbors for insertion (short, consistent names)
    pprev_to = routet[ptidx - 1]
    pnext_to = routet[ptidx]
    qprev_to = routet[qtidx - 2]    # -2 because shift after pickup insertion
    qnext_to = routet[qtidx - 1]    # -1 because shift after pickup insertion
    cost_delta_t = 0

    if qtidx == ptidx + 1:  # Adjacent insertion
        cost_delta_t -= D[pprev_to][qnext_to]
        cost_delta_t += D[pprev_to][pf] + D[pf][qf] + D[qf][qnext_to]
    else:  # Non-adjacent insertion
        cost_delta_t -= D[pprev_to][pnext_to] + D[qprev_to][qnext_to]
        cost_delta_t += D[pprev_to][pf] + D[pf][pnext_to] + D[qprev_to][qf] + D[qf][qnext_to]

    rtcost_after = rtcost + cost_delta_t


    # //// Compute the next max cost
    remain_costs = [
        partial.route_costs[i]
        for i in range(partial.problem.K) if i != rfidx and i != rtidx
    ]
    next_cost = max(rfcost_after, rtcost_after, *remain_costs)

    return rfcost_after, rtcost_after, cur_cost - next_cost




def relocate_from_to(
        partial: PartialSolution,   # Partial solution object
        route_from_idx: int,        # Donor route id
        route_to_idx: int,          # Receiver route id
        steps: int,                 # Number of steps to consider
        mode: str,                  # Mode of operation
        uplift: int = 1,            # Minimum improvement required
        seed: Optional[int] = None, # Seed for reproducibility
        verbose: bool = False       # Verbose output flag
    ) -> Tuple[PartialSolution, List[bool], int]:
    """
    Attempt to relocate requests from one predefined vehicle route to another that
    improves the solution as a helper to the main relocate_operator.

    Parameters:
    - partial: PartialSolution object representing the current solution.
    - from_route_idx: Index of the route from which to relocate requests.
    - to_route_idx: Index of the route to which to relocate requests.
    - steps: Number of steps to consider
    - mode: Mode of operation
    - uplift: Integer controlling the extent of improvement required.
    - seed: Random seed for stochastic modes.
    - verbose: If True, print detailed logs.
    """
    # RNG for stochastic behavior (if used later)
    rng = random.Random(seed)

    # Instance object data
    prob = partial.problem
    current_par = partial.copy()

    # Route data
    route_from = current_par.routes[route_from_idx]
    route_to = current_par.routes[route_to_idx]
    n_from = len(route_from)
    n_to = len(route_to)

    # Early exit if donor route too short
    if n_from < 5:
        return partial, [False] * prob.K, 0    # No requests to relocate


    # //// Build data structures
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

    # Build load delta managers
    load_delta_from_manager = build_ds(route_from, n_from)
    load_delta_to_manager = build_ds(route_to, n_to)

    # Capacity of routes
    cap_from = prob.Q[route_from_idx]
    cap_to = prob.Q[route_to_idx]


    # //// Checker helpers
    def check_consecutive(req: RelocateRequest) -> bool:
        """
        Ensure that pickup and drop indices are consecutive for 'serveP' requests.
        """
        pfidx, qfidx, ptidx, qtidx, kind = req

        if kind == "serveL":
            return True


        return qtidx == ptidx + 1


    def check_load(req: RelocateRequest) -> bool:
        """
        Ensure load stays within [0,cap] after relocating a full request
        (pfidx, qfidx) to (ptidx, qtidx) for both routes.
        """
        pfidx, qfidx, ptidx, qtidx, kind = req
        if kind == "serveP":
            return True

        # Compute load delta
        pf = route_from[pfidx]
        if prob.is_lpick(pf):
            jid = prob.rev_lpick(pf)
            load_delta = prob.q[jid - 1]
        else:
            load_delta = 0

        # Check segment loads on from_route
        load_min_fr = load_delta_from_manager.query_min_prefix(pfidx, qfidx)
        load_max_fr = load_delta_from_manager.query_max_prefix(pfidx, qfidx)
        if load_min_fr - load_delta < 0:
            return False
        if load_max_fr - load_delta > cap_from:
            return False

        # Check segment loads on to_route
        load_min_to = load_delta_to_manager.query_min_prefix(ptidx - 1, qtidx - 1)
        load_max_to = load_delta_to_manager.query_max_prefix(ptidx - 1, qtidx - 1)
        if load_min_to + load_delta < 0:
            return False
        if load_max_to + load_delta > cap_to:
            return False

        return True


    def check_relocate(req: RelocateRequest) -> Optional[CostChange]:
        """
        Check feasibility of relocating the request defined by (pfidx, qfidx)
        from route_from to route_to at insertion indices (ptidx, qtidx).
        
        Returns a tuple of:
            - feasibility (bool)
            - after_cost_a (int): cost of from_route after relocation
            - after_cost_b (int): cost of to_route after relocation
            - dec (int): total cost decrement if relocation is performed
        """
        # Check consecutiveness and load feasibility
        if not check_consecutive(req):
            return None
        if not check_load(req):
            return None

        # Compute cost change
        cost_change = cost_decrement_relocate(
            current_par, route_from_idx, route_to_idx,
            req[0], req[1], req[2], req[3],
        )
        return cost_change


    def find_candidates() -> Iterator[Tuple[RelocateRequest, CostChange]]:
        """
        Find candidate relocation requests according to the specified mode.
        
        Yields tuples of the form (Request, CostChange).
        """
        # Build position map for route_from to locate corresponding drops
        pos_from = {node: i for i, node in enumerate(route_from)}

        # Enumerate all pickup positions in route_from
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

        # Enumerate feasible insertion (pickup, drop) index pairs in route_to
        insert_pairs_pserve = [
            (ptidx, ptidx + 1)
            for ptidx in range(1, n_to)
            if not prob.is_ppick(route_to[ptidx - 1])  # Cannot insert after a pickup
        ]
        insert_pairs_lserve = [
            (ptidx, qtidx)
            for ptidx in range(1, n_to)
            if not prob.is_ppick(route_to[ptidx - 1])
            for qtidx in range(ptidx + 1, n_to + 1)
            if not prob.is_ppick(route_to[qtidx - 2])
        ]

        # Iterate over all delete pairs in route_from
        for (pfidx, qfidx, kind) in delete_pairs:
            insert_pairs = insert_pairs_pserve if kind == "serveP" else insert_pairs_lserve
            for (ptidx, qtidx) in insert_pairs:
                request = (pfidx, qfidx, ptidx, qtidx, kind)
                costchange = check_relocate(request)

                # If costchange is None, relocation is infeasible
                if costchange is None:
                    continue

                # Check uplift requirement
                after_cost_a, after_cost_b, dec = costchange
                if dec < uplift:
                    continue

                if mode == 'first':
                    yield (request, costchange)
                    return
                else:
                    yield (request, costchange)


    def select_candidate() -> Optional[Tuple[RelocateRequest, CostChange]]:
        """
        Select a candidate relocation based on the specified mode.
        """
        cand_list = list(find_candidates())
        if not cand_list:
            return None
        if mode == 'stochastic':
            return rng.choice(cand_list)
        elif mode == 'best':
            # Choose by maximum decrement (index 2 of CostChange)
            return max(cand_list, key=lambda x: x[1][2])
        else:
            return cand_list[0]


    # //// Update helpers
    def update_partial_solution(action: RelocateAction):
        """
        Apply relocation to routes and update costs / max cost for the
        current partial solution object.
        """
        nonlocal route_from, route_to, current_par

        # Unpack action
        (p_from, q_from, p_to, q_to, _), (new_cost_from, new_cost_to, dec) = action

        # Extract nodes to move (pickup & drop) from route_from
        pf = route_from[p_from]
        qf = route_from[q_from]


        # 1. Remove from route_from (remember to remove drop first)
        del route_from[q_from]
        del route_from[p_from]

        # 2. Insert into route_to (remember to insert pickup first)
        route_to.insert(p_to, pf)
        route_to.insert(q_to, qf)

        # 3. Update partial routes
        current_par.routes[route_from_idx] = route_from
        current_par.routes[route_to_idx] = route_to

        # 4. Update partial costs
        current_par.route_costs[route_from_idx] = new_cost_from
        current_par.route_costs[route_to_idx] = new_cost_to
        current_par.max_cost -= dec

        # 5. Update node assignment
        current_par.node_assignment[pf] = route_to_idx
        current_par.node_assignment[qf] = route_to_idx


    def update_precalc(action: RelocateAction):
        """
        Incrementally update passenger & load delta managers after a relocation
        using MinMaxPfsumArray insert/delete operations (avoid full rebuild).

        action: (p_from, q_from, p_to, q_to, new_cost_from, new_cost_to, dec)
        Indices p_from,q_from,p_to,q_to refer to ORIGINAL pre-mutation routes.
        Relocation sequence applied earlier in update_partial_solution:
            1. Remove q_from, then p_from from donor route_from.
            2. Insert pickup at p_to in route_to.
            3. Insert drop at drop_insert_index = (q_to if q_to was final depot else q_to+1).
        Here we commit those operations on the delta managers.
        """
        nonlocal load_delta_from_manager, load_delta_to_manager, route_from, route_to

        # Unpack action
        (pfidx, qfidx, ptidx, qtidx, _), _costchange = action
        pf = route_from[pfidx]
        qf = route_from[qfidx]

        # Helper to map node -> load
        def node_load_delta(nodeid: int) -> int:
            if prob.is_lpick(nodeid):
                jid = prob.rev_lpick(nodeid)
                return prob.q[jid - 1]
            elif prob.is_ldrop(nodeid):
                jid = prob.rev_ldrop(nodeid)
                return -prob.q[jid - 1]
            else:
                return 0


        # 1. Donor route updates
        load_delta_from_manager.delete(qfidx)
        load_delta_from_manager.delete(pfidx)

        # 2. Receiver route updates
        load_delta_to_manager.insert(ptidx, node_load_delta(pf))
        load_delta_to_manager.insert(qtidx, node_load_delta(qf))


    def relocate_to_convergence() -> Tuple[List[bool], int]:
        """
        Perform relocation steps until no further improvement is possible
        or the specified number of steps is reached.

        Returns a tuple of:
        - modified_routes: List of booleans indicating which routes were modified.
        - reloc_done: Number of relocations performed.
        """
        nonlocal n_from, n_to, route_from, route_to

        reloc_done = 0
        modified_routes = [False] * prob.K
        while reloc_done < steps:
            cand = select_candidate()
            if cand is None:
                break

            # Apply the selected relocation
            update_precalc(cand)
            update_partial_solution(cand)

            # Update counters and flags
            reloc_done += 1
            modified_routes[route_from_idx] = True
            modified_routes[route_to_idx] = True

            # Update local route lengths
            n_from -= 2
            n_to += 2
            if n_from < 5:
                break

        return modified_routes, reloc_done

    # Execute relocation to convergence
    modified_pair, reloc_done = relocate_to_convergence()

    return current_par, modified_pair, reloc_done




def relocate_operator(
        partial: PartialSolution,           # Current partial solution
        steps: Optional[int] = None,        # Number of steps to consider
        mode: str = 'first',                # Mode of operation
        uplift: int = 1,                    # Minimum improvement required
        seed: Optional[int] = None,         # Random seed for stochastic mode
        verbose: bool = False,              # Verbosity flag
    ) -> Tuple[PartialSolution, List[bool], int]:
    """
    Attempt to relocate the request from a vehicle route to another that
    improves the solution to different extents controlled by ``uplift``.
    Perform up to ``steps`` relocations based on the specified ``mode``.
    Use in post-processing or local search.

    The procedure attempts to relocate requests from the highest-cost route
    to the 1/3 lower-cost routes (traverse from the lowest to the highest) by
    iterating over all insertion pairs of the receiver routes, in ascending 
    order of their in-out contribution to the route cost.
    If not successful, it moves to the next donor route.

    Parameters:
    - partial: PartialSolution object representing the current solution.
    - steps: Number of relocation steps that the operation should perform.
    - mode: Mode of operation, can be 'best', 'first', or 'stochastic'.
    - uplift: Integer controlling the extent of improvement required.
    - seed: Random seed for stochastic modes.
    - verbose: If True, print detailed logs.

    Returns: a tuple signature containing:
    - A new PartialSolution object with the specified requests relocated.
    - A list of booleans indicating which routes were modified.
    - An integer count of the number of relocations performed.
    """
    k_vehicles = partial.problem.K
    if k_vehicles < 2:
        return partial.copy(), [False] * k_vehicles, 0

    if steps == None:
        steps = 10**9    # Effectively unlimited

    # RNG for stochastic behavior
    rng = random.Random(seed)

    # Initialize tracking variables
    current_par: PartialSolution = partial.copy()
    modified_total: List[bool] = [False] * k_vehicles
    total_moves = 0


    # //// Main relocation loop
    while total_moves < steps:
        # Sort candidate donor and receiver routes ascending by current costs
        taxi_cost: List[Tuple[int, int]] = list(enumerate(current_par.route_costs))
        donor_index = max(taxi_cost, key=lambda x: x[1])[0]
        receiver_indices = [
            idx for idx, _ in sorted(taxi_cost, key=lambda x: x[1])
        ]

        # Break if the donor is too short
        if len(current_par.routes[donor_index]) < 5:
            break

        # Iterate over receiver routes
        improved = False
        for r_idx in receiver_indices:
            # Skip self-relocation
            if r_idx == donor_index:
                continue

            # Skip too-short receivers (no place to insert between depots)
            if len(current_par.routes[r_idx]) < 2:
                continue


            # Attempt relocation from donor to receiver
            remain = steps - total_moves
            new_partial, modified_pair, moves_made = relocate_from_to(
                current_par,
                route_from_idx=donor_index,
                route_to_idx=r_idx,
                steps=remain,
                mode=mode,
                uplift=uplift,
                seed=rng.randint(10, 10**9),  # vary seed between attempts
                verbose=verbose,
            )


            # Analyze results
            if moves_made > 0:
                current_par = new_partial
                total_moves += moves_made
                for i in range(k_vehicles):
                    if modified_pair[i]:
                        modified_total[i] = True
                improved = True

                # Verbose logging

                   

                break   # break receivers loop, re-sort donors/receivers

        # Exit if no improvement found in this iteration (convergence)
        if not improved:
            break

    # Logging

       
       
       
       
       
       
       

    return current_par, modified_total, total_moves




 
# ================ ACO Policies Implementation ================
def _default_value_function(
        partial: PartialSolution,
        perturbed_samples: int = 6,     # perturb
        seed: Optional[int] = None,     # pylint: disable=unused-argument
    ) -> float:
    _, stats = _, stats = iterative_greedy_solver(
        partial.problem, partial, iterations=perturbed_samples, time_limit=0.1, seed=seed
    )

    return stats["average_cost"]


def _default_finalize_policy(
        partial: PartialSolution,
        seed: Optional[int] = None,
    ) -> Optional[Solution]:
    sol, _info = iterative_greedy_solver(
        partial.problem,
        partial,
        iterations=3000,
        time_limit=3.0,
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




# ================ ACO Components Implementation ================
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
        """Update metrics with a single solution."""
        cost = solution.max_cost

        self.count += 1
        self.total_cost += cost
        self.worst_cost = max(self.worst_cost, cost)

        if cost < self.best_cost:
            self.best_cost = cost
            self.best_solution = solution


    def _update_from_swarm(self, swarm: PartialSolutionSwarm) -> None:
        """Update metrics with a swarm of partial solutions."""
        for partial in swarm.partial_lists:
            if not partial.is_completed():
                continue

            sol = partial.to_solution()
            if not sol:
                continue

            self._update_from_solution(sol)


    def _update_from_list(self, solutions: List[Optional[Solution]]) -> None:
        """Update metrics with a list of solutions."""
        for sol in solutions:
            if sol is None:
                continue

            self._update_from_solution(sol)


    def stats(self) -> Dict[str, float]:
        """Return statistics of the population."""
        avg_cost = self.total_cost / self.count if self.count > 0 else 0.0
        return {
            "best_cost": self.best_cost,
            "worst_cost": self.worst_cost,
            "avg_cost": avg_cost,
            "count": self.count,
        }


    def opt(self) -> Optional[Solution]:
        """Return the best solution found so far."""
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

        assert 0.0 < self.rho < 1.0, "Evaporation rate rho must be in (0,1)."
        assert sigma >= 1, "Number of elitists sigma must be at least 1."


    def get(self, prev: ActionNode, curr: ActionNode) -> float:
        """Get pheromone level on transition from prev action to curr action."""
        return self.tau[prev[1]][curr[0]]


    def update(
            self,
            swarm: PartialSolutionSwarm,
            opt: Optional[Solution],
        ) -> None:

        # //// Helper function to extract edges from a PartialSolution
        def extract_edges(partial: PartialSolution) -> List[Tuple[int, int]]:
            """Extract all (prev_out, curr_in) edges from a partial solution's routes."""
            edges: List[Tuple[int, int]] = []

            for route_idx, _ in enumerate(partial.routes):
                actnodes = partial.enumerate_action_nodes(route_idx)
                for prev, nxt in zip(actnodes[:-1], actnodes[1:]):
                    edges.append((prev[1], nxt[0]))

            return edges


        # //// Sort partials by max_cost (ascending) and take elitists
        ranked_partials = sorted(
            ((par.max_cost, par) for par in swarm.partial_lists),
            key=lambda x: x[0]
        )[:self.sigma]


        # //// Update pheromone matrix
        # Evaporation and clamp to tau_min
        self.tau = [
            [max(self.tau_min, self.rho * val) for val in row]
            for row in self.tau
        ]

        increased_edges: Set[Tuple[int, int]] = set()

        # Add ranked solution contributions: (sigma - rank) * (1 / L_r)
        for rank, (cost, partial) in enumerate(ranked_partials):
            elitist_weight = (self.sigma - rank) / cost

            # Loop through edges
            for (i, j) in extract_edges(partial):
                increased_edges.add((i, j))
                if 0 <= i < self.size and 0 <= j < self.size:
                    self.tau[i][j] += elitist_weight

        # Add best-so-far solution contribution: sigma * (1 / L_best)
        if opt is not None and opt.max_cost > 0:
            best_weight = self.sigma / opt.max_cost

            # Temporarily convert best solution to PartialSolution
            best_partial = PartialSolution.from_solution(opt)

            # Loop through edges
            for (i, j) in extract_edges(best_partial):
                increased_edges.add((i, j))
                if 0 <= i < self.size and 0 <= j < self.size:
                    self.tau[i][j] += best_weight

        # Clamp increased edges to tau_max
        for (i, j) in increased_edges:
            self.tau[i][j] = min(self.tau_max, self.tau[i][j])



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

        # Initialize saving matrix
        self.saving_matrix: List[List[float]] = []
        D = self.problem.D      # pylint: disable=C0103
        for i in range(self.size):
            row = []

            for j in range(self.size):
                if i == j:
                    row.append(0)   # Dummy
                else:
                    # Clark-Wright savings: positive when combining saves distance
                    slack_ij = max(D[0][i] + D[j][0] - D[i][j], 0)
                    # Add 1 to savings to ensure non-zero desirability even with no savings
                    saving_term = (1 + slack_ij) ** self.phi

                    row.append(saving_term)

            self.saving_matrix.append(row)


    def get(
            self,
            prev: ActionNode,
            curr: ActionNode,
            partial: PartialSolution,
            action: Action
        ) -> float:
        # Extract variables
        route_idx, kind, node, inc = action
        Q = self.problem.Q[route_idx]        # pylint: disable=C0103
        state = partial.states[route_idx]
        new_cap = state["load"]
        if kind == "pickL":
            new_cap += self.problem.q[node - 1]
        if kind == "dropL":
            new_cap -= self.problem.q[node - 1]

        # Formulate terms
        distance_term = (1 + weighted(kind, inc)) ** self.chi
        saving_term = self.saving_matrix[prev[1]][curr[0]]
        people_term = 2 - int(kind == "pickP")
        parcel_term = (1 + self.gamma * (Q - new_cap) / Q) * self.kappa

        return (saving_term / distance_term) * people_term * parcel_term


class NearestExpansionCache:
    def __init__(
            self,
            problem: ShareARideProblem,
            n_nearest: int = 3
        ) -> None:
        self.nearest_actions: List[List[Tuple[str, int, int]]] = []


        # //// Build a surrogate PartialSolution to query possible expansions
        for nodeid in range(problem.num_nodes):
            # For depot (node 0), use [[0], ...] to avoid marking taxi as ended
            # For other nodes, use [[0, node_idx], ...] to simulate being at that node
            if nodeid == 0:
                routes = [[0] for _ in range(problem.K)]
            elif problem.is_ppick(nodeid):
                # Decisions are never made at pickup nodes for passengers
                self.nearest_actions.append([])
                continue
            elif problem.is_pdrop(nodeid):
                pid = problem.rev_pdrop(nodeid)
                pick = problem.pserve(pid)[0]
                routes = [[0, pick, nodeid]] + [[0] for _ in range(problem.K - 1)]
            elif problem.is_lpick(nodeid):
                routes = [[0, nodeid]] + [[0] for _ in range(problem.K - 1)]
            elif problem.is_ldrop(nodeid):
                lid = problem.rev_ldrop(nodeid)
                pick = problem.lpick(lid)
                routes = [[0, pick, nodeid]] + [[0] for _ in range(problem.K - 1)]
            else:
                print(f"[ACO] [Error] Cache error: Unknown node type for node {nodeid}.")
                self.nearest_actions.append([])
                continue

            partial = PartialSolution(problem, routes=routes)


            # //// Get possible expansions and keep nearest num_nearest
            t_acts = partial.possible_expand(0)
            t_acts.sort(key=lambda item: weighted(item[0], item[2]))
            t_acts = t_acts[:n_nearest]

            # Store in cache
            self.nearest_actions.append(t_acts)


    def query(self, partial: PartialSolution, n_queried: int) -> List[Action]:
        if partial.num_actions < partial.problem.num_expansions:
            return []   # No expansions when every request is served (only returns)


        # //// Collect expansion actions ////
        current_max = partial.max_cost
        prioritized: List[Tuple[float, Action]] = []
        secondary: List[Tuple[float, Action]] = []

        for route_idx, state in enumerate(partial.states):
            if state["ended"]:
                continue
            pos = state["pos"]
            cached: List[Tuple[str, int, int]] = self.nearest_actions[pos]

            # Filter and prioritize expansion actions
            for unassigned_action in cached:
                # Extract action components for prioritization
                kind, node_idx, inc = unassigned_action

                # Check validity
                if not partial.check_expand(route_idx, kind, node_idx):
                    continue

                # Reassign taxi for the action
                action: Action = (route_idx, kind, node_idx, inc)
                weight = weighted(kind, inc)

                # Prioritize like enumerate_actions_greedily
                if partial.route_costs[route_idx] + inc <= current_max:
                    prioritized.append((weight, action))
                else:
                    secondary.append((weight, action))

        # Sort by weight
        prioritized.sort(key=lambda x: x[0])
        secondary.sort(key=lambda x: x[0])

        # Combine and return top num_queried actions
        all_actions = [action for _, action in prioritized + secondary]

        return all_actions[:n_queried]


class Ant:

    class ProbaExpandSampler:
        partial: PartialSolution
        cache: "NearestExpansionCache"
        alpha: float
        beta: float
        omega: float
        q_prob: float
        width: int

        def __init__(
                self,
                partial: PartialSolution,
                cache: "NearestExpansionCache",
                alpha: float,
                beta: float,
                omega: float,
                q_prob: float,
                width: int
            ) -> None:
            self.partial = partial
            self.cache = cache
            self.alpha = alpha
            self.beta = beta
            self.omega = omega
            self.q_prob = q_prob
            self.width = width


        def _get_action_node(self, action: Action) -> ActionNode:
            """Get ActionNode for a given action."""
            _route_idx, kind, actid, _ = action
            prob = self.partial.problem
            if kind == "serveP":
                return prob.pserve(actid)
            elif kind == "pickL":
                node = prob.lpick(actid)
                return (node, node)
            elif kind == "dropL":
                node = prob.ldrop(actid)
                return (node, node)
            else:
                return (0, 0)


        def _collect_actions(self) -> List[Tuple[float, Action]]:
            partial = self.partial
            width = self.width


            # //// Ending case
            if partial.num_actions >= partial.problem.num_expansions:
                actions = enumerate_actions_greedily(partial, width)
                first_weight = action_weight(actions[0])
                return [(first_weight / action_weight(action), action) for action in actions]


            # //// Expansion case
            # First try to get from cache
            actions = self.cache.query(partial, width)

            # If not enough, fill with greedy enumeration
            if len(actions) < width:
                actions = enumerate_actions_greedily(partial, width)[:width]

            # Now we reweight actions based on priority
            # Note that the first secondary action has incremental cost
            # smaller than the last prioritized action
            # Because of that, the trick is to raise the weight of secondary
            # actions above the max weight of prioritized actions
            weight_actions: List[Tuple[float, Action]] = []
            found_secondary = False
            reweight_base = 0.0
            first_action = actions[0]
            first_weight = action_weight(first_action)
            weight_actions.append((first_weight, first_action))

            for prev_action, curr_action in zip(actions[:-1], actions[1:]):
                prev_inc = prev_action[3]
                curr_inc = curr_action[3]

                if not found_secondary and curr_inc > prev_inc:
                    found_secondary = True

                    max_prior_weight = action_weight(curr_action)
                    reweight_base = max_prior_weight + 1.0

                weight = action_weight(curr_action)
                if found_secondary:
                    weight += reweight_base

                weight_actions.append((weight, curr_action))

            # Finally, invert and normalize weights by first action's inc cost
            fitted_actions = [
                (first_weight / weight, action) for weight, action in weight_actions
            ]

            return fitted_actions


        def _compute_log_proba(
                self,
                tau: PheromoneMatrix,
                eta: DesirabilityMatrix,
                fit: float,
                action: Action,
            ) -> float:
            # Extract from_node and to_node
            route_idx = action[0]
            state = self.partial.states[route_idx]
            prev_out = state["pos"]
            prev_node: ActionNode = (10**18, prev_out)   # Dummy prev node, we only need phys_out
            curr_node: ActionNode = self._get_action_node(action)

            # Get tau and eta values
            tau_val = tau.get(prev_node, curr_node)
            eta_val = eta.get(prev_node, curr_node, self.partial, action)

            # Clamp to avoid log(0)
            tau_val = max(tau_val, 1e-300)
            eta_val = max(eta_val, 1e-300)

            log_proba = (
                + self.alpha * math.log(tau_val)
                + self.beta * math.log(eta_val)
                + self.omega * math.log(fit)
            )

            return log_proba


        def sample_action(
                self,
                tau: PheromoneMatrix,
                eta: DesirabilityMatrix,
                rng: random.Random,
            ) -> Optional[Action]:
            # Collect actions using cache-first strategy
            actions = self._collect_actions()
            if not actions:
                return None

            # Compute log probabilities for each action
            log_probas: List[float] = []
            for weight, action in actions:
                log_proba = self._compute_log_proba(tau, eta, weight, action)
                log_probas.append(log_proba)

            # Convert to probabilities using log-sum-exp trick for numerical stability
            # p_i = exp(log_p_i - max_log_p) / sum(exp(log_p_j - max_log_p))
            max_log = max(log_probas)
            exp_shifted = [math.exp(lp - max_log) for lp in log_probas]
            total = sum(exp_shifted)
            probas = [e / total for e in exp_shifted]

            # Select action: exploit vs explore
            select_idx: int
            if rng.random() < self.q_prob:
                # Exploitation: choose action with max probability
                select_idx = probas.index(max(probas))
            else:
                # Exploration: sample from distribution
                select_idx = sample_from_weight(rng, probas)

            return actions[select_idx][1]


    def __init__(
            self,
            partial: PartialSolution,
            cache: "NearestExpansionCache",
            tau: PheromoneMatrix,
            eta: DesirabilityMatrix,
            alpha: float,
            beta: float,
            omega: float,
            q_prob: float,
            width: int,
            rng: random.Random,
        ) -> None:

        """Initialize ant with parameters and partial solution."""
        self.problem = partial.problem
        self.partial = partial
        self.cache = cache
        self.tau = tau
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.q_prob = q_prob
        self.width = width
        self.rng = rng

        # Initialize probabilistic expansion strategy
        self.sampler = Ant.ProbaExpandSampler(
            partial=self.partial,
            cache=cache,
            alpha=alpha,
            beta=beta,
            omega=omega,
            q_prob=q_prob,
            width=width
        )


    def expand(self) -> bool:
        if self.partial.is_completed():
            return False

        # Sample action using cache-first strategy
        sampled_action = self.sampler.sample_action(
            self.tau, self.eta, self.rng
        )
        if not sampled_action:
            return False

        # Commit action
        apply_general_action(self.partial, sampled_action)

        return True


class AntPopulation:

    def __init__(
            self,
            swarm: PartialSolutionSwarm,

            # Cache, Pheromone and desirability matrices (initialized earlier)
            cache: NearestExpansionCache,
            tau: PheromoneMatrix,
            eta: DesirabilityMatrix,
            lfunc: SolutionTracker,

            # Ant params
            alpha: float,
            beta: float,
            omega: float,
            q_prob: float,
            width: int,

            # Run parameters
            depth: int,
            time_limit: float,
            seed: Optional[int],
            verbose: bool
        ) -> None:
        self.swarm = swarm.copy()
        self.completed = [par.is_completed() for par in self.swarm.partial_lists]

        self.cache = cache
        self.tau = tau
        self.eta = eta
        self.lfunc = lfunc

        self.depth = depth
        self.time_limit = time_limit
        self.seed = seed
        self.verbose = verbose

        # Initialize ants
        self.ants: List[Ant] = []
        for idx, partial in enumerate(self.swarm.partial_lists):
            ant = Ant(
                partial=partial,
                cache=cache,
                tau=self.tau,
                eta=self.eta,
                alpha=alpha,
                beta=beta,
                omega=omega,
                q_prob=q_prob,
                width=width,
                rng=random.Random(hash(seed + 100 * idx) if seed else None),
            )
            self.ants.append(ant)

        # Miscellaneous utilities
        self.num_ants = len(self.ants)
        self.max_actions = swarm.problem.num_actions
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
                raise RuntimeError("Ant expansion failure.")

        return is_expanded


    def update(self) -> None:
        self.lfunc.update(
            source=self.swarm,  # Update from the current swarm
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

        # Loop
        for ite in range(self.depth):
            if self.tle():
                if self.verbose:
                    print("[ACO] Time limit reached, skipping this iteration.")
                return self.swarm

            # Logging
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

            # Perform expansion and pheromone update
            if not self.expand():
                if self.verbose:
                    print("[ACO] All ants have completed their solutions.")
                break

            self.update()


        # Final logging
        if self.verbose:
            num_sol = sum(1 for par in self.swarm.partial_lists if par.is_completed())
            run_opt = self.swarm.opt()
            global_opt = self.lfunc.opt()
            print(
                f"[ACO] Finished all depth.\n"
                f"Complete solutions found: {num_sol}/{self.num_ants}.\n"
                f"Run best cost: {run_opt.max_cost if run_opt else 'N/A'}, "
                f"Opt cost: {global_opt.max_cost if global_opt else 'N/A'}."
            )

        return self.swarm   # A new swarm with updated partials


class SwarmTracker:
    def __init__(
            self,
            swarm: PartialSolutionSwarm,
            value_function: ValueFunction,
            finalize_policy: FinalizePolicy,
            seed: Optional[int] = None,
        ) -> None:

        # Initialization fields
        self.seed = seed

        # Tracking fields
        self.frontier_swarm: List[PartialSolution] = [
            partial.copy() for partial in swarm.partial_lists
        ]
        self.num_partials = swarm.num_partials

        # Fitness fields (value_function now returns float fitness values)
        self.frontier_fitness: List[float] = [
            value_function(partial.copy())
            for partial in self.frontier_swarm
        ]

        # Finalized solutions
        self.finals: List[Solution]
        self.is_finalized: bool = False

        # Policy fields
        self.value_function = value_function
        self.finalize_policy = finalize_policy


    def update(self, source: PartialSolutionSwarm) -> List[float]:
        assert self.num_partials == source.num_partials

        for idx, partial in enumerate(source.partial_lists):
            fitness = self.value_function(
                partial.copy(), seed=self.seed + 10 * idx if self.seed else None
            )

            # Compare to the current frontier partial (lower fitness is better)
            if fitness < self.frontier_fitness[idx]:
                self.frontier_swarm[idx] = partial.copy()
                self.frontier_fitness[idx] = fitness

        return self.frontier_fitness


    def finalize(self, cutoff: Optional[int]) -> List[Solution]:
        # //// Select partials
        sorted_partials = sorted(
            zip(self.frontier_swarm, self.frontier_fitness),
            key=lambda x: x[1]
        )
        chosen_partials = sorted_partials[:cutoff] if cutoff else sorted_partials


        # //// Finalizing
        finalized: List[Solution] = []
        for idx, (par, _fitness) in enumerate(chosen_partials):
            sol = self.finalize_policy(
                par, seed=self.seed + 20 * idx if self.seed else None
            )
            if sol:
                finalized.append(sol)


        # /// Update and return
        finalized.sort(key=lambda s: s.max_cost)
        finalized = finalized[:cutoff] if cutoff else finalized

        # Update final state
        self.is_finalized = True
        self.finals = finalized

        # Return finalized solutions
        return finalized


    def top(
            self,
            k: int,
            cutoff: Optional[int] = None,
        ) -> List[Solution]:

        if cutoff is None:
            cutoff = k

        if not self.is_finalized:
            self.finalize(cutoff)

        return self.finals[:k]


    def opt(
            self,
            cutoff: Optional[int] = None,
        ) -> Solution:
        if not self.is_finalized:
            self.finalize(cutoff)


        return self.finals[0]  # Already sorted by max_cost



 
# ================ ACO Logic Implementation ================
def _run_aco(
    problem: ShareARideProblem,
    swarm: PartialSolutionSwarm,

    # Run parameters
    n_cutoff: Optional[int],
    iterations: int,
    depth: Optional[int],

    # Hyperparameters
    # / Ants
    q_prob: float,
    alpha: float,
    beta: float,
    omega: float,

    # / Desirability
    phi: float,
    chi: float,
    gamma: float,
    kappa: float,
    # / Pheromone
    sigma: int,
    rho: float,

    # / Width
    width: int,

    # Policies
    value_function: Callable,
    finalize_policy: Callable,

    # Meta parameters
    seed: Optional[int],
    time_limit: float,
    verbose: bool,
) -> Tuple[SwarmTracker, Dict[str, Any]]:

    start = time.time()
    if depth is None:
        depth = problem.num_actions

    # Run an initial greedy solver to estimate initial cost for pheromone initialization

       
    init_sol, _info = iterative_greedy_solver(
        problem=problem,
        iterations=1000,
        time_limit=2.5,
        seed=10*seed if seed else None,
        verbose=False,
    )

    init_cost = init_sol.max_cost

       


    # Initialize caches for nearest expansion

       
    cache = NearestExpansionCache(problem, n_nearest=5)

    # Initialize pheromone and desirability matrices

       
    tau = PheromoneMatrix(problem, sigma=sigma, rho=rho, init_cost=init_cost)
    eta = DesirabilityMatrix(problem, phi, chi, gamma, kappa)

    # Initialize solution tracker with the initial greedy solution

       
    lfunc = SolutionTracker()
    lfunc.update(init_sol)
    tracker = SwarmTracker(
        swarm=swarm,
        value_function=value_function,
        finalize_policy=finalize_policy,
    )


    # Main ACO iterations
    iterations_completed = 0
    status = "done"
    for run in range(iterations):
        if time.time() - start >= 0.75 * time_limit:
            break


           

        # Initialize a new population with all components
        population = AntPopulation(
            swarm=swarm,
            cache=cache,
            tau=tau,
            eta=eta,
            lfunc=lfunc,
            alpha=alpha,
            beta=beta,
            omega=omega,
            q_prob=q_prob,
            width=width,
            depth=depth,
            time_limit=time_limit,
            seed=hash(seed + 10 * run) if seed else None,
            verbose=verbose,
        )

        # Run the ACO process and get updated swarm

           
        result_swarm = population.run()

        # Update the trackers with the new swarm

           
           

        tracker.update(result_swarm)
        lfunc.update(result_swarm)

        iterations_completed = run + 1

    # Finalize the best partials into complete solutions

       
       
    tracker.finalize(n_cutoff)

    # Inject the best solution found by ants (if better)
    if lfunc.best_solution:
        tracker.finals.append(lfunc.best_solution)
        tracker.finals.sort(key=lambda s: s.max_cost)
        if n_cutoff and len(tracker.finals) > n_cutoff:
             tracker.finals = tracker.finals[:n_cutoff]

    # Summary info
    elapsed = time.time() - start
    best_sol = tracker.opt(cutoff=n_cutoff)
    best_cost = best_sol.max_cost if best_sol else float("inf")
    stats: Dict[str, Any] = {
        "iterations": iterations_completed,
        "best_cost": best_cost,
        "elitists_count": tracker.num_partials,
        "time": elapsed,
        "status": status,
    }

    return tracker, stats




# ================ ACO API Functions ================
def aco_enumerator(
    problem: ShareARideProblem,
    swarm: Optional[PartialSolutionSwarm] = None,

    # Run parameters
    n_partials: int = 50,
    n_cutoff: int = 10,
    n_return: int = 5,
    iterations: int = 10,
    depth: Optional[int] = None,

    # Tuning hyperparameters
    q_prob: float = 0.71,
    alpha: float = 1.36,
    beta: float = 1.38,
    omega: float = 3,
    phi: float = 0.43,
    chi: float = 1.77,
    gamma: float = 0.40,
    kappa: float = 2.34,
    sigma: int = 12,
    rho: float = 0.62,
    width: int = 4,

    # Policies
    value_function: Callable = _default_value_function,
    finalize_policy: Callable = _default_finalize_policy,

    # Meta parameters
    seed: Optional[int] = None,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> Tuple[List[Solution], Dict[str, Any]]:
    """
    Run ACO to enumerate the best k complete solutions.
    
    This function executes ACO with a SwarmTracker that tracks the best
    variation of each partial across all depth. At the end, it finalizes
    the top `n_cutoff` partials and returns the best `n_return` solutions.
    
    Args:
        - problem: ShareARideProblem instance.
        - swarm: Initial PartialSolutionSwarm (optional, created if None).

        - n_partials: Number of ants (must match swarm size if provided).
        - n_return: Number of best solutions to return.
        - n_cutoff: Maximum partials to finalize (time-saving cutoff). Should be
                >= n_return to avoid missing potentially better solutions.
        - iterations: Number of ACO iterations to run.
        - depth: Number of depth to run.
        
        - q_prob: Exploitation probability (0 = explore, 1 = exploit).
        - alpha: Pheromone influence exponent.
        - beta: Desirability influence exponent.
        - phi: Savings influence exponent for desirability.
        - chi: Distance influence exponent for desirability.
        - gamma: Parcel influence factor for desirability.
        - kappa: Parcel influence exponent for desirability.
        - sigma: Number of elitists for pheromone update.
        - rho: Evaporation rate for pheromone update.
        - width: Maximum actions to consider per expansion.

        - value_function: Function to evaluate potential of partial solutions.
        - finalize_policy: Function to complete partial solutions.

        - seed: Random seed for reproducibility.
        - time_limit: Total time limit in seconds.
        - verbose: If True, print detailed logs.
    
    Returns:
        - solutions: List of top n_return solutions from finalized partials.
        - info: Dictionary with run statistics.
    """
    # Create initial swarm if not provided
    if swarm is None:
        initial_partials = [
            PartialSolution(problem=problem, routes=[]) for _ in range(n_partials)
        ]
        swarm = PartialSolutionSwarm(solutions=initial_partials)


    # //// Run ACO
    tracker, run_info = _run_aco(
        problem=problem,
        swarm=swarm,

        n_cutoff=n_cutoff,
        iterations=iterations,
        depth=depth,

        q_prob=q_prob,
        alpha=alpha,
        beta=beta,
        omega=omega,
        phi=phi,
        chi=chi,
        gamma=gamma,
        kappa=kappa,
        sigma=sigma,
        rho=rho,
        width=width,

        value_function=value_function,
        finalize_policy=finalize_policy,

        seed=seed,
        time_limit=time_limit,
        verbose=verbose,
    )

    # Get top n_return solutions
    top_solutions = tracker.top(
        k=n_return,
        cutoff=n_cutoff,
    )

    # Build info from run_info
    stats: Dict[str, Any] = {
        "iterations": run_info["iterations"],
        "time": run_info['time'],
        "best_cost": run_info["best_cost"],
        "solutions_found": len(top_solutions),
        "elitists_count": run_info["elitists_count"],
        "status": run_info['status'],
    }
       
       
       

    return top_solutions, stats


def aco_solver(
    problem: ShareARideProblem,
    swarm: Optional[PartialSolutionSwarm] = None,

    # Run parameters
    n_partials: int = 50,
    n_cutoff: int = 10,
    iterations: int = 40,
    depth: Optional[int] = None,

    # Hyperparameters
    q_prob: float = 0.72,
    alpha: float = 1.36,
    beta: float = 1.38,
    omega: float = 3,
    phi: float = 0.43,
    chi: float = 1.77,
    gamma: float = 0.40,
    kappa: float = 2.34,
    sigma: int = 12,
    rho: float = 0.62,
    width: int = 4,

    # Policies
    value_function: Callable = _default_value_function,
    finalize_policy: Callable = _default_finalize_policy,

    # Meta parameters
    seed: Optional[int] = None,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Run ACO solver and return the best complete solution.
    
    This function executes the ACO algorithm using SwarmTracker to track
    the best partials across depth and returns the optimal solution.
    
    Args:
        - problem: ShareARideProblem instance.
        - swarm: Initial PartialSolutionSwarm (optional, created if None).

        - n_partials: Number of ants (must match swarm size if provided).
        - n_cutoff: Maximum partials to finalize (time-saving cutoff). Should be
                > 1 to avoid missing potentially better solutions.
        - iterations: Number of ACO iterations to run.
        - depth: Number of depth to run.
        
        - q_prob: Exploitation probability (0 = explore, 1 = exploit).
        - alpha: Pheromone influence exponent.
        - beta: Desirability influence exponent.
        - phi: Savings influence exponent for desirability.
        - chi: Distance influence exponent for desirability.
        - gamma: Parcel influence factor for desirability.
        - kappa: Parcel influence exponent for desirability.
        - sigma: Number of elitists for pheromone update.
        - rho: Evaporation rate for pheromone update.
        - width: Maximum actions to consider per expansion.

        - value_function: Function to evaluate potential of partial solutions.
        - finalize_policy: Function to complete partial solutions.

        - seed: Random seed for reproducibility.
        - time_limit: Total time limit in seconds.
        - verbose: If True, print detailed logs.
    
    Returns:
        - solution: Best solution found from the tracker (or None if failed).
        - info: Dictionary with run statistics.
    """
    # Create initial swarm if not provided
    if swarm is None:
        initial_partials = [
            PartialSolution(problem=problem, routes=[]) for _ in range(n_partials)
        ]
        swarm = PartialSolutionSwarm(solutions=initial_partials)


    # //// Run ACO
    tracker, run_info = _run_aco(
        problem=problem,
        swarm=swarm,

        n_cutoff=n_cutoff,
        iterations=iterations,
        depth=depth,

        q_prob=q_prob,
        alpha=alpha,
        beta=beta,
        omega=omega,
        phi=phi,
        chi=chi,
        gamma=gamma,
        kappa=kappa,
        sigma=sigma,
        rho=rho,
        width=width,

        value_function=value_function,
        finalize_policy=finalize_policy,

        seed=seed,
        time_limit=time_limit,
        verbose=verbose,
    )

    # Get optimal solution from tracker
    best_solution = tracker.opt()

    # Build info from run_info
    stats: Dict[str, Any] = {
        "iterations": run_info["iterations"],
        "time": run_info['time'],
        "best_cost": run_info["best_cost"],
        "elitists_count": run_info["elitists_count"],
        "status": run_info['status']
    }
 
       
       

    return best_solution, stats



 


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
    if n <= 100:
        n_partials, n_cutoff, iterations, width = 150, 25, 150, 6
    elif n <= 250:
        n_partials, n_cutoff, iterations, width = 60, 15, 60, 5
    elif n <= 500:
        n_partials, n_cutoff, iterations, width = 25, 10, 25, 4
    elif n <= 1000:
        n_partials, n_cutoff, iterations, width = 12, 4, 8, 3
    else:
        n_partials, n_cutoff, iterations, width = 6, 2, 3, 2

    solution, _ = aco_solver(
        problem,
        seed=42,
        verbose=verbose,
        n_partials=n_partials,
        n_cutoff=n_cutoff,
        iterations=iterations,
        width=width,
        time_limit=240.0
    )

    solution.stdin_print(verbose=verbose)


if __name__ == "__main__":
    main(verbose=False)

 
