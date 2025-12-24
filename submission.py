
import sys
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence




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
            if verbose:
                print(f"// Route cost: {cost}")
                print("------------------------------")
                print()

        if verbose:
            print(f"//// Max route cost: {self.max_cost} ////")


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
            if verbose:
                print("Invalid: Mismatch in number of routes, states, or costs.")
            return False
        if len(self.node_assignment) != len(prob.D):
            if verbose:
                print("Invalid: Mismatch in node assignment length.")
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
                if verbose:
                    print(f"Invalid: Route {t_idx} does not start with depot 0.")
                return False
            is_ended = route_len > 1 and route[-1] == 0
            if state["ended"] != is_ended:
                if verbose:
                    print(f"Invalid: Ended state mismatch for taxi {t_idx}.")
                return False

             
            parcel_onboard: set[int] = set()
            route_len = len(route)
            load = 0
            prev = route[0]
            cumulated_cost = 0


             
            for idx, node in enumerate(route[1:], start=1):
                 
                if not 0 <= node < prob.num_nodes:
                    if verbose:
                        print(f"Invalid: Node {node} out of range in route {t_idx}.")
                    return False

                 
                if node != 0:
                    assigned = node_assignment_check[node]
                    if assigned not in (-1, t_idx):
                        if verbose:
                            print(
                                f"Invalid: Node {node} assigned to an unintended route "
                                f"{assigned} instead of {t_idx}."
                            )
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
                            if verbose:
                                print(
                                    f"Invalid: Passenger {pid} pickup at node {node} "
                                    f"not followed by drop at node {drop_node} "
                                    f"in route {t_idx}."
                                )
                            return False

                     
                    total_actions += 1
                    expected_pass_serve.discard(pid)

                 
                elif prob.is_pdrop(node):
                    pass

                 
                elif prob.is_lpick(node):
                    lid = prob.rev_lpick(node)

                     
                    if lid in parcel_onboard:
                        if verbose:
                            print(
                                f"Invalid: Parcel {lid} picked up multiple times "
                                f"in route {t_idx}."
                            )
                        return False

                     
                    load += prob.q[lid - 1]
                    if load > prob.Q[t_idx]:
                        if verbose:
                            print(
                                f"Invalid: Taxi {t_idx} load {load} exceeds capacity "
                                f"{prob.Q[t_idx]} after picking parcel {lid} in index {idx}."
                            )
                        return False

                     
                    total_actions += 1
                    parcel_onboard.add(lid)
                    expected_parc_pick.discard(lid)
                    expected_parc_drop.add(lid)

                 
                elif prob.is_ldrop(node):
                    lid = prob.rev_ldrop(node)

                     
                    if lid not in parcel_onboard:
                        if verbose:
                            print(
                                f"Invalid: Parcel {lid} dropped without being picked up "
                                f"in route {t_idx}."
                            )
                        return False

                     
                    load -= prob.q[lid - 1]
                    if load < 0:
                        if verbose:
                            print(
                                f"Invalid: Taxi {t_idx} has negative load "
                                f"after dropping parcel {lid}."
                            )
                        return False

                     
                    total_actions += 1
                    parcel_onboard.remove(lid)
                    expected_parc_drop.discard(lid)

                else:    
                    if idx != route_len - 1:
                        if verbose:
                            print(
                                f"Invalid: Depot node in the middle of route {t_idx}."
                            )
                        return False
                    if load != 0 or parcel_onboard:
                        if verbose:
                            print(
                                f"Invalid: Taxi {t_idx} has load remaining "
                                f"upon returning to depot."
                            )
                        return False
                    if not is_ended:
                        if verbose:
                            print(
                                f"Invalid: Taxi {t_idx} route does not end properly "
                                f"after returning to depot."
                            )
                        return False

                    total_actions += 1

             
            if state["parcels"] != parcel_onboard:
                if verbose:
                    print(
                        f"Invalid: Parcel onboard state mismatch for taxi {t_idx}. "
                        f"Expected {parcel_onboard}, got {state['parcels']}."
                    )
                return False
            if state["load"] != load:
                if verbose:
                    print(
                        f"Invalid: Load state mismatch for taxi {t_idx}. "
                        f"Expected {state['load']}, got {load}."
                    )
                return False
            if self.route_costs[t_idx] != cumulated_cost:
                if verbose:
                    print(
                        f"Invalid: Cost state mismatch for taxi {t_idx}. "
                        f"Expected {self.route_costs[t_idx]}, got {cumulated_cost}."
                    )
                return False

             
            max_cost_check = max(max_cost_check, cumulated_cost)
            cost_sum += cumulated_cost


         
         
        if expected_pass_serve != self.remaining_pass_serve:
            if verbose:
                print("Invalid: Remaining passenger serve set mismatch.")
            return False
        if expected_parc_pick != self.remaining_parc_pick:
            if verbose:
                print("Invalid: Remaining parcel pick set mismatch.")
            return False
        if expected_parc_drop != self.remaining_parc_drop:
            if verbose:
                print("Invalid: Remaining parcel drop set mismatch.")
            return False
        if node_assignment_check != self.node_assignment:
            if verbose:
                print("Invalid: Node assignment mismatch.")
            return False

         
        if self.max_cost != max_cost_check:
            if verbose:
                print("Invalid: Max cost mismatch.")
            return False
        if self.num_actions != total_actions:
            if verbose:
                print(
                    "Invalid: Total actions mismatch: "
                    f"expected {self.num_actions}, got {total_actions}."
                )
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


    def stdin_print(self, verbose: bool = False):
         
        print(self.problem.K)
        for route, cost in zip(self.routes, self.route_costs):
            print(len(route))
            print(" ".join(map(str, route)))
            if verbose:
                print(f"// Route cost: {cost}")
                print("------------------------------")
                print()

        if verbose:
            print(f"//// Max route cost: {self.max_cost} ////")


     
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
            if verbose:
                print(
                    f"Not completed: current partial actions {self.num_actions} "
                    f"does not suffice total actions {self.problem.num_actions}."
                )
            return False

        if not all(state["ended"] for state in self.states):
            if verbose:
                print("Not completed: at least one route has not ended at depot.")
            return False

        return True


    def to_solution(self) -> Optional[Solution]:
         
        if not self.is_completed(verbose=True):
            print("Warning: Solution is not complete, cannot convert.")
            return None

        if not self.is_valid(verbose=True):
            print("Warning: Solution is not valid, cannot convert.")
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


Action = Tuple[int, str, int, int]
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

 


Action = Tuple[int, str, int, int]




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
            print("[Warning] No feasible actions found before closing depth")
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
            if verbose:
                print(f"[Repair] Route {route_idx} has no feasible actions, return to depot.")
            partial.apply_return(route_idx)
            added_actions += 1
            break

         
        incs = [weighted(kind, inc) for kind, _, inc in actions]
        weights = softmax_weighter(incs, T)
        selected_idx = sample_from_weight(rng, weights)

         
        kind, node_idx, inc = actions[selected_idx]
        partial.apply_extend(route_idx, kind, node_idx, inc)
        added_actions += 1

        if verbose:
            print(f"[Repair] Route {route_idx} select action {actions[selected_idx]}")

    if verbose:
        print(f"[Repair] Route {route_idx} finished building, added {added_actions} actions.")

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

        if verbose:
            print(f"[Repair]: Repairing route {r_idx} with up to {steps} steps.")


     
    if verbose:
        print()
        print("[Repair] Operator completed.")
        print(f"Total routes repaired: {repair_count};")
        print(f"Total actions added: {total_added_actions}.")
        print("------------------------------")
        print()

    partial.stdin_print()

    return partial, modified, total_added_actions

 

 
 
 
from typing import List, Tuple, Optional, Union

 
 
 
 




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

     
    if verbose:
        print(f"[Destroy] Route {route_idx}: removed {actions_removed} actions.")

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

     
    if verbose:
        print()
        print("[Destroy] Operation complete.")
        print(
            f"[Destroy] Destroyed {len(selected_ids)} routes, "
            f"removed {actions_removed} nodes total."
        )
        print("------------------------------")
        print()

     
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
            if verbose:
                print("[Greedy] [Error] The partial has no feasible actions available.")

            return None, {
                "iterations": iterations,
                "time": time.time() - start,
                "actions_done": partial.num_actions - pre_depth,
                "status": "error",
            }

         
        action = actions[0]
        apply_general_action(partial, action)

        if verbose:
            taxi, kind, idx, _inc = action
            print(
                f"[Greedy] [Depth {partial.num_actions}] "
                f"Taxi {taxi} extended route with action {kind} on passenger/parcel {idx}"
            )

     
    sol = partial.to_solution()
    stats = {
        "iterations": iterations,
        "time": time.time() - start,
        "actions_done": partial.num_actions - pre_depth,
        "status": "done",
    }

     
    if verbose:
        print()
        print("[Greedy] Completed.")
        print(f"[Greedy] Solution max cost: {sol.max_cost if sol else 'N/A'}")
        print(f"[Greedy] Time taken: {stats['time']:.4f} seconds")
        print("------------------------------")
        print()

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

    if verbose:
        print(f"[Iterative Greedy] [Iter 0] initial best cost: {best_cost}")


     
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

             
            if verbose:
                print(f"[Iterative Greedy] [Iter {it}] improved best to {best_cost}")

     
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

     
    if verbose:
        print()
        print(f"[Iterative Greedy] Finished after {iterations_done} iterations.")
        print(
            f"[Iterative Greedy] Best solution max cost: "
            f"{best_sol.max_cost if best_sol else 'N/A'}."
        )
        print(f"[Iterative Greedy] Time taken: {elapsed:.4f} seconds.")
        print("------------------------------")
        print()

    return best_sol, stats


 
 
 
import heapq
from typing import List, Optional, Tuple, Dict

 

 
 
from typing import Callable, Union, Sequence, List, Tuple
 
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


Request = Tuple[int, int, str]



ActionNode = Tuple[int, int]
ValueFunction = Callable
FinalizePolicy = Callable




 
def _default_value_function(
        partial: PartialSolution,
        perturbed_samples: int = 8,     
        seed: Optional[int] = None,      
    ) -> float:
     
    _, stats = _, stats = iterative_greedy_solver(
        partial.problem, partial, iterations=perturbed_samples, time_limit=0.1, seed=seed
    )

    return stats["average_cost"]


def _default_finalize_policy(
        partial: PartialSolution,
        iterations: int = 1000,
        seed: Optional[int] = None,
    ) -> Optional[Solution]:

    sol, _info = iterative_greedy_solver(
        partial.problem,
        partial,
        iterations=iterations,
        time_limit=3.0,
        seed=seed,
        verbose=False
    )

    if not sol:
        return None
    return sol




 
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
            if not partial.is_completed():
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

    def get(self, prev: ActionNode, curr: ActionNode) -> float:
        return self.tau[prev[1]][curr[0]]


    def update(
            self,
            swarm: PartialSolutionSwarm,
            opt: Optional[Solution],
        ) -> None:

         
        def extract_edges(partial: PartialSolution) -> List[Tuple[int, int]]:
            edges: List[Tuple[int, int]] = []

            for route_idx, _ in enumerate(partial.routes):
                actnodes = partial.enumerate_action_nodes(route_idx)
                for prev, nxt in zip(actnodes[:-1], actnodes[1:]):
                    edges.append((prev[1], nxt[0]))

            return edges


         
        ranked_partials = sorted(
            ((par.max_cost, par) for par in swarm.partial_lists),
            key=lambda x: x[0]
        )[:self.sigma]


         
         
        self.tau = [
            [max(self.tau_min, self.rho * val) for val in row]
            for row in self.tau
        ]

        increased_edges: Set[Tuple[int, int]] = set()

         
        for rank, (cost, partial) in enumerate(ranked_partials):
            elitist_weight = (self.sigma - rank) / cost

             
            for (i, j) in extract_edges(partial):
                increased_edges.add((i, j))
                if 0 <= i < self.size and 0 <= j < self.size:
                    self.tau[i][j] += elitist_weight

         
        if opt is not None and opt.max_cost > 0:
            best_weight = self.sigma / opt.max_cost

             
            best_partial = PartialSolution.from_solution(opt)

             
            for (i, j) in extract_edges(best_partial):
                increased_edges.add((i, j))
                if 0 <= i < self.size and 0 <= j < self.size:
                    self.tau[i][j] += best_weight

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

         
        self.saving_matrix: List[List[float]] = []
        D = self.problem.D       
        for i in range(self.size):
            row = []

            for j in range(self.size):
                if i == j:
                    row.append(0)    
                else:
                     
                    slack_ij = max(D[0][i] + D[j][0] - D[i][j], 0)
                     
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
         
         
        route_idx, kind, node, inc = action
        Q = self.problem.Q[route_idx]         
        state = partial.states[route_idx]
        new_cap = state["load"]
        if kind == "pickL":
            new_cap += self.problem.q[node - 1]
        if kind == "dropL":
            new_cap -= self.problem.q[node - 1]

         
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


         
        for nodeid in range(problem.num_nodes):
             
             
            if nodeid == 0:
                routes = [[0] for _ in range(problem.K)]
            elif problem.is_ppick(nodeid):
                 
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


             
            t_acts = partial.possible_expand(0)
            t_acts.sort(key=lambda item: weighted(item[0], item[2]))
            t_acts = t_acts[:n_nearest]

             
            self.nearest_actions.append(t_acts)


    def query(self, partial: PartialSolution, n_queried: int) -> List[Action]:
         
        if partial.num_actions < partial.problem.num_expansions:
            return []    


         
        current_max = partial.max_cost
        prioritized: List[Tuple[float, Action]] = []
        secondary: List[Tuple[float, Action]] = []

        for route_idx, state in enumerate(partial.states):
            if state["ended"]:
                continue
            pos = state["pos"]
            cached: List[Tuple[str, int, int]] = self.nearest_actions[pos]

             
            for unassigned_action in cached:
                 
                kind, node_idx, inc = unassigned_action

                 
                if not partial.check_expand(route_idx, kind, node_idx):
                    continue

                 
                action: Action = (route_idx, kind, node_idx, inc)
                weight = weighted(kind, inc)

                 
                if partial.route_costs[route_idx] + inc <= current_max:
                    prioritized.append((weight, action))
                else:
                    secondary.append((weight, action))

         
        prioritized.sort(key=lambda x: x[0])
        secondary.sort(key=lambda x: x[0])

         
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


             
            if partial.num_actions >= partial.problem.num_expansions:
                actions = enumerate_actions_greedily(partial, width)
                first_weight = action_weight(actions[0])
                return [(first_weight / action_weight(action), action) for action in actions]


             
             
            actions = self.cache.query(partial, width)

             
            if len(actions) < width:
                actions = enumerate_actions_greedily(partial, width)[:width]

             
             
             
             
             
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
             
             
            route_idx = action[0]
            state = self.partial.states[route_idx]
            prev_out = state["pos"]
            prev_node: ActionNode = (10**18, prev_out)    
            curr_node: ActionNode = self._get_action_node(action)

             
            tau_val = tau.get(prev_node, curr_node)
            eta_val = eta.get(prev_node, curr_node, self.partial, action)

             
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
             

             
            actions = self._collect_actions()
            if not actions:
                return None

             
            log_probas: List[float] = []
            for weight, action in actions:
                log_proba = self._compute_log_proba(tau, eta, weight, action)
                log_probas.append(log_proba)

             
             
            max_log = max(log_probas)
            exp_shifted = [math.exp(lp - max_log) for lp in log_probas]
            total = sum(exp_shifted)
            probas = [e / total for e in exp_shifted]

             
            select_idx: int
            if rng.random() < self.q_prob:
                 
                select_idx = probas.index(max(probas))
            else:
                 
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
            swarm: PartialSolutionSwarm,

             
            cache: NearestExpansionCache,
            tau: PheromoneMatrix,
            eta: DesirabilityMatrix,
            lfunc: SolutionTracker,

             
            alpha: float,
            beta: float,
            omega: float,
            q_prob: float,
            width: int,

             
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

         
        for ite in range(self.depth):
            if self.tle():
                if self.verbose:
                    print("[ACO] Time limit reached, skipping this iteration.")
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
            num_sol = sum(1 for par in self.swarm.partial_lists if par.is_completed())
            run_opt = self.swarm.opt()
            global_opt = self.lfunc.opt()
            print(
                f"[ACO] Finished all depth.\n"
                f"Complete solutions found: {num_sol}/{self.num_ants}.\n"
                f"Run best cost: {run_opt.max_cost if run_opt else 'N/A'}, "
                f"Opt cost: {global_opt.max_cost if global_opt else 'N/A'}."
            )

        return self.swarm    


class SwarmTracker:
     
    def __init__(
            self,
            swarm: PartialSolutionSwarm,
            value_function: ValueFunction,
            finalize_policy: FinalizePolicy,
            seed: Optional[int] = None,
        ) -> None:

         
        self.seed = seed

         
        self.frontier_swarm: List[PartialSolution] = [
            partial.copy() for partial in swarm.partial_lists
        ]
        self.num_partials = swarm.num_partials

         
        self.frontier_fitness: List[float] = [
            value_function(partial)
            for partial in self.frontier_swarm
        ]

         
        self.finals: List[Solution]
        self.is_finalized: bool = False

         
        self.value_function = value_function
        self.finalize_policy = finalize_policy


    def update(self, source: PartialSolutionSwarm) -> List[float]:
         
 
        for idx, partial in enumerate(source.partial_lists):
            fitness = self.value_function(
                partial, seed=self.seed + 10 * idx if self.seed else None
            )

             
            if fitness < self.frontier_fitness[idx]:
                self.frontier_swarm[idx] = partial.copy()
                self.frontier_fitness[idx] = fitness

        return self.frontier_fitness


    def finalize(self, cutoff: Optional[int]) -> List[Solution]:
         
         
        sorted_partials = sorted(
            zip(self.frontier_swarm, self.frontier_fitness),
            key=lambda x: x[1]
        )
        chosen_partials = sorted_partials[:cutoff] if cutoff else sorted_partials


         
        finalized: List[Solution] = []
        for idx, (par, _fitness) in enumerate(chosen_partials):
            sol = self.finalize_policy(
                par, seed=self.seed + 20 * idx if self.seed else None
            )
            if sol:
                finalized.append(sol)


         
        finalized.sort(key=lambda s: s.max_cost)
        finalized = finalized[:cutoff] if cutoff else finalized

         
        self.is_finalized = True
        self.finals = finalized

         
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

 
        return self.finals[0]   




 
def _run_aco(
    problem: ShareARideProblem,
    swarm: PartialSolutionSwarm,

     
    n_cutoff: Optional[int],
    iterations: int,
    depth: Optional[int],

     
     
    q_prob: float,
    alpha: float,
    beta: float,
    omega: float,

     
    phi: float,
    chi: float,
    gamma: float,
    kappa: float,
     
    sigma: int,
    rho: float,

     
    width: int,

     
    value_function: ValueFunction,
    finalize_policy: FinalizePolicy,

     
    seed: Optional[int],
    time_limit: float,
    verbose: bool,
) -> Tuple[SwarmTracker, Dict[str, Any]]:
     
    start = time.time()
    if depth is None:
        depth = problem.num_actions

     
    if verbose:
        print("[ACO] [Init] Estimating costs from initial greedy solver...")
    init_sol, _info = greedy_solver(problem, None, False)
    init_cost = init_sol.max_cost
    if verbose:
        print(f"[ACO] [Init] Greedy solution cost: {init_cost:.3f}")


     
    if verbose:
        print("[ACO] [Init] Initializing nearest expansion cache...")
    cache = NearestExpansionCache(problem, n_nearest=5)

     
    if verbose:
        print("[ACO] [Init] Initializing matrices...")
    tau = PheromoneMatrix(problem, sigma=sigma, rho=rho, init_cost=init_cost)
    eta = DesirabilityMatrix(problem, phi, chi, gamma, kappa)

     
    if verbose:
        print("[ACO] [Init] Initializing trackers...")
    lfunc = SolutionTracker()
    lfunc.update(init_sol)
    tracker = SwarmTracker(
        swarm=swarm,
        value_function=value_function,
        finalize_policy=finalize_policy,
    )


     
    iterations_completed = iterations
    status = "done"
    for run in range(iterations):
        if time.time() - start >= 0.75 * time_limit:
            iterations_completed = run
            status = "overtime"
            if verbose:
                print(f"[ACO] Time limit approaching, stopping at run {run + 1}/{iterations}.")
            break

        if verbose:
            print(f"[ACO] [Run {run + 1}/{iterations}] Starting the population run...")

         
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

         
        if verbose:
            print(f"[ACO] [Run {run + 1}/{iterations}] Running the ant population")
        result_swarm = population.run()

         
        if verbose:
            print(f"[ACO] [Run {run + 1}/{iterations}] Updating swarm tracker")
        tracker.update(result_swarm)
        lfunc.update(result_swarm)

     
    if verbose:
        print(f"[ACO] Finalizing top {n_cutoff} partial into solutions...")
    tracker.finalize(n_cutoff)

     
    elapsed = time.time() - start
    best_sol = tracker.opt(cutoff=n_cutoff)
    best_cost = best_sol.max_cost if best_sol else float("inf")
    stats: Dict[str, Any] = {
        "iterations_completed": iterations_completed,
        "best_cost": best_cost,
        "elitists_count": tracker.num_partials,
        "time": elapsed,
        "status": status,
    }

     
    if verbose:
        print(
            f"[ACO] The run finished. "
            f"iterations_completed={stats['iterations_completed']}, "
            f"Best_cost={stats['best_cost']:.3f}, "
            f"Time={stats['time']:.3f}s."
        )

    return tracker, stats




 
def aco_enumerator(
    problem: ShareARideProblem,
    swarm: Optional[PartialSolutionSwarm] = None,

     
    n_partials: int = 50,
    n_cutoff: int = 20,
    n_return: int = 5,
    iterations: int = 10,
    depth: Optional[int] = None,

     
    q_prob: float = 0.75,
    alpha: float = 1.2,
    beta: float = 1.4,
    omega: float = 4,
    phi: float = 0.5,
    chi: float = 1.5,
    gamma: float = 0.4,
    kappa: float = 2.0,
    sigma: int = 10,
    rho: float = 0.55,
    width: int = 8,

     
    value_function: ValueFunction = _default_value_function,
    finalize_policy: FinalizePolicy = _default_finalize_policy,

     
    seed: Optional[int] = None,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> Tuple[List[Solution], Dict[str, Any]]:
     
     
    if swarm is None:
        initial_partials = [
            PartialSolution(problem=problem, routes=[]) for _ in range(n_partials)
        ]
        swarm = PartialSolutionSwarm(solutions=initial_partials)


     
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

     
    top_solutions = tracker.top(
        k=n_return,
        cutoff=n_cutoff,
    )

     
    stats: Dict[str, Any] = {
        "iterations_completed": run_info["iterations_completed"],
        "time": run_info['time'],
        "best_cost": run_info["best_cost"],
        "solutions_found": len(top_solutions),
        "elitists_count": run_info["elitists_count"],
        "status": run_info['status'],
    }

     
    if verbose:
        print()
        print("[ACO] Enumeration complete.")
        print(f"[ACO] Total solutions found: {stats['solutions_found']}.")
        print(
            f"[ACO] Solution costs range: "
            f"{top_solutions[0].max_cost:.3f} "
            f"- {top_solutions[-1].max_cost:.3f}."
        )
        print(f"[ACO] Total time: {stats['time']:.3f}s")
        print("------------------------------")
        print()

    return top_solutions, stats


def aco_solver(
    problem: ShareARideProblem,
    swarm: Optional[PartialSolutionSwarm] = None,

     
    n_partials: int = 20,
    n_cutoff: int = 10,
    iterations: int = 10,
    depth: Optional[int] = None,

     
    q_prob: float = 0.75,
    alpha: float = 1.2,
    beta: float = 1.4,
    omega: float = 4,
    phi: float = 0.5,
    chi: float = 1.5,
    gamma: float = 0.4,
    kappa: float = 2.0,
    sigma: int = 10,
    rho: float = 0.55,
    width: int = 5,

     
    value_function: ValueFunction = _default_value_function,
    finalize_policy: FinalizePolicy = _default_finalize_policy,

     
    seed: Optional[int] = None,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> Tuple[Optional[Solution], Dict[str, Any]]:
     
     
    if swarm is None:
        initial_partials = [
            PartialSolution(problem=problem, routes=[]) for _ in range(n_partials)
        ]
        swarm = PartialSolutionSwarm(solutions=initial_partials)


     
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

     
    best_solution = tracker.opt()

     
    stats: Dict[str, Any] = {
        "iterations_completed": run_info["iterations_completed"],
        "time": run_info['time'],
        "best_cost": run_info["best_cost"],
        "elitists_count": run_info["elitists_count"],
        "status": run_info['status']
    }

     
    if verbose:
        print()
        print("[ACO] Solver complete.")
        if best_solution is not None:
            print(f"[ACO] Best solution cost: {best_solution.max_cost}")
        else:
            print("[ACO] No valid solution found.")
        print(f"[ACO] Total time: {stats['time']:.3f}s")
        print("------------------------------")
        print()

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
    main(verbose=True)

 
