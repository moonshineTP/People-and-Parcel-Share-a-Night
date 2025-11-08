


import sys
import time
import time, random
from typing import Any, List, Optional, Tuple, Dict



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
                    id = prob.rev_ppick(node)

                    
                    if id in visited_pass_pick:
                        return False
                    if len(passenger_onboard) >= 1:
                        return False

                    
                    passenger_onboard.add(id)
                    visited_pass_pick.add(id)

                
                elif prob.is_pdrop(node):
                    id = prob.rev_pdrop(node)

                    
                    if id not in passenger_onboard:
                        return False

                    
                    passenger_onboard.remove(id)

                
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


    def stdin_print(self, verbose: int = 0):
        
        if verbose:
            print(f"*** Max route cost: {self.max_cost} ***")

        print(self.problem.K)
        assert len(self.routes) == len(self.route_costs)
        for route, cost in zip(self.routes, self.route_costs):
            if verbose:
                print(f"- Route cost: {cost}")
            print(len(route))
            print(" ".join(map(str, route)))


class PartialSolution:
    

    def __init__(
            self,
            problem: ShareARideProblem,
            routes: List[List[int]] = [],
        ):
        

        
        self.problem = problem
        self.routes = self._init_routes(routes)
        self.route_costs = self._init_costs(routes)

        
        self.max_cost = max(self.route_costs)
        self.node_assignment = self._init_node_assignment()
        (   self.remaining_pass_pick, self.remaining_pass_drop, self.remaining_parc_pick,
            self.remaining_parc_drop, self.route_states
        ) = self._init_states()


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
                "parcels": set(onboard_parcels),
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


    def copy(self):
        
        return PartialSolution(
            problem=self.problem,
            routes=[list(route) for route in self.routes]
        )


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
        self.node_assignment[node] = t_idx
        self.route_costs[t_idx] = state["cost"]
        self.max_cost = max(self.max_cost, state["cost"])


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
        self.route_costs[t_idx] = state["cost"]
        self.max_cost = max(self.max_cost, state["cost"])
        state["route"].append(0)
        state["pos"] = 0
        state["ended"] = True



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
    def from_solution(sol: Solution):
        
        routes_copy = [list(route) for route in sol.routes]
        return PartialSolution(problem=sol.problem, routes=routes_copy)

from typing import List, Tuple, Optional

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






def softmax_weighter(incs: List[int], T: float) -> List[float]:
    
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
        prob: ShareARideProblem,
        premature_routes: List[List[int]] = [],
        verbose: bool = False
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    

    start_time = time.time()
    partial = PartialSolution(problem=prob, routes=premature_routes)
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

        if verbose:
            print(f"Taxi with min cost: {argmin_t_idx}")
            print(f"Actions available: {actions}")

        
        if not actions:
            partial.apply_return_to_depot(argmin_t_idx)
            continue

        
        kind, idx, inc = min(actions, key=lambda x: x[2])
        partial.apply_action(argmin_t_idx, kind, idx, inc)

        if verbose:
            print(f"Taxi: {argmin_t_idx}: {taxi_states[argmin_t_idx]['route']}\n")


    
    for t_idx, t_state in enumerate(taxi_states):
        if not t_state["ended"]:
            partial.apply_return_to_depot(t_idx)

    if verbose:
        print("All tasks completed.")

    
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
    return sol, info


def iterative_greedy_balanced_solver(
        prob: ShareARideProblem,
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

    
    best_sol, base_info = greedy_balanced_solver(prob, verbose=False)
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
        print(f"[iter 0] initial best cost: {best_cost}")


    
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
            prob,
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
                print(f"[iter {it}] improved best to {best_cost}")


    
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

    return best_sol, info








import heapq
from typing import List, Optional, Tuple






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
        
        
        bid, iid = self._find_block(idx)
        self.block_arr[bid].insert(iid, val)

        
        self.n_data += 1
        self._rebuild_indexing()


    def delete(self, idx):
        
        
        bid, iid = self._find_block(idx)
        self.block_arr[bid].erase(iid)

        
        self.n_data -= 1
        if self.block_arr[bid].size == 0:
            del self.block_arr[bid]
        self._rebuild_indexing()


    def query_min_prefix(self, l, r):
        
        csum = 0
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
        
        cur = 0
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
    if a_idx < b_idx - 1:
        delta = (
            D[route[a_idx - 1]][route[a_idx]] + D[route[a_idx]][route[a_idx + 1]]
            + D[route[b_idx - 1]][route[b_idx]] + D[route[b_idx]][route[b_idx + 1]]
            - D[route[a_idx - 1]][route[b_idx]] - D[route[b_idx]][route[a_idx + 1]]
            - D[route[b_idx - 1]][route[a_idx]] - D[route[a_idx]][route[b_idx + 1]]
        )
    else:
        delta = (
            D[route[a_idx - 1]][route[a_idx]] + D[route[a_idx]][route[b_idx]]
            + D[route[b_idx]][route[b_idx + 1]]
            - D[route[a_idx - 1]][route[b_idx]] - D[route[b_idx]][route[a_idx]]
            - D[route[a_idx]][route[b_idx + 1]]
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


    
    if n < 3:
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
    if p_idx_a + 1 == d_idx_a:
        in_out_cost_a_before = (
            D[route_a[p_idx_a - 1]][route_a[p_idx_a]]
            + D[route_a[p_idx_a]][route_a[d_idx_a]]
            + D[route_a[d_idx_a]][route_a[d_idx_a + 1]]
        )
        in_out_cost_a_after = (
            D[route_a[p_idx_a - 1]][route_b[p_idx_b]]
            + D[route_b[p_idx_b]][route_b[d_idx_b]]
            + D[route_b[d_idx_b]][route_a[d_idx_a + 1]]
        )
    else:
        in_out_cost_a_before = (
            D[route_a[p_idx_a - 1]][route_a[p_idx_a]] 
            + D[route_a[p_idx_a]][route_a[p_idx_a + 1]]
            + D[route_a[d_idx_a - 1]][route_a[d_idx_a]]
            + D[route_a[d_idx_a]][route_a[d_idx_a + 1]]
        )
        in_out_cost_a_after = (
            D[route_a[p_idx_a - 1]][route_b[p_idx_b]]
            + D[route_b[p_idx_b]][route_a[p_idx_a + 1]]
            + D[route_a[d_idx_a - 1]][route_b[d_idx_b]]
            + D[route_b[d_idx_b]][route_a[d_idx_a + 1]]
        )
    if p_idx_b + 1 == d_idx_b:
        in_out_cost_b_before = (
            D[route_b[p_idx_b - 1]][route_b[p_idx_b]]
            + D[route_b[p_idx_b]][route_b[d_idx_b]]
            + D[route_b[d_idx_b]][route_b[d_idx_b + 1]]
        )
        in_out_cost_b_after = (
            D[route_b[p_idx_b - 1]][route_a[p_idx_a]]
            + D[route_a[p_idx_a]][route_a[d_idx_a]]
            + D[route_a[d_idx_a]][route_b[d_idx_b + 1]]
        )
    else:
        in_out_cost_b_before = (
            D[route_b[p_idx_b - 1]][route_b[p_idx_b]] 
            + D[route_b[p_idx_b]][route_b[p_idx_b + 1]]
            + D[route_b[d_idx_b - 1]][route_b[d_idx_b]]
            + D[route_b[d_idx_b]][route_b[d_idx_b + 1]]
        )
        in_out_cost_b_after = (
            D[route_b[p_idx_b - 1]][route_a[p_idx_a]]
            + D[route_a[p_idx_a]][route_b[p_idx_b + 1]]
            + D[route_b[d_idx_b - 1]][route_a[d_idx_a]]
            + D[route_a[d_idx_a]][route_b[d_idx_b + 1]]
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

    
    if n_a < 3 or n_b < 3:
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
            i for i in range(1, n_a - 1)
            if prob.is_ppick(route_a[i]) or prob.is_parc_pick(route_a[i])
        ]
        pickup_indices_b = [
            j for j in range(1, n_b - 1)
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
    sol, info1 = iterative_greedy_balanced_solver(
        prob=prob,
        iterations=100000, time_limit=60.0, seed=42, verbose=False,
        destroy_proba=0.5, destroy_steps=min(6, prob.num_nodes // (2 * prob.K) + 1), destroy_T=1.0,
        rebuild_proba=0.25, rebuild_steps=2, rebuild_T=10.0
    )
    assert sol is not None, "No solution found by IG solver."
    if verbose:
        print(f"Initial solution cost: {sol.max_cost:.2f}")
        print()
        print()

    st1 = time.time()
    par = PartialSolution.from_solution(sol)
    new_par, modified, n_relocates = inter_swap_operator(
        partial=par,
        steps=None,
        mode='first',
        seed=100,
        verbose=False
    )

    sol = new_par.to_solution()
    assert sol, "No solution found after int."
    if verbose:
        print(f"Total inter-swap performed: {n_relocates}")
        print(f"Cost after inter-swap: {sol.max_cost:.2f}")
        print(f"Time for inter-swap: {time.time() - st1:.2f} seconds")
        print()
        print()

    st2 = time.time()
    par = PartialSolution.from_solution(sol)
    new_par, modified, n_relocates = intra_swap_operator(
        partial=par,
        steps=None,
        mode='best',
        seed=200,
        verbose=False
    )

    sol = new_par.to_solution()
    assert sol, "No solution found after relocates."
    if verbose:
        print(f"Total relocates performed: {n_relocates}")
        print(f"Cost after relocates: {sol.max_cost:.2f}")
        print(f"Time for relocates: {time.time() - st2:.2f} seconds")
        print()
        print()

    sol.stdin_print(verbose=False)


if __name__ == "__main__":
    main(verbose=False)


