from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING

from share_a_ride.problem import ShareARideProblem
from share_a_ride.solvers.operator import Operator
from share_a_ride.utils.helper import route_cost_from_sequence

if TYPE_CHECKING:
    import matplotlib.pyplot as plt



class Solution:
    """
    Solution object representing K routes.
    - routes: list of lists of node indices (each route includes depot 0 at start and end)
    - route_lengths: list of ints
    - max_length: int (objective to minimize)
    """
    def __init__(self, problem: ShareARideProblem,
                routes: List[List[int]], route_costs: Optional[List[int]] = None):

        if problem is None:
            raise ValueError("Problem instance cannot be None.")

        assert len(routes) == len(route_costs)
        self.problem = problem
        self.routes = routes
        self.route_costs = route_costs
        if route_costs is None:
            route_costs = [
                route_cost_from_sequence(route, problem.D) 
                for route in routes
            ]
        self.max_cost = max(route_costs) if route_costs else 0


    def is_valid(self) -> bool:
        """
            Check depot, precedence, and capacity constraints.
        """

        prob = self.problem
        N, M, K = prob.N, prob.M, prob.K
        if len(self.routes) != K:
            return False

        for route_idx, route in enumerate(self.routes):
            # must start and end at depot
            if not (route[0] == 0 and route[-1] == 0):
                return False

            # track passengers and parcels
            passenger_onboard = set()
            parcel_onboard = set()
            parcel_load = 0

            visited_pass_pick = set()
            visited_parc_pick = set()

            for node in route[1:-1]:

                # passenger pickup
                if prob.is_ppick(node):
                    id = prob.rev_ppick(node)

                    # check
                    if id in visited_pass_pick:
                        return False
                    if len(passenger_onboard) >= 1:
                        return False

                    # add
                    passenger_onboard.add(id)
                    visited_pass_pick.add(id)

                # passenger drop
                elif prob.is_pdrop(node):
                    id = prob.rev_pdrop(node)

                    # check
                    if id not in passenger_onboard:
                        return False

                    # remove
                    passenger_onboard.remove(id)

                # parcel pickup
                elif prob.is_parc_pick(node):
                    jd = prob.rev_parc_pick(node)

                    # check
                    if jd in visited_parc_pick or jd in parcel_onboard:
                        return False
                    parcel_load += prob.q[jd - 1]
                    if parcel_load > prob.Q[route_idx]:
                        return False

                    # add
                    visited_parc_pick.add(jd)
                    parcel_onboard.add(jd)

                # parcel drop
                elif prob.is_parc_drop(node):
                    jd = prob.rev_parc_drop(node)

                    # check
                    if jd not in parcel_onboard:
                        return False

                    # remove
                    assert parcel_load - prob.q[jd - 1] >= 0
                    parcel_load -= prob.q[jd - 1]
                    parcel_onboard.remove(jd)

            # at end of route
            if passenger_onboard:
                return False
            if parcel_load != 0:
                return False

        return True


    def stdin_print(self, verbose: int = 0):
        """
            Print the solution in the specified format.
            Verbose option for more details.
            - verbose=0: for online judge
            - verbose=1: detailed route info
        """
        if verbose:
            print(f"*** Max route cost: {self.max_cost} ***")

        print(self.problem.K)

        for route, cost in zip(self.routes, self.route_costs):
            if verbose:
                print(f"- Route cost: {cost}")
            print(len(route))
            print(" ".join(map(str, route)))


    def file_print(self, file_path: str):
        """
        Print solution to a .sol file in the TSPLIB format.
        Remember that TSPLIB .sol format exclude depot 1 in the route.
        """
        with open(file_path, 'w') as f:
            for i, route in enumerate(self.routes):
                route_str = ' '.join(map(str, route[1:-1]))
                f.write(f"Route #{i + 1}: {route_str}\n")

            f.write(f"Cost {self.max_cost}\n")


    def visualize(self, ax: plt.Axes) -> None:
        """
        Visualize the solution routes on top of the problem instance.
        If no Axes provided, creates a new figure and its Axes
        You should import matplotlib.pyplot as plt before using this function.
        """
        if self.problem.coords is None:
            print("No coordinates available for visualization.")
            return

        only_show = False
        if ax is None:
            only_show = True
            plt.figure(figsize=(6,6))
            ax = plt.gca()

        # Use matplotlib's built-in color palette
        cmap = plt.get_cmap('tab10')

        # First visualize the problem instance (nodes)
        self.problem.visualize(ax)

        # Draw routes
        for route_idx, route in enumerate(self.routes):
            route_color = cmap(route_idx % cmap.N)

            # Define style for this route's edges
            route_edge_style = {
                'arrowstyle'    : '->',
                'lw'            : 1.5,
                'linewidth'     : 1.0,
                'color'         : route_color,
                'alpha'         : 0.5
            }

            # Draw edges for this route
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                x_from, y_from = self.problem.coords[from_node]
                x_to, y_to = self.problem.coords[to_node]

                # Draw arrow with defined style
                ax.annotate('', xy=(x_to, y_to), xytext=(x_from, y_from), 
                        arrowprops=route_edge_style)

        # Add route information to title
        ax.set_title(f"Share-a-Ride Solution (Max Cost: {self.max_cost})")
        if only_show:
            plt.show()



class PartialSolution:
    """
    PartialSolution object representing a partial assignment of nodes to K routes.
    Used for constructive heuristics and branch-and-bound algorithms.

    Attributes:
    - problem: ShareARideProblem instance
    - routes: list of lists of node indices (partial routes, always start with depot 0)
    - premature_routes: copy of the current routes to resume greedy extensions
    - route_costs: list of current route costs
    - max_cost: maximal route cost among current routes
    - node_assignment: list mapping each node to its assigned route
        following -1 for unassigned, 0..K-1 for assigned to route, depot 0 is undefined
    - route_states / taxi_states: list of dicts holding per-route state
        ({route, pos, cost, load, passenger, parcels, ended})
    - remaining_pass_pick: set of passenger ids whose pickup has not been scheduled
    - remaining_pass_drop: set of passenger ids picked but not dropped yet
    - remaining_parc_pick: set of parcel ids whose pickup has not been scheduled
    - remaining_parc_drop: set of parcel ids picked but not dropped yet
    """

    def __init__(
            self,
            problem: ShareARideProblem,
            routes: List[List[int]] = [],
        ):
        """
        Initialize PartialSolution with problem and given route list.
        If routes are provided, validate them.
        Else, initialize K empty routes starting at depot 0.
        """

        if problem is None:
            raise ValueError("Problem instance cannot be None.")

        # Initialize route and route costs
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

        # Validating
        if routes is None:
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
        # Validating
        if routes is None:
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

        # Initialize remaining pickups and drops, and taxi states
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
    

    def possible_actions(self, t_idx: int) -> List[tuple[str, int, int]]:
        state = self.route_states[t_idx]
        if state["ended"]:
            return []

        prob = self.problem
        pos = state["pos"]
        actions: List[tuple[str, int, int]] = []

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

        # Check if route already ended
        if state["ended"]:
            return
        if state["pos"] == 0 and len(state["route"]) > 1:
            state["ended"] = True
            return

        # Ensure taxi is empty before returning to depot
        if state["passenger"] != 0 or state["parcels"]:
            raise ValueError(f"Taxi {t_idx} must drop all loads before returning to depot.")

        # Update state to end route
        state["cost"] += self.problem.D[state["pos"]][0]
        self.route_costs[t_idx] = state["cost"]
        self.max_cost = max(self.max_cost, state["cost"])
        state["route"].append(0)
        state["pos"] = 0
        state["ended"] = True
        


    def is_complete(self) -> bool:
        return all(state["ended"] for state in self.route_states)


    def to_solution(self) -> Solution:
        """
        Convert the PartialSolution to a full Solution if complete.
        """
        if not self.is_complete():
            print("Cannot convert to Solution: not all routes have ended at depot.")
            return None

        solution = Solution(
            problem=self.problem,
            routes=self.routes,
            route_costs=self.route_costs
        )
        
        if not solution.is_valid():
            print("Warning: Converted solution is not valid.")

        return solution

    def operate(self, op: Operator):
        pass



class SolutionSwarm:
    """
    SolutionSwarm representing a collection of solutions.
    Used for population-based metaheuristics.
    """

    def __init__(self, solutions: List[Solution]):
        """
        Initialize SolutionSwarm with a list of Solution objects.
        """
        if not solutions:
            raise ValueError("Solutions list cannot be empty.")
        self.solutions = solutions
        self.min_cost = min(sol.max_cost for sol in solutions)
        self.best_solution = min(solutions, key=lambda s: s.max_cost)
        self.avg_cost = sum(sol.max_cost for sol in solutions) / len(solutions)

    def operate(self, op: Operator):
        pass

        