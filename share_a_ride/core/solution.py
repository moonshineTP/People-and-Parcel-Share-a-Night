"""
Module defining objects for representing solutions to the Share-a-Ride problem.
Includes:
- Solution: complete solution with K routes. Only use when it is assured that the
solution is complete and valid.
- PartialSolution: partial solution for constructive heuristics. 
Use in most algorithms as it allows incomplete solutions and saves route states.
- SolutionSwarm: collection of PartialSolution objects for population-based methods
"""

from __future__ import annotations

from typing import Optional, List, Tuple, TYPE_CHECKING

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.utils.helper import route_cost_from_sequence

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
                    idx = prob.rev_ppick(node)

                    # check
                    if idx in visited_pass_pick:
                        return False
                    if len(passenger_onboard) >= 1:
                        return False

                    # add
                    passenger_onboard.add(idx)
                    visited_pass_pick.add(idx)

                # passenger drop
                elif prob.is_pdrop(node):
                    idx = prob.rev_pdrop(node)

                    # check
                    if idx not in passenger_onboard:
                        return False

                    # remove
                    passenger_onboard.remove(idx)

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


    def stdin_print(self, verbose: bool = False):
        """
            Print the solution in the specified format.
            Verbose option for more details.
            - verbose=False: only print K, routes
            - verbose=True: print max cost and each route cost
        """
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


    def file_print(self, file_path: str):
        """
        Print solution to a .sol file in the TSPLIB format.
        Remember that TSPLIB .sol format exclude depot 1 in the route.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for i, route in enumerate(self.routes):
                route_str = ' '.join(map(str, route[1:-1]))
                f.write(f"Route #{i + 1}: {route_str}\n")

            f.write(f"Cost {self.max_cost}\n")


    def visualize(self, ax = None) -> None:
        """
        Visualize the solution routes on top of the problem instance.
        If no Axes provided, creates a new figure and its Axes
        You should import matplotlib.pyplot as plt before using this function.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for visualization.")
            return

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
    - route_states: list of dicts holding per-route state
        ({route, pos, cost, load, passenger, parcels, ended})
    - remaining_pass_pick: set of passenger ids whose pickup has not been scheduled
    - remaining_pass_drop: set of passenger ids picked but not dropped yet
    - remaining_parc_pick: set of parcel ids whose pickup has not been scheduled
    - remaining_parc_drop: set of parcel ids picked but not dropped yet
    """

    def __init__(
            self,
            problem: ShareARideProblem,
            routes: List[List[int]],
        ):
        """
        Initialize PartialSolution with problem and given route list.
        If routes are provided, validate them.
        Else, initialize K empty routes starting at depot 0.
        """

        # Initialize route and route costs
        self.problem = problem
        self.routes = self._init_routes(routes)
        self.route_costs = self._init_costs(routes)

        # Initialize other attributes
        self.max_cost = max(self.route_costs)
        self.avg_cost = sum(self.route_costs) / problem.K
        self.node_assignment = self._init_node_assignment()
        (   self.remaining_pass_pick, self.remaining_pass_drop, self.remaining_parc_pick,
            self.remaining_parc_drop, self.route_states
        ) = self._init_states()

        self.n_actions = sum(len(route) - 1 for route in self.routes)


    def _init_routes(self, routes):
        K = self.problem.K

        # Validating
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
        # Validating
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
        """
        Validate the PartialSolution for feasibility:
        - No route exceeds taxi capacity
        - No passenger or parcel is picked up more than once
        - No passenger or parcel is dropped before being picked up
        - All pickups and drops are within valid node indices
        """
        prob = self.problem
        N, M, K = prob.N, prob.M, prob.K

        # Check lengths
        if not len(self.routes) == len(self.route_states) == len(self.route_costs) == K:
            return False
        if len(self.node_assignment) != len(prob.D):
            return False

        # Initialize expected sets
        expected_pass_pick = set(range(1, N + 1))
        expected_pass_drop: set[int] = set()
        expected_parc_pick = set(range(1, M + 1))
        expected_parc_drop: set[int] = set()

        # Initialize assignment and counter
        node_assignment_check = [-1] * len(prob.D)
        total_actions = 0
        max_cost_check = 0
        cost_sum = 0

        # Iterate over routes
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

            # Iterate over nodes in the route
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

            # Update counters
            total_actions += len(route) - 1
            max_cost_check = max(max_cost_check, computed_cost)
            cost_sum += computed_cost


        # Check for expected storage sets
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
        """Check if two partial solutions encode the same action history.

        Routes are compared in a permutation-invariant way: we only care about
        the multiset of per-taxi action sequences, not about their ordering.
        """
        if self is other:
            return True

        # Quick consistency checks to avoid extra work.
        if self.problem is not other.problem:
            return False
        if self.n_actions != other.n_actions:
            return False

        # Remaining work must match regardless of taxi ordering
        def _canonical_node_assignment(
                ps: PartialSolution
            ) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]]:
            per_route: List[Tuple[int, ...]] = []
            unassigned: List[int] = []
            buckets: dict[int, List[int]] = {}

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

        # Formalize a lightweight route signature using only the first two actions.
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
        """
        Create a copy of the PartialSolution without re-running the constructor.
        """
        return PartialSolution(
            problem=self.problem,
            routes=[route.copy() for route in self.routes]
        )


    def stdin_print(self, verbose: bool = False):
        """
            Print the partial solution in the specified format.
            Verbose option for more details.
        """
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
        """
        Get all feasible actions for taxi t_idx in its current state.
        Each action is a tuple (kind, node_idx, inc_cost) where:
        - kind: 'pickP', 'dropP', 'pickL', 'dropL'
        - node_idx: idx of the passenger or parcel.
            + If passenger, it's the passenger idx (1..N)
            + If parcel, it's the parcel idx (1..M)
        - inc_cost: incremental cost of performing the action
        """

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


        actions.sort(key=lambda x: x[2])  # Sort by incremental cost
        return actions


    def apply_action(self, t_idx: int, kind: str, node_idx: int, inc: int) -> None:
        """
        Apply the specified action to taxi t_idx and update the PartialSolution state.
        - kind: 'pickP', 'dropP', 'pickL', 'dropL'
        - node_idx: idx of the passenger or parcel.
            + If passenger, it's the passenger idx (1..N)
            + If parcel, it's the parcel idx (1..M)
        - inc: incremental cost of performing the action
        """
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
        """
        Apply the action of returning taxi t_idx to the depot (node 0)
        and update the PartialSolution state.
        """
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
        state["route"].append(0)
        state["pos"] = 0
        state["actions"] += 1
        state["ended"] = True
        self.route_costs[t_idx] = state["cost"]
        self.max_cost = max(self.max_cost, state["cost"])
        self.avg_cost = sum(self.route_costs) / self.problem.K
        self.n_actions += 1


    def reverse_action(self, t_idx: int) -> None:
        """
        Reverse the last action taken by taxi t_idx and update the PartialSolution state.
        The reversed state is guaranteed to be valid by the design, so no extra checks are needed.
        """
        state = self.route_states[t_idx]

        if len(state["route"]) <= 1:
            raise ValueError(f"No actions to reverse for taxi {t_idx}.")

        # Update route state
        last_node = state["route"].pop()
        prev_node = state["route"][-1]
        dec_cost = self.problem.D[prev_node][last_node]
        state["cost"] -= dec_cost
        state["pos"] = prev_node
        state["actions"] -= 1
        state["ended"] = False


        # Update remaining pickups/drops and onboard loads
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
            # If last node is depot, just mark route as not ended
            state["ended"] = False

        # Update instance attributes
        self.route_costs[t_idx] = state["cost"]
        self.max_cost = max(self.route_costs)
        self.avg_cost = sum(self.route_costs) / self.problem.K
        self.node_assignment[last_node] = -1
        self.n_actions -= 1


    def is_complete(self) -> bool:
        """
        Check if all routes have ended at the depot.
        """
        return all(state["ended"] for state in self.route_states)


    def to_solution(self) -> Optional[Solution]:
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

        if solution is None or not solution.is_valid():
            print("Warning: Converted solution is not valid.")

        return solution


    @staticmethod
    def from_solution(sol: Solution) -> "PartialSolution":
        """
        Create a PartialSolution from a complete Solution, reusing __init__
        to recompute route costs and internal states.
        """
        routes_copy = [route.copy() for route in sol.routes]
        return PartialSolution(problem=sol.problem, routes=routes_copy)



class PartialSolutionSwarm:
    """
    SolutionSwarm representing a collection of solutions.
    Used for population-based metaheuristics.
    """

    def __init__(
            self,
            solutions: Optional[List[PartialSolution]] = None,
            n_partials: Optional[int] = None
        ):
        """
        Initialize SolutionSwarm with a list of Solution objects.
        """
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
        """
        Apply action to one PartialSolution in the swarm.
        Update swarm statistics accordingly.
        """
        parsol = self.parsol_list[sol_idx]
        parsol.apply_action(t_idx, kind, node_idx, inc)

        # Update statistics
        self.parsol_nact[sol_idx] = parsol.n_actions
        self.costs[sol_idx] = parsol.max_cost

        self.min_cost = min(self.costs)
        self.max_cost = max(self.costs)
        self.avg_cost = sum(self.costs) / len(self.costs)
        if parsol.max_cost == self.min_cost:
            self.best_parsol = parsol


    def apply_return_to_depot_one(self, sol_idx: int, t_idx: int):
        """
        Apply return to depot action to one PartialSolution in the swarm.
        Update swarm statistics accordingly.
        """
        parsol = self.parsol_list[sol_idx]
        parsol.apply_return_to_depot(t_idx)

        # Update statistics
        self.parsol_nact[sol_idx] = parsol.n_actions
        self.costs[sol_idx] = parsol.max_cost

        self.min_cost = min(self.costs)
        self.max_cost = max(self.costs)
        self.avg_cost = sum(self.costs) / len(self.costs)
        if parsol.max_cost == self.min_cost:
            self.best_parsol = parsol


    def copy(self):
        """
        Create a deep copy of the PartialSolutionSwarm.
        """
        copied_solutions = [sol.copy() for sol in self.parsol_list]
        return PartialSolutionSwarm(solutions=copied_solutions)


    def extract_best_solution(self) -> Optional[Solution]:
        """
        Extract the best complete Solution from the swarm if available.
        """
        if self.best_parsol and self.best_parsol.is_complete():
            return self.best_parsol.to_solution()

        return None
