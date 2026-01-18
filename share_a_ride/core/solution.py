"""
Module defining Solution, Partial Solution and Swarm objects
"""
from typing import Optional, List, Tuple, Dict

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.utils import route_cost_from_sequence




class Solution:
    """
    Solution object representing a complete solution to a Share-a-Ride problem.

    Attributes:
    - problem: ShareARideProblem instance
    - routes: list of lists of node indices (each route starts and ends with depot 0)
    - route_costs: list of costs for each route
    - num_actions: total number of actions across all routes
    - max_cost: maximal route cost among all routes
    """
    def __init__(self, problem: ShareARideProblem,
                routes: List[List[int]], route_costs: Optional[List[int]] = None):
        """
        Initialize Solution with problem, routes, and optional route costs.
        If route costs not provided, compute them from routes and problem distance matrix.
        """

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


    def is_valid(self, verbose: bool = False) -> bool:
        """
        Verify if the solution components describe a valid solution.
        The main constraints are:
        - coverage: all pickups and drops are included exactly once
        - precedence: pickups occur before drops, performed by a same vehicle,
                        pickup-drop of passenger are done consecutively
        - capacity: parcel load does not exceed vehicle capacity
        
        Args:
            verbose: If True, print detailed validation information
            
        Returns:
            bool: True if solution is valid, False otherwise
        """
        prob = self.problem
        K = prob.K      # pylint: disable=invalid-name

        if verbose:
            print(f"\n{'='*60}")
            print("SOLUTION VALIDATION")
            print(f"{'='*60}")
            print(f"Number of vehicles: {K}")
            print(f"Number of routes: {len(self.routes)}")

        if len(self.routes) != K:
            if verbose:
                print(f"✗ FAILED: Expected {K} routes, got {len(self.routes)}")
            return False

        if verbose:
            print("✓ Route count matches")

        # //// Iterate over routes
        visited_nodes = set()
        for route_idx, route in enumerate(self.routes):
            # Check depot at start and end
            if not (route[0] == 0 and route[-1] == 0):
                if verbose:
                    print(f"✗ FAILED: Vehicle {route_idx} route does not start/end at depot 0")
                    print(f"  Route: {route}")
                return False

            if verbose:
                print(f"\nVehicle {route_idx}: {' → '.join(map(str, route))}")

            # Tracking containers
            route_len = len(route)
            parcel_load = 0
            parcel_onboard = set()

            # Simulate over the route
            for idx, node in enumerate(route[1:-1], start=1):

                # Global coverage check
                if node in visited_nodes:
                    if verbose:
                        print(f"  ✗ FAILED: Node {node} visited multiple times")
                    return False
                visited_nodes.add(node)

                if prob.is_ppick(node):
                    pid = prob.rev_ppick(node)

                    # Check consecutivity
                    drop_node = prob.pserve(pid)[1]
                    if idx + 1 >= route_len or route[idx + 1] != drop_node:
                        if verbose:
                            print(f"  ✗ FAILED: Passenger {pid} pickup at {node} not followed by drop at {drop_node}")
                        return False
                    
                    if verbose:
                        print(f"  ✓ Passenger {pid}: {node} → {drop_node}")

                # passenger drop
                elif prob.is_pdrop(node):
                    pass        # Already handled in pickup

                # parcel pickup
                elif prob.is_lpick(node):
                    lid = prob.rev_lpick(node)

                    if lid in parcel_onboard:
                        if verbose:
                            print(f"  ✗ FAILED: Parcel {lid} picked up multiple times")
                        return False
                    
                    parcel_load += prob.q[lid - 1]
                    if parcel_load > prob.Q[route_idx]:
                        if verbose:
                            print(f"  ✗ FAILED: Vehicle {route_idx} capacity exceeded")
                            print(f"    Load before parcel {lid}: {parcel_load - prob.q[lid - 1]}")
                            print(f"    Parcel weight: {prob.q[lid - 1]}")
                            print(f"    Vehicle capacity: {prob.Q[route_idx]}")
                        return False

                    parcel_onboard.add(lid)
                    if verbose:
                        print(f"  ✓ Parcel {lid} pickup: load = {parcel_load}/{prob.Q[route_idx]}")

                # parcel drop
                elif prob.is_ldrop(node):
                    lid = prob.rev_ldrop(node)

                    if lid not in parcel_onboard:
                        if verbose:
                            print(f"  ✗ FAILED: Parcel {lid} dropped without pickup")
                        return False

                    if parcel_load - prob.q[lid - 1] < 0:
                        if verbose:
                            print(f"  ✗ FAILED: Negative load when dropping parcel {lid}")
                        return False
                    
                    parcel_load -= prob.q[lid - 1]
                    parcel_onboard.remove(lid)
                    if verbose:
                        print(f"  ✓ Parcel {lid} drop: load = {parcel_load}/{prob.Q[route_idx]}")

            # at end of route
            if parcel_load != 0:
                if verbose:
                    print(f"  ✗ FAILED: Vehicle {route_idx} has {parcel_load} load at end of route")
                return False
            
            if verbose:
                print(f"  ✓ Vehicle {route_idx} route valid (cost: {self.route_costs[route_idx]})")

        # Final coverage check
        if len(visited_nodes) != prob.num_requests * 2:
            if verbose:
                print("\n✗ FAILED: Coverage check failed")
                print(f"  Expected {prob.num_requests * 2} nodes, visited {len(visited_nodes)}")
                print(f"  Visited: {sorted(visited_nodes)}")
            return False

        if verbose:
            print(f"\n{'='*60}")
            print("[VALID] SOLUTION VALID")
            print(f"  Max route cost: {self.max_cost}")
            print(f"{'='*60}")
            print()

        return True


    def stdin_print(self, verbose: bool = False):
        """
            Print the solution in the specified format.
            Verbose option for more details.
        """
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
            import matplotlib.pyplot as plt     # pylint: disable=C0415
        except ImportError:
            print("matplotlib is required for visualization.")
            return

        # Check coordinates availability
        if self.problem.coords is None:
            print("No coordinates available for visualization.")
            return


        # //// Setting up the plot
        # Axes
        only_show = False
        if ax is None:
            only_show = True
            plt.figure(figsize=(6,6))
            ax = plt.gca()

        # Visualize problem first
        self.problem.visualize(ax)

        # Color map
        cmap = plt.get_cmap('tab10')

        # Iterate and draw routes
        for route_idx, route in enumerate(self.routes):
            route_color = cmap(route_idx % cmap.N)
            route_edge_style = {
                'arrowstyle'    : '->',
                'lw'            : 1.5,
                'linewidth'     : 1.0,
                'color'         : route_color,
                'alpha'         : 0.5
            }

            # Draw edges for this route
            for from_node, to_node in zip(route[:-1], route[1:]):
                x_from, y_from = self.problem.coords[from_node]
                x_to, y_to = self.problem.coords[to_node]

                # Draw arrow with defined style
                ax.annotate('', xy=(x_to, y_to), xytext=(x_from, y_from),
                        arrowprops=route_edge_style)

        # Finalize
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
    - route_costs: list of current route costs
    - max_cost: maximal route cost among current routes
    - node_assignment: list mapping each node to its assigned route
        following -1 for unassigned, 0..K-1 for assigned to route, depot 0 is undefined
    - route_states: list of dicts holding per-route state
        ({pos, parcels, load, actions, active, ended})
    - remaining_pass_serve: set of passenger ids not yet served
    - remaining_parc_pick: set of parcel ids not picked yet
    - remaining_parc_drop: set of parcel ids picked but not dropped yet
    """

    def __init__(
            self,
            problem: ShareARideProblem,
            routes: Optional[List[List[int]]] = None,
        ):
        """
        Initialize PartialSolution with problem and given route list.
        If routes are provided, validate them. If not, initialize empty routes.
        """
        # Initialize route and route costs
        self.problem = problem
        self.routes = self._init_routes(routes)
        self.route_costs = self._init_costs(routes)

        # Initialize stats
        self.min_cost = min(self.route_costs)
        self.max_cost = max(self.route_costs)
        self.avg_cost = sum(self.route_costs) / problem.K

        # Initialize detailed states
        self.node_assignment = self._init_node_assignment()
        self.remaining_pass_serve, self.remaining_parc_pick, self.remaining_parc_drop, \
        self.states, self.num_actions = self._init_states()


    def _init_routes(self, routes: Optional[List[List[int]]] = None):
        K = self.problem.K      # pylint: disable=C0103

        # Validating
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
        remaining_pass_serve = set(range(1, prob.N + 1))
        remaining_parc_pick = set(range(1, prob.M + 1))
        remaining_parc_drop = set()
        route_states = []
        num_actions = 0

        # //// Simulate over route
        for _, route in enumerate(self.routes):
            route_len = len(route)
            onboard_parcels = set()
            current_load = 0
            route_actions = 0

            # Simulate over the route
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

                    route_actions += 1

                    remaining_pass_serve.discard(pid)

                elif prob.is_pdrop(nodeid):
                    pass    # already handled in pickup

                elif prob.is_lpick(nodeid):
                    lid = prob.rev_lpick(nodeid)

                    onboard_parcels.add(lid)
                    current_load += prob.q[lid - 1]
                    route_actions += 1

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
                    route_actions += 1

                    remaining_parc_drop.discard(lid)

                else:   # depot
                    if nodeid != 0:
                        raise RuntimeError(
                            "Invalid route: node id out of range."
                        )
                    route_actions += 1

            # Finalize route state
            pos = route[-1]
            active = route_len > 1
            ended = active and route[-1] == 0
            state = {
                "pos": pos,                     # current position (node id)
                "parcels": onboard_parcels,     # current parcels onboard
                "load": current_load,           # current parcel load
                "actions": route_actions,       # number of actions taken (for this route)
                "active": active,               # whether taxi has started
                "ended": ended                  # whether taxi has returned to depot
            }
            route_states.append(state)

            # Update global counter of actions
            num_actions += route_actions

        # //// Return initialized components
        return (
            remaining_pass_serve,
            remaining_parc_pick,
            remaining_parc_drop,
            route_states,
            num_actions
        )


    def is_valid(self, verbose: bool = False) -> bool:
        """
        Validate the PartialSolution for feasibility:
        - No route exceeds taxi capacity
        - No passenger or parcel is picked up more than once
        - No passenger or parcel is dropped before being picked up
        - Passenger pickup and drop must be consecutive
        - All pickups and drops are within valid node indices
        """
        prob = self.problem
        N, M, K = prob.N, prob.M, prob.K    # pylint: disable=C0103

        # Check lengths
        if not len(self.routes) == len(self.states) == len(self.route_costs) == K:
            if verbose:
                print("Invalid: Mismatch in number of routes, states, or costs.")
            return False
        if len(self.node_assignment) != len(prob.D):
            if verbose:
                print("Invalid: Mismatch in node assignment length.")
            return False

        # Initialize expected sets
        expected_pass_serve = set(range(1, N + 1))
        expected_parc_pick = set(range(1, M + 1))
        expected_parc_drop: set[int] = set()

        # Initialize assignment and counter
        node_assignment_check = [-1] * len(prob.D)
        total_actions = 0
        max_cost_check = 0
        cost_sum = 0


        # //// Iterate and validate over routes
        for t_idx in range(K):
            route = self.routes[t_idx]
            route_len = len(route)
            state = self.states[t_idx]

            # Basic validity checks
            if not route or route[0] != 0:
                if verbose:
                    print(f"Invalid: Route {t_idx} does not start with depot 0.")
                return False

            is_active = route_len > 1
            if state["active"] != is_active:
                if verbose:
                    print(f"Invalid: Active state mismatch for taxi {t_idx}.")
                return False

            is_ended = route_len > 1 and route[-1] == 0
            if state["ended"] != is_ended:
                if verbose:
                    print(f"Invalid: Ended state mismatch for taxi {t_idx}.")
                return False


            # //// Iterate over nodes in the route
            parcel_onboard: set[int] = set()
            route_len = len(route)
            load = 0
            prev = route[0]
            cumulated_cost = 0

            for idx, node in enumerate(route[1:], start=1):
                # Check node range
                if not 0 <= node < prob.num_nodes:
                    if verbose:
                        print(f"Invalid: Node {node} out of range in route {t_idx}.")
                    return False

                # Check node assignment for non-depot nodes
                if node != 0:
                    assigned = node_assignment_check[node]
                    if assigned not in (-1, t_idx):
                        if verbose:
                            print(
                                f"Invalid: Node {node} assigned to an unintended route "
                                f"{assigned} instead of {t_idx}."
                            )
                        return False

                    # Update node assignment
                    node_assignment_check[node] = t_idx

                # Update cost
                cumulated_cost += prob.D[prev][node]
                prev = node


                # //// Validate node type and update state accordingly
                # Passenger nodes
                if prob.is_ppick(node):
                    pid = prob.rev_ppick(node)

                    # Check consecutivity
                    # If not last node, next node must be drop
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

                    # Update
                    total_actions += 1
                    expected_pass_serve.discard(pid)

                # Passenger drop
                elif prob.is_pdrop(node):
                    pass

                # Parcel pickup
                elif prob.is_lpick(node):
                    lid = prob.rev_lpick(node)

                    # Check onboard
                    if lid in parcel_onboard:
                        if verbose:
                            print(
                                f"Invalid: Parcel {lid} picked up multiple times "
                                f"in route {t_idx}."
                            )
                        return False

                    # Check capacity
                    load += prob.q[lid - 1]
                    if load > prob.Q[t_idx]:
                        if verbose:
                            print(
                                f"Invalid: Taxi {t_idx} load {load} exceeds capacity "
                                f"{prob.Q[t_idx]} after picking parcel {lid} in index {idx}."
                            )
                        return False

                    # Update
                    total_actions += 1
                    parcel_onboard.add(lid)
                    expected_parc_pick.discard(lid)
                    expected_parc_drop.add(lid)

                # Parcel drop
                elif prob.is_ldrop(node):
                    lid = prob.rev_ldrop(node)

                    # Check onboard
                    if lid not in parcel_onboard:
                        if verbose:
                            print(
                                f"Invalid: Parcel {lid} dropped without being picked up "
                                f"in route {t_idx}."
                            )
                        return False

                    # Check capacity
                    load -= prob.q[lid - 1]
                    if load < 0:
                        if verbose:
                            print(
                                f"Invalid: Taxi {t_idx} has negative load "
                                f"after dropping parcel {lid}."
                            )
                        return False

                    # Update
                    total_actions += 1
                    parcel_onboard.remove(lid)
                    expected_parc_drop.discard(lid)

                else:   # Depot node
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

            # Final state checks
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

            # Update counter
            max_cost_check = max(max_cost_check, cumulated_cost)
            cost_sum += cumulated_cost


        # //// Final checks
        # Expected sets check
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

        # Expected statistics check
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
        """
        Check if the partial solution is not complete yet (lightweight).
        """
        return self.num_actions < self.problem.num_actions


    def is_identical(self, other: "PartialSolution") -> bool:
        """
        Check if two partial solutions encode the same action history.
        Routes are compared in a permutation-invariant way: we only care about
        the multiset of per-taxi action sequences, not about their ordering.
        """
        if self is other:
            return True

        if self.problem is not other.problem or self.num_actions != other.num_actions:
            return False

        return sorted(tuple(r[:3]) for r in self.routes) == sorted(tuple(r[:3]) for r in other.routes)


    def copy(self):
        """
        Create a copy of the PartialSolution without re-running the constructor.
        """
        return PartialSolution(
            problem=self.problem,
            routes=[route.copy() for route in self.routes] if self.routes else None
        )


    def stdin_print(self, verbose: bool = False):
        """
        Print the partial solution in the specified format.
        Verbose option for more details.
        """
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


    # //// Actions and state updates API ////
    def enumerate_action_nodes(self, route_idx: int) -> List[Tuple[int, int]]:
        """
        Enumerate all action nodes (including depot) in taxi ``route_idx`` route
        """
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
        """
        Decrease the cost of taxi route_idx by dec.
        Also updates max, min, and avg costs.
        No validations are performed.
        """
        self.route_costs[route_idx] -= dec
        self.max_cost = max(self.route_costs)
        self.min_cost = min(self.min_cost, self.route_costs[route_idx])
        self.avg_cost = sum(self.route_costs) / self.problem.K


    def possible_expand(self, t_idx: int) -> List[Tuple[str, int, int]]:
        """
        Get all feasible expand actions for taxi t_idx in its current state.

        Params:
        - t_idx: index of the taxi to expand

        Returns: a list of possible actions. Each action is a tuple (kind, actid, costinc) where:
        - kind: 'serveP', 'pickL', 'dropL'
        - actid: idx of the passenger or parcel.
            + If passenger, it's the passenger idx (1..N)
            + If parcel, it's the parcel idx (1..M)
        - costinc: incremental cost of performing the action
        """

        state = self.states[t_idx]
        if state["ended"]:
            return []

        # Extracts variables
        prob = self.problem
        pos = state["pos"]
        expansions: List[Tuple[str, int, int]] = []

        # serveP
        for pid in self.remaining_pass_serve:
            pick, drop = prob.pserve(pid)
            inc = prob.D[pos][pick] + prob.D[pick][drop]
            expansions.append(("serveP", pid, inc))

        # pickL
        for lid in self.remaining_parc_pick:
            parcel_weight = prob.q[lid - 1]
            if state["load"] + parcel_weight <= prob.Q[t_idx]:
                inc = prob.D[pos][prob.lpick(lid)]
                expansions.append(("pickL", lid, inc))

        # dropL
        for lid in state["parcels"]:
            inc = prob.D[pos][prob.ldrop(lid)]
            expansions.append(("dropL", lid, inc))

        # Sort actions by incremental cost
        expansions.sort(key=lambda x: x[2])
        return expansions


    def check_expand(self, route_idx: int, kind: str, actid: int) -> bool:
        """
        Check if the specified action (not return) is valid for taxi t_idx.
        Validates precedence, coverage, capacity, and consecutivity constraints.

        Args:
            - t_idx: taxi index
            - kind: 'serveP', 'pickL', 'dropL', or 'return'
            - node_idx: idx of the passenger or parcel (0 for return)

        Returns:
            - True if action is valid, False otherwise.
        """
        state = self.states[route_idx]
        prob = self.problem

        # Taxi must not ended
        if state["ended"]:
            return False

        # Validate action kinds
        if kind == "serveP":
            return actid in self.remaining_pass_serve
        if kind == "pickL":
            return (
                actid in self.remaining_parc_pick        # not picked yet
                and state["load"] + prob.q[actid - 1] <= prob.Q[route_idx]   # enough capacity
            )
        if kind == "dropL":
            return actid in state["parcels"]    # parcel must be onboard

        # Unknown action kind
        raise ValueError(f"Unknown action kind: {kind}")


    def check_return(self, route_idx: int) -> bool:
        """Check if taxi t_idx can return to depot (node 0) in its current state."""
        state = self.states[route_idx]

        return not state["ended"] and not state["parcels"]


    def apply_extend(self, route_idx: int, kind: str, actid: int, inc: int) -> None:
        """
        Apply the specified action to taxi t_idx and update the PartialSolution state.
        
        Params:
        - route_idx: index of the taxi performing the action
        - kind: type of action ('serveP', 'pickL', 'dropL')
        - actid: id of the passenger or parcel involved in the action
        - costinc: incremental cost of performing the action
        """
        route = self.routes[route_idx]
        state = self.states[route_idx]
        prob = self.problem

        # Basic validations
        if state["ended"]:
            raise ValueError(f"Cannot apply action on ended route {route_idx}.")

        # Mark taxi as active on first action
        state["active"] = True


        # //// Apply action based on kind
        if kind == "serveP":
            pick_node, drop_node = prob.pserve(actid)

            # Update assignments
            route.append(pick_node)
            route.append(drop_node)
            self.node_assignment[pick_node] = route_idx
            self.node_assignment[drop_node] = route_idx

            # Update state
            self.remaining_pass_serve.discard(actid)
            state["pos"] = drop_node
            state["actions"] += 1

            # Update stats
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

        # Update state
        state["pos"] = node
        state["actions"] += 1
        route.append(node)
        self.node_assignment[node] = route_idx

        # Update stats
        self.route_costs[route_idx] += inc
        self.max_cost = max(self.max_cost, self.route_costs[route_idx])
        self.min_cost = min(self.route_costs)
        self.avg_cost = sum(self.route_costs) / self.problem.K
        self.num_actions += 1


    def apply_return(self, t_idx: int) -> None:
        """
        Apply the action of returning taxi t_idx to the depot (node 0)
        and update the PartialSolution state.
        """
        route = self.routes[t_idx]
        state = self.states[t_idx]

        # Validity
        if state["ended"]:
            return
        if state["pos"] == 0 and state["actions"] > 0:
            state["ended"] = True
            return
        if state["parcels"]:
            raise ValueError(f"Taxi {t_idx} must drop all loads before returning to depot.")

        # Update state
        inc = self.problem.D[state["pos"]][0]
        route.append(0)
        state["pos"] = 0
        state["actions"] += 1
        state["ended"] = True

        # Update stats
        self.route_costs[t_idx] += inc
        self.max_cost = max(self.max_cost, self.route_costs[t_idx])
        self.min_cost = min(self.route_costs)
        self.avg_cost = sum(self.route_costs) / self.problem.K
        self.num_actions += 1


    def reverse_action(self, t_idx: int) -> None:
        """
        Reverse the last action taken by taxi t_idx and update the PartialSolution state.
        The reversed state is guaranteed to be valid by the design, so no extra checks are needed.
        """
        route = self.routes[t_idx]
        state = self.states[t_idx]
        if len(route) <= 1:
            raise ValueError(f"No actions to reverse for taxi {t_idx}.")

        prob = self.problem
        last_node = route[-1]
        # Check if it's a passenger drop (part of serveP)
        if prob.is_pdrop(last_node):
            # It must be a serveP action, so we need to pop two nodes
            drop_node = route.pop()
            pick_node = route.pop()

            pid = prob.rev_pdrop(drop_node)
            if prob.rev_ppick(pick_node) != pid:
                raise ValueError(
                        "Inconsistent route state: "
                        "pdrop not preceded by corresponding ppick."
                    )

            # Calculate cost decrement
            prev_node = route[-1]
            dec = prob.D[prev_node][pick_node] + prob.D[pick_node][drop_node]

            # Update state
            state["pos"] = prev_node
            state["actions"] -= 1
            state["ended"] = False
            if len(route) == 1:
                state["active"] = False

            # Update containers
            self.remaining_pass_serve.add(pid)
            self.node_assignment[drop_node] = -1
            self.node_assignment[pick_node] = -1

            # Update instance stats
            self.route_costs[t_idx] -= dec
            self.max_cost = max(self.route_costs)
            self.min_cost = min(self.route_costs)
            self.avg_cost = sum(self.route_costs) / self.problem.K
            self.num_actions -= 1

            return

        # Normal single node reversal (pickL, dropL, or return)
        last_node = route.pop()
        prev_node = route[-1]
        dec = prob.D[prev_node][last_node]

        # Update state
        state["pos"] = prev_node
        state["actions"] -= 1
        state["ended"] = False
        if len(route) == 1:
            state["active"] = False

        # Update remaining pickups/drops and onboard loads
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

        elif last_node == 0:    # Returned to depot
            pass

        else:
            # Should not happen
            raise ValueError(f"Unexpected node type to reverse: {last_node}")

        # Update stats
        self.route_costs[t_idx] -= dec
        self.max_cost = max(self.route_costs)
        self.min_cost = min(self.route_costs)
        self.avg_cost = sum(self.route_costs) / self.problem.K
        self.node_assignment[last_node] = -1
        self.num_actions -= 1


    def is_completed(self, verbose=False) -> bool:
        """
        Check if all routes have ended at the depot.
        Because the code works well and safe, we should only implement a lightweight check.
        """
        if self.num_actions != self.problem.num_actions:
            if verbose:
                print(
                    f"Not completed: current partial actions {self.num_actions} "
                    f"does not suffice total actions {self.problem.num_actions}."
                )
            return False

        return True


    def to_solution(self) -> Optional[Solution]:
        """Convert the PartialSolution to a full Solution if complete and valid."""
        if not self.is_completed(verbose=True):
            print("Warning: Solution is not complete, cannot convert.")
            return None

        solution = Solution(
            problem=self.problem,
            routes=self.routes,
            route_costs=self.route_costs
        )

        return solution


    @staticmethod
    def from_solution(sol: Solution) -> "PartialSolution":
        """Create a PartialSolution from a complete Solution"""
        routes_copy = [route.copy() for route in sol.routes]
        return PartialSolution(problem=sol.problem, routes=routes_copy)




class PartialSolutionSwarm:
    """
    SolutionSwarm representing a collection of solutions.

    Attributes:
    - problem: ShareARideProblem instance
    - num_partials: number of partial solutions in the swarm
    - partial_lists: list of PartialSolution objects
    - partial_num_actions: list of number of actions taken in each partial solution
    - costs: list of maximal route costs for each partial solution
    - min_cost: minimal maximal route cost among partial solutions
    - max_cost: maximal maximal route cost among partial solutions
    - avg_cost: average maximal route cost among partial solutions
    """
    def __init__(
            self,
            solutions: Optional[List[PartialSolution]],
        ):
        """Initialize SolutionSwarm with a list of Solution objects."""

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
        """Update swarm statistics after modifications to partial solutions."""
        self.partial_num_actions = [sol.num_actions for sol in self.partial_lists]
        self.partial_costs = [sol.max_cost for sol in self.partial_lists]
        self.min_cost = min(self.partial_costs)
        self.max_cost = max(self.partial_costs)
        self.avg_cost = sum(sol.max_cost for sol in self.partial_lists) / len(self.partial_lists)


    def opt(self) -> Optional[Solution]:
        """Extract the best complete Solution from the swarm if available."""
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
        """Get statistics of the swarm as a dictionary."""
        self.update()

        return {
            "num_partials": self.num_partials,
            "min_cost": self.min_cost,
            "max_cost": self.max_cost,
            "avg_cost": self.avg_cost
        }


    def copy(self):
        """Create a deep copy of the PartialSolutionSwarm."""
        copied_solutions = [sol.copy() for sol in self.partial_lists]
        return PartialSolutionSwarm(solutions=copied_solutions)
