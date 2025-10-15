from typing import List

import matplotlib.pyplot as plt 

from share_a_ride.problem import ShareARideProblem
from share_a_ride.utils.generator import generate_instance_coords

class Solution:
    """
    Solution object representing K routes.
    - routes: list of lists of node indices (each route includes depot 0 at start and end)
    - route_lengths: list of ints
    - max_length: int (objective to minimize)
    """
    def __init__(self, problem: ShareARideProblem,
                routes: List[List[int]], route_lengths: List[int]):

        if problem is None:
            raise ValueError("Problem instance cannot be None.")

        assert len(routes) == len(route_lengths)
        self.problem = problem
        self.routes = routes
        self.route_costs = route_lengths
        self.max_cost = max(route_lengths) if route_lengths else 0


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


    def visualize(self, ax: plt.Axes = None) -> None:
        """
        Visualize the solution routes on top of the problem instance.
        If no Axes provided, creates a new figure and its Axes
        """
        if self.problem.coords is None:
            print("No coordinates available for visualization.")
            return
        
        if ax is None:
            plt.figure(figsize=(6,6))
            ax = plt.gca()

        # First visualize the problem instance (nodes)
        self.problem.visualize(ax)

        # Use matplotlib's built-in color palette
        cmap = plt.get_cmap('tab10')

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


if __name__ == "__main__":
    from share_a_ride.data.parser import parse_sarp_to_problem
    from share_a_ride.data.router import path_router
    from share_a_ride.solvers.algo.greedy import greedy_balanced_solver
    from share_a_ride.solvers.algo.bnb import branch_and_bound_solver
    from share_a_ride.problem import ShareARideProblem

    # Generate a small test instance
    in_path = path_router("H", "H-n6-m5-k3", "read")
    prob = parse_sarp_to_problem(in_path)

    print("Problem instance:")
    prob.pretty_print(verbose=0)

    # Create a sample solution (you would normally get this from a solver)
    # Example: Simple routes that visit all nodes
    sol, info1 = greedy_balanced_solver(prob, verbose=0)
    print(sol.max_cost)
    sol, info2 = branch_and_bound_solver(prob, target_cost=sol.max_cost, time_limit=20.0)

    print("\nSolution:")
    sol.stdin_print(verbose=1)

    plt.figure(figsize=(6,6))
    ax = plt.gca()
    sol.visualize(ax)
    plt.show()

    out_path = path_router("H", "H-n10-m10-k5", "solve", "bnb")
    sol.file_print(out_path)
        