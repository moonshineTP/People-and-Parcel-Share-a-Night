from typing import List
from utils.helper import route_cost_from_sequence

class Solution:
    """
    Solution object representing K routes.
    - routes: list of lists of node indices (each route includes depot 0 at start and end)
    - route_lengths: list of ints
    - max_length: int (objective to minimize)
    """
    def __init__(self, problem: "ShareARideProblem",
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

    def pretty_print(self, verbose: int = 0):
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

        