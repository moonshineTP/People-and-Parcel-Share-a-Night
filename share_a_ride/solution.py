from typing import List
from problem import ShareARideProblem

class Solution:
    """
    Solution object representing K routes.
    - routes: list of lists of node indices (each route includes depot 0 at start and end)
    - route_lengths: list of ints
    - max_length: int (objective to minimize)
    """
    def __init__(self, problem: "ShareARideProblem",
                 routes: List[List[int]], route_lengths: List[int]):
        assert len(routes) == len(route_lengths)
        self.problem = problem
        self.routes = routes
        self.route_lengths = route_lengths
        self.max_length = max(route_lengths) if route_lengths else 0

    def is_valid(self) -> bool:
        """Check depot, precedence, and capacity constraints."""
        prob = self.problem
        N, M, K = prob.N, prob.M, prob.K

        for r_idx, route in enumerate(self.routes):
            # must start and end at depot
            if not (route[0] == 0 and route[-1] == 0):
                return False

            # track passengers and parcels
            passenger_onboard = set()
            parcel_load = 0

            visited_pass_pick = set()
            visited_parc_pick = set()

            for node in route[1:-1]:  # exclude depots
                # passenger pickup
                for i in range(1, N + 1):
                    if node == prob.ppick(i):
                        if i in visited_pass_pick:
                            return False
                        if len(passenger_onboard) >= 1:
                            return False  # taxi already has a passenger
                        passenger_onboard.add(i)
                        visited_pass_pick.add(i)

                    if node == prob.pdrop(i):
                        if i not in passenger_onboard:
                            return False
                        passenger_onboard.remove(i)

                # parcel pickup
                for j in range(1, M + 1):
                    if node == prob.parc_pick(j):
                        if j in visited_parc_pick:
                            return False
                        parcel_load += prob.q[j - 1]
                        if parcel_load > prob.Q[r_idx]:
                            return False
                        visited_parc_pick.add(j)

                    if node == prob.parc_drop(j):
                        if j not in visited_parc_pick:
                            return False
                        parcel_load -= prob.q[j - 1]

            # at end of route
            if passenger_onboard:
                return False
            if parcel_load != 0:
                return False

        return True