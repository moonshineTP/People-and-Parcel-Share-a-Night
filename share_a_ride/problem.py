from typing import List

class ShareARideProblem:
    def __init__(self, N: int, M: int, K: int,
                 parcel_qty: List[int], vehicle_caps: List[int],
                 dist: List[List[int]]):
        self.N = N
        self.M = M
        self.K = K
        self.q = list(parcel_qty)
        self.Q = list(vehicle_caps)
        self.D = [row[:] for row in dist]
        self.num_nodes = 2*N + 2*M + 1

        # index helpers
        self.ppick = lambda i: i
        self.pdrop = lambda i: N + M + i
        self.parc_pick = lambda j: N + j
        self.parc_drop = lambda j: 2*N + M + j

    def copy(self):
        return ShareARideProblem(
            self.N, self.M, self.K,
            list(self.q), list(self.Q),
            [row[:] for row in self.D]
        )

    def pretty_print(self, verbose: int = 0):
        print(f"Share-a-Ride: N={self.N} passengers, M={self.M} parcels, "
              f"K={self.K}, num_nodes={self.num_nodes}")
        if verbose >= 1:
            print("Parcel quantities (q):", self.q)
            print("Vehicle capacities (Q):", self.Q)
            print("Distance matrix D:")
            for row in self.D:
                print(" ", row)

    # @staticmethod
    # def from_terminal() -> "ShareARideProblem":
    #     """Read problem instance from stdin (for Hustack submission)."""
    #     N, M, K = map(int, input().split())
    #     q = list(map(int, input().split()))
    #     Q = list(map(int, input().split()))
    #     D = [list(map(int, input().split())) for _ in range(2 * N + 2 * M + 1)]
    #     return ShareARideProblem(N, M, K, q, Q, D)