from typing import List

class ShareARideProblem:
    def __init__(self, N: int, M: int, K: int,
                parcel_qty: List[int], vehicle_caps: List[int],
                dist: List[List[int]]):
        
        # Basic parameters
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
        
        self.rev_ppick = lambda i: i
        self.rev_pdrop = lambda n: n - (N + M)
        self.rev_parc_pick = lambda n: n - N
        self.rev_parc_drop = lambda n: n - (2 * N + M)

        # Node check helpers
        self.is_ppick = lambda x: 1 <= x <= N
        self.is_pdrop = lambda x: N + M + 1 <= x <= 2 * N + M
        self.is_parc_pick = lambda x: N + 1 <= x <= N + M
        self.is_parc_drop = lambda x: 2 * N + M + 1 <= x <= 2 * (N + M)

    def copy(self):
        return ShareARideProblem(self.N, self.M, self.K,
            list(self.q), list(self.Q), [row[:] for row in self.D]
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