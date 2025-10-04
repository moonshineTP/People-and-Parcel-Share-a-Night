from random import random

from share_a_ride.solvers.brute_force import brute_force_enumerate
from share_a_ride.solvers.greedy import greedy_balanced_solver
from utils.generators import generate_instance_coords

# ---------------------- Demo / main --------------------------------------------
def demo_generate_and_solve():
    random.seed(12345)
    typeI = []
    typeII = []
    params_I = [(2,2,2),(2,3,2),(3,2,2),(3,3,2),(2,2,3)]
    for idx,(N,M,K) in enumerate(params_I):
        prob = generate_instance_coords(N,M,K, area=15, seed=100+idx)
        typeI.append(prob)
    params_II = [(4,4,3),(4,5,3),(5,4,3),(5,5,3),(6,4,3)]
    for idx,(N,M,K) in enumerate(params_II):
        prob = generate_instance_coords(N,M,K, area=40, seed=200+idx)
        typeII.append(prob)

    print("Solving Type I tests with brute-force enumerator (exhaustive pair assignment + perms):\n")
    typeI_results = []
    for i,prob in enumerate(typeI):
        print(f"=== Type I test {i+1}: N={prob.N}, M={prob.M}, K={prob.K} ===")
        prob.pretty_print(verbose=0)
        sols, info = brute_force_enumerate(prob, max_solutions=2000, verbose=False, time_limit=15.0)
        print("Enumeration info:", info)
        if sols:
            best = sols[0]
            best.pretty_print()
        else:
            print("No solution enumerated within limits/time.")
        print("\n")
        typeI_results.append((sols, info))

    print("Type II tests: show greedy baseline (no exhaustive solve)\n")
    typeII_results = []
    for i,prob in enumerate(typeII):
        print(f"=== Type II test {i+1}: N={prob.N}, M={prob.M}, K={prob.K} ===")
        prob.pretty_print(verbose=0)
        greedy = greedy_balanced_solver(prob)
        greedy.pretty_print()
        typeII_results.append(greedy)
        print("\n")
    return typeI_results, typeII_results

if __name__ == "__main__":
    demo_generate_and_solve()