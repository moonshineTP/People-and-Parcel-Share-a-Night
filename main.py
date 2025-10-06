import random

from share_a_ride.solvers.exhaustive import exhaustive_enumerate
from share_a_ride.solvers.bnb import branch_and_bound
from utils.generators import generate_instance_coords

# ---------------------- Demo / main --------------------------------------------
def demo_sprint_1():
    random.seed(12345)

    # Prepare type I test instances
    type_I = []
    params_I = [(2,2,2), (2,3,2), (3,2,2), (3,3,2), (2,2,3)]
    for idx, (N, M, K) in enumerate(params_I):
        prob = generate_instance_coords(N, M, K, area=15, seed=100+idx)
        type_I.append(prob)

    # Prepare type II test instances
    type_II = []
    params_II = [(4,4,3), (4,5,3), (5,4,3), (5,5,2), (6,4,2)]
    for idx, (N, M, K) in enumerate(params_II):
        prob = generate_instance_coords(N, M, K, area=50, seed=200+idx)
        type_II.append(prob)

    # Solve Type I
    print("Solving Type I tests with brute-force enumerator:\n")
    type_I_results = []
    for prob in type_I:
        # Describe
        prob.pretty_print(verbose=1)

        # Solve
        sols, info = exhaustive_enumerate(prob, max_solutions=500000,
                time_limit=10.0, verbose=False)

        # Show result info
        print("Enumeration info:", info)
        if sols:
            best = sols[0]
            best.pretty_print()
        else:
            print("No solution enumerated within time limit.")
        print("\n")

        type_I_results.append((sols, info))


    # Solve Type II
    print("Solving Type II tests with branch and bound\n")
    type_II_results = []
    for prob in type_II:
        # Describe
        prob.pretty_print(verbose=1)

        # Solve
        sol, info = branch_and_bound(prob, time_limit=30.0, verbose=False)

        # Show result info
        print("Enumeration info:", info)

        # Show results
        if sol:
            sol.pretty_print()
        else:
            print("No solution found within time limit.")
        print("\n")

        type_II_results.append((sol, info))

    return type_I_results, type_II_results

if __name__ == "__main__":
    demo_sprint_1()