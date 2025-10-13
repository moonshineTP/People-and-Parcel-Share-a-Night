import random

from share_a_ride.solvers.algo.exhaustive import exhaustive_enumerate
from share_a_ride.solvers.algo.bnb import branch_and_bound
from share_a_ride.solvers.algo.greedy import greedy_balanced_solver
from share_a_ride.utils.generator import generate_instance_coords, generate_instance_lazy

# ---------------------- Demo / main --------------------------------------------
def demo_sprint_1():
    # Prepare type I test instances
    type_I = []
    params_I = [(2,2,1), (2,3,2), (3,2,2), (2,3,2), (3,3,2)]
    for idx, (N, M, K) in enumerate(params_I):
        prob = generate_instance_coords(N, M, K, area=100, seed=100+idx)
        # prob = generate_instance_lazy(
        #     N, M, K, low=10, high=99, lmbd=20.0,
        #     use_poisson=True, seed=100+idx)
        type_I.append(prob)

    # Prepare type II test instances
    type_II = []
    params_II = [(3,4,2), (4,3,2), (2,6,2), (3,5,3), (4,5,3)]
    for idx, (N, M, K) in enumerate(params_II): 
        prob = generate_instance_coords(N, M, K, area=100, seed=200+idx)
        # prob = generate_instance_lazy(
        #     N, M, K, low=10, high=99, lmbd=20.0,
        #     use_poisson=True, seed=200+idx
        # )
        type_II.append(prob)


    # Solve Type I
    print("Solving Type I tests with brute-force enumerator:\n")
    type_I_results = []
    for prob in type_I:
        # Describe
        prob.pretty_print(verbose=1)

        # Solve
        sols, info = exhaustive_enumerate(prob, max_solutions=2000000,
                time_limit=10.0, verbose=False)

        # Show result info
        print("Enumeration info:", info)
        if sols:
            best = sols[0]
            best.stdin_print(verbose=1)
        else:
            print("No solution enumerated within time limit.")
        print("\n")

    # Solve Type II
    print("Solving Type II tests with branch and bound\n")
    type_II_results = []
    for prob in type_II:
        # Describe
        prob.pretty_print(verbose=1)

        # Solve
        sol, info = branch_and_bound(prob, time_limit=10.0, verbose=False)

        # Show result info
        print("Enumeration info:", info)

        # Show results
        if sol:
            sol.stdin_print(verbose=1)
        else:
            print("No solution found within time limit.")
        print("\n")

        type_II_results.append((sol, info))


def demo_sprint_2():
    pass

if __name__ == "__main__":
    demo_sprint_1()